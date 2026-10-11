"""Run the minimal acceptance evaluation for saved draft checkpoints."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import load_sharegpt, row_image_path, row_question
from ..dflash import draft_model as draft_model_module
from ..dflash.factory import load_checkpoint
from ..provenance import assert_identity, identity_from_contract, record_stage
from ..target.load import load_target
from ..target.prompts import encode_prompt
from ..utils import ensure_dir, log, write_json, write_jsonl
from .accept import summarize_samples
from .equivalence import sequences_equal


def _stops(model) -> list[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        raise RuntimeError("Target generation config has no eos_token_id.")
    return list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]


def evaluate(
    cfg: dict,
    run_dir: str | Path,
    contract: dict,
    num_prompts: int | None = None,
    max_new_tokens: int | None = None,
) -> dict:
    run_dir = Path(run_dir)
    eval_cfg = cfg.get("eval", {})
    num_prompts = int(num_prompts if num_prompts is not None else eval_cfg.get("num_prompts", 64))
    max_new_tokens = int(max_new_tokens if max_new_tokens is not None else eval_cfg.get("max_new_tokens", 128))
    checkpoint_names = list(eval_cfg.get("checkpoints", ["ckpt_final.pt", "ckpt_init.pt"]))
    sequential_verification = bool(eval_cfg.get("sequential_target_verification", True))
    verify_backend = str(eval_cfg.get("verify_sdpa_backend", "parity"))
    if verify_backend not in {"parity", "reference"}:
        raise ValueError(f"eval.verify_sdpa_backend must be 'parity' or 'reference', got {verify_backend!r}")
    # "parity" disables the reference verify-time SDPA backend narrowing so the
    # verification logits are computed with the same backend as generate()
    # (see docs/REFERENCE.md; bf16 near-ties otherwise flip argmax).
    draft_model_module.set_verify_sdpa_backend_switch(verify_backend == "reference")
    parallel_diagnostic_prompts = int(eval_cfg.get("parallel_diagnostic_prompts", 0))

    rows, media_dir = load_sharegpt(run_dir / "data" / "val.jsonl")
    rows = rows[:num_prompts]
    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    stop_ids = set(_stops(model))
    block_size = int(cfg["drafter"]["block_size"])
    dtype_name = str(cfg.get("target", {}).get("dtype", "bfloat16"))
    attn_name = str(cfg.get("target", {}).get("attn_implementation", "sdpa"))
    draft_dtype = getattr(torch, dtype_name, torch.bfloat16)

    log(f"eval: target-only greedy on {len(rows)} prompts")
    prompts = []
    for row in rows:
        image = Image.open(row_image_path(row, media_dir)).convert("RGB")
        inputs = encode_prompt(processor, row_question(row), image)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0
            )
        wall = time.perf_counter() - started
        prompt_len = int(inputs["input_ids"].shape[1])
        prompts.append(
            {
                "id": row.get("id"),
                "prompt_len": prompt_len,
                "tokens": generated[0, prompt_len:].tolist(),
                "wall_seconds": wall,
                "inputs": inputs,
            }
        )

    all_results, equivalence_runs, per_sample_rows = {}, {}, []
    for checkpoint_name in checkpoint_names:
        checkpoint_path = run_dir / "train" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing draft checkpoint {checkpoint_path}.")
        draft, meta = load_checkpoint(checkpoint_path)
        if meta.get("identity") is not None:
            assert_identity(identity_from_contract(contract), meta["identity"], f"eval vs {checkpoint_name}")
        draft.to(device=device, dtype=draft_dtype)
        draft.eval()
        samples, equivalent_flags = [], []
        log(f"eval: speculative greedy with {checkpoint_name}")
        for entry in prompts:
            inputs = entry["inputs"]
            target_kwargs = {
                "pixel_values": inputs.get("pixel_values"),
                "image_grid_thw": inputs.get("image_grid_thw"),
                "attention_mask": inputs["attention_mask"],
                "mm_token_type_ids": inputs.get("mm_token_type_ids"),
            }
            target_kwargs = {key: value for key, value in target_kwargs.items() if value is not None}
            started = time.perf_counter()
            with torch.inference_mode():
                output_ids, stats = draft.spec_generate(
                    target=model,
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=sorted(stop_ids),
                    temperature=0.0,
                    target_kwargs=target_kwargs,
                    return_stats=True,
                    sequential_target_verification=sequential_verification,
                )
            wall = time.perf_counter() - started
            prompt_len = entry["prompt_len"]
            spec_tokens = output_ids[0, prompt_len:].tolist()
            equivalent = sequences_equal(spec_tokens, entry["tokens"], stop_ids)
            equivalent_flags.append(equivalent)
            samples.append(
                {
                    "id": entry["id"],
                    "new_tokens": len(spec_tokens),
                    "verify_steps": int(stats["verify_steps"]),
                    "accepted_tokens": int(stats["accepted_tokens"]),
                    "draft_tokens": int(stats["draft_tokens"]),
                    "acceptance_events": stats["acceptance_events"],
                    "wall_seconds": wall,
                    "target_verify_seconds": stats.get("target_verify_seconds", 0.0),
                    "target_prefill_seconds": stats.get("target_prefill_seconds", 0.0),
                    "draft_seconds": stats.get("draft_seconds", 0.0),
                }
            )
            per_sample_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "id": entry["id"],
                    "equivalent": equivalent,
                    "new_tokens": len(spec_tokens),
                    "target_new_tokens": len(entry["tokens"]),
                    "spec_tokens": spec_tokens,
                    "target_tokens": entry["tokens"],
                }
            )
        summary = summarize_samples(samples, block_size)
        summary["role"] = (
            "acceptance baseline (random-init drafter)" if "init" in checkpoint_name else "trained drafter"
        )
        all_results[checkpoint_name] = summary
        equivalence_runs[checkpoint_name] = {
            "num_prompts": len(equivalent_flags),
            "token_exact": bool(all(equivalent_flags)),
            "failed_ids": [
                entry["id"] for entry, ok in zip(prompts, equivalent_flags) if not ok
            ],
        }

    parallel_diagnostic = None
    if parallel_diagnostic_prompts > 0:
        parallel_diagnostic = _run_parallel_diagnostic(
            model, draft_model_module, run_dir, contract, cfg, prompts, stop_ids,
            max_new_tokens, block_size, parallel_diagnostic_prompts,
        )

    result = {
        "format": "vlm-dflash-eval-v1",
        "identity": identity_from_contract(contract),
        "al_definition": "AL = new_tokens / target_verification_steps (bonus-token-inclusive tokens per target forward)",
        "settings": {
            "num_prompts": len(rows),
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "dtype": dtype_name,
            "attn_impl": attn_name,
            "block_size": block_size,
        },
        "equivalence_check": {
            "num_prompts": len(rows),
            "token_exact": all(run["token_exact"] for run in equivalence_runs.values()),
            "runs": equivalence_runs,
            "dtype": dtype_name,
            "attn_impl": attn_name,
            "max_new_tokens": max_new_tokens,
            "sequential_target_verification": sequential_verification,
            "verify_sdpa_backend": verify_backend,
            "note": (
                "Parity mode (sequential verification + generate()-backend logits) reproduces "
                "target-only greedy exactly. Parallel block verification can flip bf16 near-tie "
                "argmax decisions (see parallel_diagnostic)."
            ),
        },
        "parallel_diagnostic": parallel_diagnostic,
        "runs": all_results,
        "target_only": {
            "role": "output/speed reference (no drafter; AL is undefined without draft proposals)",
            "mean_new_tokens": sum(len(entry["tokens"]) for entry in prompts) / len(prompts),
            "wall_seconds_per_sample": sum(entry["wall_seconds"] for entry in prompts) / len(prompts),
        },
    }
    eval_dir = ensure_dir(run_dir / "eval")
    write_json(eval_dir / "eval_result.json", result)
    write_jsonl(eval_dir / "eval_samples.jsonl", per_sample_rows)
    record_stage(
        run_dir,
        "eval",
        {"identity": identity_from_contract(contract), "result": result, "code": "vlm_dflash.eval.run.evaluate"},
    )
    return result

def _run_parallel_diagnostic(
    model,
    draft_model_module,
    run_dir,
    contract,
    cfg,
    prompts,
    stop_ids,
    max_new_tokens,
    block_size,
    num_prompts,
):
    """Measure parallel-verify equivalence plus the target-logit margin at mismatches."""
    from .equivalence import cut_at_first_stop

    draft, _ = load_checkpoint(run_dir / "train" / eval_cfg_checkpoint(cfg))
    draft.to(model.device, dtype=torch.bfloat16)
    draft.eval()
    mismatches = []
    exact = 0
    draft_model_module.set_verify_sdpa_backend_switch(True)
    try:
        for entry in prompts[:num_prompts]:
            inputs = entry["inputs"]
            target_kwargs = {key: inputs[key] for key in ("pixel_values", "image_grid_thw", "attention_mask", "mm_token_type_ids") if key in inputs}
            scores = None
            with torch.inference_mode():
                scored = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    repetition_penalty=1.0, output_scores=True, return_dict_in_generate=True,
                )
                scores = scored.scores
                reference = scored.sequences[0, entry["prompt_len"]:].tolist()
                output_ids = draft.spec_generate(
                    target=model,
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=sorted(stop_ids),
                    temperature=0.0,
                    target_kwargs=target_kwargs,
                    return_stats=False,
                    sequential_target_verification=False,
                )
            spec = output_ids[0, entry["prompt_len"]:].tolist()
            spec_cut = cut_at_first_stop(spec, stop_ids)
            ref_cut = cut_at_first_stop(reference, stop_ids)
            if spec_cut == ref_cut:
                exact += 1
                continue
            divergence = next(
                (index for index, (a, b) in enumerate(zip(spec_cut, ref_cut)) if a != b),
                min(len(spec_cut), len(ref_cut)) - 1,
            )
            margin = None
            if scores is not None and 0 <= divergence < len(scores):
                top2 = torch.topk(scores[divergence][0].float(), 2).values
                margin = float(top2[0] - top2[1])
            mismatches.append(
                {"id": entry["id"], "first_divergence": divergence, "target_top2_margin": margin}
            )
    finally:
        draft_model_module.set_verify_sdpa_backend_switch(False)
    return {
        "num_prompts": min(num_prompts, len(prompts)),
        "token_exact": exact,
        "token_exact_rate": exact / max(1, min(num_prompts, len(prompts))),
        "mismatches": mismatches,
        "note": "Reference verify backend, parallel block verification; target_top2_margin is the logit gap at the first divergent step.",
    }


def eval_cfg_checkpoint(cfg):
    return list(cfg.get("eval", {}).get("checkpoints", ["ckpt_final.pt"]))[0]
