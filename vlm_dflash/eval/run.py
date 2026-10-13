"""Run the minimal acceptance evaluation for saved draft checkpoints."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

from ..algorithms import get_algorithm
from ..data.sharegpt import load_sharegpt, row_image_path, row_question
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


def _target_head_is_tied(model) -> bool:
    """Whether the target's output head is the input embedding matrix."""
    embeddings = model.get_input_embeddings()
    weight = getattr(getattr(model, "lm_head", None), "weight", None)
    return weight is not None and embeddings is not None and weight.data_ptr() == embeddings.weight.data_ptr()


def check_checkpoint_head(checkpoint_name: str, meta: dict, target_tied_head: bool) -> None:
    """Refuse a checkpoint whose training head does not match the target's head.

    A draft trained against the embedding-tied head on an untied target samples
    its logits through a different projection at eval time, which silently
    zeroes acceptance instead of reporting an error.
    """
    head_mode = meta.get("head", "tied_embedding")
    if not target_tied_head and head_mode != "target_lm_head":
        raise RuntimeError(
            f"{checkpoint_name} was trained against the embedding-tied head (head={head_mode!r}) but this "
            "target's lm_head is untied: the draft logits would be sampled through a different head than "
            "training used, so acceptance is meaningless. Re-run capture_features and retrain."
        )


def checkpoint_names_from_config(eval_cfg: dict) -> list[str]:
    """Checkpoint names to evaluate; an empty list would silently pass the gate."""
    names = [str(name) for name in eval_cfg.get("checkpoints", ["ckpt_final.pt", "ckpt_init.pt"])]
    if not names:
        raise ValueError("eval.checkpoints is empty; nothing would be evaluated.")
    return names


def assert_all_checkpoints_evaluated(equivalence_runs: dict, checkpoint_names: list[str]) -> None:
    """Refuse to report a token-exact pass for checkpoints that were never run."""
    if len(equivalence_runs) != len(checkpoint_names):
        raise RuntimeError(
            f"evaluated {len(equivalence_runs)} of {len(checkpoint_names)} checkpoints; "
            "refusing to write a partial eval result."
        )


def check_checkpoint_algorithm(checkpoint_name: str, meta: dict, algorithm) -> None:
    """Refuse a checkpoint trained by a different drafter algorithm.

    Scoring DFlash weights through a DSpark decoder (or vice versa) is
    meaningless; the recorded algorithm tag makes that a hard error instead of
    a silent acceptance collapse. Checkpoints written before the algorithm tag
    existed default to ``dflash``.
    """
    recorded = str(meta.get("algorithm", "dflash"))
    if recorded != algorithm.name:
        raise RuntimeError(
            f"{checkpoint_name} was trained with algorithm={recorded!r}, but this eval resolves "
            f"algorithm={algorithm.name!r}. Re-run eval with the matching algorithm or retrain."
        )


def check_checkpoint_feature_mode(checkpoint_name: str, meta: dict) -> str:
    """Feature source the checkpoint was trained from (legacy tag -> offline).

    Online checkpoints are scored by the same loader as offline ones; the tag is
    surfaced in the eval result so a report can never silently attribute an
    online-trained checkpoint to the staged pipeline (or vice versa).
    """
    mode = str(meta.get("feature_mode", "offline")).lower()
    if mode not in ("offline", "online"):
        raise RuntimeError(f"{checkpoint_name} records an unknown feature_mode={mode!r}.")
    return mode


def preflight_checkpoints(
    run_dir: Path,
    checkpoint_names: list[str],
    contract: dict,
    target_tied_head: bool,
    algorithm,
) -> dict[str, str]:
    """Validate every checkpoint's provenance before the expensive generation pass.

    Returns the recorded feature mode per checkpoint (offline for checkpoints
    written before the tag existed).
    """
    modes: dict[str, str] = {}
    for checkpoint_name in checkpoint_names:
        checkpoint_path = run_dir / "train" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing draft checkpoint {checkpoint_path}.")
        meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if meta.get("identity") is not None:
            assert_identity(identity_from_contract(contract), meta["identity"], f"eval vs {checkpoint_name}")
        check_checkpoint_head(checkpoint_name, meta, target_tied_head)
        check_checkpoint_algorithm(checkpoint_name, meta, algorithm)
        modes[checkpoint_name] = check_checkpoint_feature_mode(checkpoint_name, meta)
    return modes


def load_eval_rows(prompt_source: dict | None, run_dir: Path, default_split: str = "val.jsonl"):
    """Rows + media dir for eval: the run's held-out split by default.

    ``prompt_source`` points the evaluation at a fixed prompt set instead of
    the run's own split (used by the benchmark harness):

    - ``{"type": "run_val", "split": "val.jsonl"}`` - the run's own split;
    - ``{"type": "sharegpt_jsonl", "path": ..., "meta": ...}`` - an external
      ShareGPT-style file (paths resolve against the run dir when relative;
      environment variables are expanded).
    """
    if not prompt_source:
        return load_sharegpt(run_dir / "data" / default_split)
    source = dict(prompt_source)
    kind = str(source.get("type", "sharegpt_jsonl"))
    if kind == "run_val":
        split = str(source.get("split") or default_split)
        return load_sharegpt(run_dir / "data" / split)
    if kind == "sharegpt_jsonl":
        path_text = source.get("path")
        if not path_text:
            raise ValueError("prompt_source type=sharegpt_jsonl requires a 'path'.")
        path = Path(os.path.expandvars(str(path_text))).expanduser()
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        meta_path = None
        if source.get("meta"):
            meta_path = Path(os.path.expandvars(str(source["meta"]))).expanduser()
            if not meta_path.is_absolute():
                meta_path = (run_dir / meta_path).resolve()
        return load_sharegpt(path, meta_path=meta_path)
    raise ValueError(f"Unknown prompt_source type {kind!r} (expected 'run_val' or 'sharegpt_jsonl').")


def prepare_prompts(model, processor, rows, media_dir, *, max_new_tokens: int, device, label: str = "eval") -> list[dict]:
    """Target-only greedy anchors: the equivalence reference for each row."""
    log(f"{label}: target-only greedy on {len(rows)} prompts")
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
    return prompts


def run_checkpoints(
    model,
    run_dir: Path,
    cfg: dict,
    contract: dict,
    prompts: list[dict],
    *,
    algorithm,
    checkpoint_names: list[str],
    verification: str,
    strict_parallel_margin: float | None,
    verify_backend: str,
    block_size: int,
    draft_dtype,
    max_new_tokens: int,
    parallel_diagnostic_prompts: int = 0,
    stop_ids: set | None = None,
    label: str = "eval",
) -> tuple[dict, dict, list, dict | None]:
    """Speculative-decode every checkpoint against the prepared target anchors.

    Shared by ``evaluate`` and the benchmark harness so both measure through
    the same verification path (sequential by default; parallel/strict are
    opt-in and never mixed into correctness tables).
    """
    device = next(model.parameters()).device
    if stop_ids is None:
        stop_ids = set(_stops(model))
    sequential_verification = verification == "sequential"
    target_only_wall = sum(entry["wall_seconds"] for entry in prompts) / max(1, len(prompts))
    # "parity" disables the reference verify-time SDPA backend narrowing so the
    # verification logits are computed with the same backend as generate()
    # (see docs/REFERENCE.md; bf16 near-ties otherwise flip argmax).
    algorithm.configure_verify_backend(verify_backend == "reference")
    all_results, equivalence_runs, per_sample_rows = {}, {}, []
    for checkpoint_name in checkpoint_names:
        draft, _ = algorithm.load_decoder(run_dir / "train" / checkpoint_name)
        draft.to(device=device, dtype=draft_dtype)
        draft.eval()
        samples, equivalent_flags = [], []
        strict_rounds = 0.0
        strict_fallbacks = 0.0
        strict_min_margin = None
        log(f"{label}: speculative greedy with {checkpoint_name} ({verification})")
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
                    strict_parallel_margin=strict_parallel_margin,
                )
            wall = time.perf_counter() - started
            strict_rounds += float(stats.get("strict_parallel_rounds", 0.0) or 0.0)
            strict_fallbacks += float(stats.get("strict_fallback_rounds", 0.0) or 0.0)
            round_min_margin = stats.get("strict_min_decision_margin")
            if round_min_margin is not None:
                strict_min_margin = (
                    round_min_margin if strict_min_margin is None else min(strict_min_margin, round_min_margin)
                )
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
        # Serving-time in-block attention actually used by the draft forward
        # (JetSpec checkpoints serve the causal mask they were trained with).
        summary["draft_inference_in_block_causal"] = bool(
            getattr(draft, "inference_in_block_causal", False)
        )
        # Whether the served draft reranks its proposals with the DFlash2
        # candidate selector (recorded so a summary alone says which runtime path
        # produced the numbers).
        summary["draft_candidate_selector"] = bool(hasattr(draft, "candidate_selector"))
        sample_wall = summary.get("wall_seconds_per_sample") or 0.0
        summary["speedup_vs_target_only"] = (target_only_wall / sample_wall) if sample_wall > 0 else None
        if strict_parallel_margin is not None:
            summary["strict_parallel"] = {
                "margin": strict_parallel_margin,
                "rounds": strict_rounds,
                "fallbacks": strict_fallbacks,
                "fallback_rate": (strict_fallbacks / strict_rounds) if strict_rounds else None,
                "min_decision_margin": strict_min_margin,
            }
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
            model, algorithm, run_dir, contract, cfg, prompts, stop_ids,
            max_new_tokens, block_size, parallel_diagnostic_prompts,
        )
    return all_results, equivalence_runs, per_sample_rows, parallel_diagnostic


def evaluate(
    cfg: dict,
    run_dir: str | Path,
    contract: dict,
    num_prompts: int | None = None,
    max_new_tokens: int | None = None,
    tag: str | None = None,
    prompt_source: dict | None = None,
    workload: str | None = None,
) -> dict:
    run_dir = Path(run_dir)
    eval_cfg = cfg.get("eval", {})
    num_prompts = int(num_prompts if num_prompts is not None else eval_cfg.get("num_prompts", 64))
    max_new_tokens = int(max_new_tokens if max_new_tokens is not None else eval_cfg.get("max_new_tokens", 128))
    checkpoint_names = checkpoint_names_from_config(eval_cfg)
    verification = str(eval_cfg.get("verification") or "").strip().lower()
    if not verification:
        verification = "sequential" if bool(eval_cfg.get("sequential_target_verification", True)) else "parallel"
    if verification not in {"sequential", "parallel", "strict_parallel"}:
        raise ValueError(
            "eval.verification must be one of 'sequential', 'parallel' or 'strict_parallel', "
            f"got {verification!r}"
        )
    sequential_verification = verification == "sequential"
    strict_parallel_margin = (
        float(eval_cfg.get("strict_parallel_margin", 2.0)) if verification == "strict_parallel" else None
    )
    verify_backend = str(eval_cfg.get("verify_sdpa_backend", "parity"))
    if verify_backend not in {"parity", "reference"}:
        raise ValueError(f"eval.verify_sdpa_backend must be 'parity' or 'reference', got {verify_backend!r}")
    parallel_diagnostic_prompts = int(eval_cfg.get("parallel_diagnostic_prompts", 0))

    algorithm = get_algorithm(cfg.get("algorithm"))
    rows, media_dir = load_eval_rows(prompt_source, run_dir)
    rows = rows[:num_prompts]
    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    target_tied_head = _target_head_is_tied(model)
    contract_tied = contract.get("head", {}).get("tie_word_embeddings")
    if contract_tied is not None and bool(contract_tied) != target_tied_head:
        raise RuntimeError(
            "Loaded target head disagrees with the resolved contract: "
            f"runtime tie_word_embeddings={target_tied_head}, contract={bool(contract_tied)}."
        )
    stop_ids = set(_stops(model))
    block_size = int(cfg["drafter"]["block_size"])
    dtype_name = str(cfg.get("target", {}).get("dtype", "bfloat16"))
    attn_name = str(cfg.get("target", {}).get("attn_implementation", "sdpa"))
    draft_dtype = getattr(torch, dtype_name, torch.bfloat16)
    checkpoint_feature_modes = preflight_checkpoints(
        run_dir, checkpoint_names, contract, target_tied_head, algorithm
    )

    prompts = prepare_prompts(
        model, processor, rows, media_dir, max_new_tokens=max_new_tokens, device=device, label="eval"
    )
    target_only_wall = sum(entry["wall_seconds"] for entry in prompts) / max(1, len(prompts))

    all_results, equivalence_runs, per_sample_rows, parallel_diagnostic = run_checkpoints(
        model,
        run_dir,
        cfg,
        contract,
        prompts,
        algorithm=algorithm,
        checkpoint_names=checkpoint_names,
        verification=verification,
        strict_parallel_margin=strict_parallel_margin,
        verify_backend=verify_backend,
        block_size=block_size,
        draft_dtype=draft_dtype,
        max_new_tokens=max_new_tokens,
        parallel_diagnostic_prompts=parallel_diagnostic_prompts,
        stop_ids=stop_ids,
        label="eval",
    )

    assert_all_checkpoints_evaluated(equivalence_runs, checkpoint_names)

    result = {
        "format": "vlm-dflash-eval-v1",
        "identity": identity_from_contract(contract),
        "algorithm": algorithm.name,
        # Feature source recorded by the trainer per checkpoint ("offline" for
        # checkpoints older than the tag). Training/feature provenance only:
        # eval itself always uses the live target.
        "checkpoint_feature_modes": checkpoint_feature_modes,
        "al_definition": "AL = new_tokens / target_verification_steps (bonus-token-inclusive tokens per target forward)",
        "settings": {
            "num_prompts": len(rows),
            "workload": workload,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "dtype": dtype_name,
            "attn_impl": attn_name,
            "block_size": block_size,
            "target_tie_word_embeddings": target_tied_head,
        },
        "equivalence_check": {
            "num_prompts": len(rows),
            "token_exact": bool(equivalence_runs) and all(run["token_exact"] for run in equivalence_runs.values()),
            "runs": equivalence_runs,
            "dtype": dtype_name,
            "attn_impl": attn_name,
            "max_new_tokens": max_new_tokens,
            "verification": verification,
            "sequential_target_verification": sequential_verification,
            "strict_parallel_margin": strict_parallel_margin,
            "verify_sdpa_backend": verify_backend,
            "verify_backend_switch_enabled": _effective_verify_backend_switch(),
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
            "wall_seconds_per_sample": target_only_wall,
            "decode_tokens_per_second": (
                sum(len(entry["tokens"]) for entry in prompts) / sum(entry["wall_seconds"] for entry in prompts)
                if sum(entry["wall_seconds"] for entry in prompts) > 0
                else None
            ),
        },
    }
    eval_dir = ensure_dir(run_dir / "eval")
    suffix = f"_{tag}" if tag else ""
    write_json(eval_dir / f"eval_result{suffix}.json", result)
    write_jsonl(eval_dir / f"eval_samples{suffix}.jsonl", per_sample_rows)
    record_stage(
        run_dir,
        "eval",
        {"identity": identity_from_contract(contract), "result": result, "code": "vlm_dflash.eval.run.evaluate"},
    )
    if strict_parallel_margin is not None and not result["equivalence_check"]["token_exact"]:
        failed = {name: run["failed_ids"] for name, run in equivalence_runs.items() if not run["token_exact"]}
        raise RuntimeError(
            "strict_parallel verification was expected to reproduce target-only greedy exactly, but "
            f"these prompts diverged: {failed}. Raise eval.strict_parallel_margin (current "
            f"{strict_parallel_margin}) or inspect eval/eval_result.json (written)."
        )
    return result

def _run_parallel_diagnostic(
    model,
    algorithm,
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

    draft, _ = algorithm.load_decoder(run_dir / "train" / eval_cfg_checkpoint(cfg))
    draft.to(model.device, dtype=torch.bfloat16)
    draft.eval()
    mismatches = []
    exact = 0
    algorithm.configure_verify_backend(True)
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
        algorithm.configure_verify_backend(False)
    return {
        "num_prompts": min(num_prompts, len(prompts)),
        "token_exact": exact,
        "token_exact_rate": exact / max(1, min(num_prompts, len(prompts))),
        "mismatches": mismatches,
        "note": "Reference verify backend, parallel block verification; target_top2_margin is the logit gap at the first divergent step.",
    }


def _effective_verify_backend_switch() -> bool | None:
    """Effective state of the shared verify-time SDPA narrowing switch.

    Recorded in ``eval_result.json`` so a plugin that fails to forward
    ``configure_verify_backend`` becomes visible in the artifact instead of
    silently verifying under reference numerics while the run claims
    ``verify_sdpa_backend: parity`` (the 2026-10-12 DSpark near-tie gate
    failure).
    """
    try:
        from ..dflash.draft_model import verify_sdpa_backend_switch_enabled
    except Exception:
        return None
    return bool(verify_sdpa_backend_switch_enabled())


def eval_cfg_checkpoint(cfg):
    return list(cfg.get("eval", {}).get("checkpoints", ["ckpt_final.pt"]))[0]
