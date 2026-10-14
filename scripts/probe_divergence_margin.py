#!/usr/bin/env python
"""HF-assisted divergence diagnostic for vLLM DFlash token flips (v2, R5).

Replays the *frozen* target-only greedy prefix recorded by
``vllm_bridge_check.py`` through the repo's own HF target as one
teacher-forced forward and reports, at each first-divergence step, the
top-1/top-2 margin, the top-k candidate logits, and the rank/logit of the
token the vLLM dflash transcript emitted. Unlike the v1 probe this script
never regenerates its own prefix: the saved token ids are fed verbatim and
asserted to end up in the forward input.

Scope - this is an HF-assisted diagnostic, not same-engine evidence:

- it cannot observe vLLM-internal state (verify logits, per-request accepted
  positions, KV cache); those evidence fields are marked ``missing`` in the
  output and a same-engine (in-vLLM) protocol remains a GPU_PENDING item;
- small margins are therefore *observations*: they are consistent with
  near-tie resolution on the runtime verify path but do not establish it as
  the root cause of the bridge divergence, and the traced cases cannot be
  generalised to the untraced flips. The token-exact gate stays BLOCKED.

Causal gate (per case; all must hold for the case to count toward the causal
margin statistics):

1. the probe's prompt token length equals the vLLM-recorded ``prompt_len``;
2. image/token preprocessing is deterministic across two renders (digest
   equality, sha256 per processor output key);
3. the frozen prefix lies on the model's own greedy trajectory - every
   frozen prefix token is the argmax of the position predicting it.

Failing cases carry an ``exclusion_reason`` and are reported but excluded
from ``summary.causal_cases``.

    python scripts/probe_divergence_margin.py --run-dir /tmp/wp5/run_1d \\
        --bridge-json /tmp/wp5/bridge_check.json --max-cases 4 \\
        --out /tmp/wp5/divergence_margins_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.contract import ensure_contract  # noqa: E402
from vlm_dflash.eval.run import load_eval_rows, load_target  # noqa: E402
from vlm_dflash.target.prompts import encode_prompt  # noqa: E402
from vlm_dflash.utils import read_json, sha256_file, write_json  # noqa: E402

FORMAT = "vlm-dflash-divergence-margin-v2"

MISSING_VLLM_INPUTS = (
    "missing: bridge format v1 records no vLLM-side processor inputs or prompt token ids"
)
MISSING_SAME_ENGINE = "missing: requires in-engine (vLLM) instrumentation; GPU_PENDING"
MISSING_ACCEPTANCE_TRACE = "missing: vLLM reports aggregate spec_decode counters only"
MISSING_KV = "missing: not observable from an HF-side forward"


def _digest_value(value) -> str:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        blob = tensor.contiguous().numpy().tobytes()
    else:
        blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def inputs_digest(inputs: dict) -> dict:
    """Content hash per processor output key (preprocessing determinism)."""
    return {key: _digest_value(value) for key, value in sorted(inputs.items())}


def combined_digest(digest: dict) -> str:
    payload = json.dumps(digest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_cases(target_records: list[dict], draft_records: list[dict], max_cases: int) -> list[dict]:
    """First-divergence cases derived strictly from the saved bridge transcripts."""
    cases: list[dict] = []
    for index, (base, spec) in enumerate(zip(target_records, draft_records)):
        base_tokens = [int(token) for token in base["token_ids"]]
        draft_tokens = [int(token) for token in spec["token_ids"]]
        if base_tokens == draft_tokens:
            continue
        divergence = next(
            (position for position, (left, right) in enumerate(zip(base_tokens, draft_tokens)) if left != right),
            min(len(base_tokens), len(draft_tokens)),
        )
        cases.append(
            {
                "index": index,
                "request_id": base.get("request_id"),
                "prompt_len": int(base["prompt_len"]),
                "divergence": int(divergence),
                "prefix": base_tokens[:divergence],
                "dflash_token": draft_tokens[divergence] if divergence < len(draft_tokens) else None,
            }
        )
        if len(cases) >= max_cases:
            break
    return cases


def _evidence_marks() -> dict:
    return {
        "frozen_prefix_tokens": "available: fed verbatim from the bridge target-only transcript",
        "probe_input_digest": "available: sha256 per processor output key, two renders compared",
        "image_sha256": "available",
        "vllm_processor_inputs": MISSING_VLLM_INPUTS,
        "same_engine_verify_logits": MISSING_SAME_ENGINE,
        "per_request_acceptance_positions": MISSING_ACCEPTANCE_TRACE,
        "kv_cache_state": MISSING_KV,
    }


def probe_case(model, processor, row, media_dir, case: dict, *, device, top_k: int) -> dict:
    """Replay the frozen prefix for one case and apply the causal gate."""
    if top_k < 2:
        raise ValueError("--top-k must be >= 2 (a top1-top2 margin is required)")
    image_path = Path(media_dir) / row["images"][0]
    image = Image.open(image_path).convert("RGB")
    question = row["conversations"][0]["value"]
    first = encode_prompt(processor, question, image)
    second = encode_prompt(processor, question, image)
    digest_first = inputs_digest(first)
    deterministic = digest_first == inputs_digest(second)
    inputs = {key: value.to(device) for key, value in first.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    record = {
        "row_id": row.get("id"),
        "index": case["index"],
        "request_id": case["request_id"],
        "divergence": case["divergence"],
        "vllm_prompt_len": case["prompt_len"],
        "prompt_len": prompt_len,
        "prefix_len": len(case["prefix"]),
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path) if image_path.exists() else None,
        "input_digest": digest_first,
        "probe_input_sha256": combined_digest(digest_first),
        "input_preprocessing_deterministic": deterministic,
        "vllm_dflash_token": case["dflash_token"],
        "evidence": _evidence_marks(),
        "probe_position": None,
        "target_top1": None,
        "target_top1_logit": None,
        "margin_top1_top2": None,
        "candidates": None,
        "prefix_consistent": None,
        "first_prefix_mismatch": None,
        "vllm_dflash_logit": None,
        "vllm_token_rank": None,
        "logit_gap_top1_minus_vllm": None,
        "causal_evidence": False,
        "exclusion_reason": None,
    }

    reasons = []
    if prompt_len != case["prompt_len"]:
        reasons.append("prompt_len_mismatch")
    if not deterministic:
        reasons.append("input_preprocessing_not_deterministic")
    if reasons:
        record["exclusion_reason"] = ", ".join(reasons)
        return record

    prefix_tensor = torch.tensor([case["prefix"]], dtype=inputs["input_ids"].dtype, device=device)
    frozen_ids = torch.cat([inputs["input_ids"], prefix_tensor], dim=1)
    assert frozen_ids[0, prompt_len:].tolist() == list(case["prefix"]), "saved prefix not fed verbatim"
    forward_inputs = {key: value for key, value in inputs.items() if key != "input_ids"}
    forward_inputs["input_ids"] = frozen_ids
    mask = forward_inputs.get("attention_mask")
    if mask is not None and int(mask.shape[1]) == prompt_len:
        tail = torch.ones((mask.shape[0], len(case["prefix"])), dtype=mask.dtype, device=mask.device)
        forward_inputs["attention_mask"] = torch.cat([mask, tail], dim=1)

    with torch.inference_mode():
        outputs = model(**forward_inputs)
    logits = outputs.logits[0].float()

    first_mismatch = None
    for position, token in enumerate(case["prefix"]):
        if int(logits[prompt_len - 1 + position].argmax()) != int(token):
            first_mismatch = position
            break
    record["prefix_consistent"] = first_mismatch is None
    record["first_prefix_mismatch"] = first_mismatch

    probe_position = prompt_len + case["divergence"] - 1
    step_logits = logits[probe_position]
    top = torch.topk(step_logits, top_k)
    record["probe_position"] = probe_position
    record["target_top1"] = int(top.indices[0])
    record["target_top1_logit"] = float(top.values[0])
    record["margin_top1_top2"] = float(top.values[0] - top.values[1])
    record["candidates"] = [
        {"rank": rank + 1, "token_id": int(token), "logit": float(value)}
        for rank, (value, token) in enumerate(zip(top.values, top.indices))
    ]
    if case["dflash_token"] is not None:
        token_logit = float(step_logits[int(case["dflash_token"])])
        record["vllm_dflash_logit"] = token_logit
        record["vllm_token_rank"] = int((step_logits > token_logit).sum().item()) + 1
        record["logit_gap_top1_minus_vllm"] = float(top.values[0] - token_logit)

    if first_mismatch is not None:
        record["exclusion_reason"] = "prefix_not_on_model_trajectory"
    else:
        record["causal_evidence"] = True
    return record


def run_probe(
    model,
    processor,
    rows,
    media_dir,
    bridge: dict,
    *,
    max_cases: int = 4,
    top_k: int = 5,
    run_dir=None,
    bridge_path=None,
    contract: dict | None = None,
) -> dict:
    """Probe selection + case loop; returns the v2 output document."""
    dflash = bridge.get("dflash") or {}
    if not dflash.get("records"):
        raise ValueError("bridge json has no dflash phase (run vllm_bridge_check.py with --draft)")
    target_records = bridge["target_only"]["records"]
    cases = select_cases(target_records, dflash["records"], max_cases)
    print(f"probing {len(cases)} divergences: {[(case['index'], case['divergence']) for case in cases]}")
    device = next(model.parameters()).device
    results = [
        probe_case(model, processor, rows[case["index"]], media_dir, case, device=device, top_k=top_k)
        for case in cases
    ]
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key != "evidence"}, indent=2))
    spec = dflash.get("spec_decode") or {}
    preprocessing = None
    if contract is not None:
        image_bounds = contract.get("image", {})
        preprocessing = {
            "image_min_pixels": image_bounds.get("min_pixels"),
            "image_max_pixels": image_bounds.get("max_pixels"),
            "note": "must match the --min-pixels/--max-pixels passed to vllm_bridge_check.py",
        }
    return {
        "format": FORMAT,
        "evidence_class": "hf-assisted-diagnostic",
        "bridge_json": str(bridge_path) if bridge_path else None,
        "run_dir": str(run_dir) if run_dir else None,
        "preprocessing": preprocessing,
        "top_k": int(top_k),
        "cases": results,
        "summary": {
            "num_cases": len(results),
            "causal_cases": [result["index"] for result in results if result["causal_evidence"]],
            "excluded_cases": [
                {"index": result["index"], "reason": result["exclusion_reason"]}
                for result in results
                if not result["causal_evidence"]
            ],
            "run_level_per_position_acceptance": spec.get("per_position_acceptance", {"status": "missing"}),
            "attribution": {
                "level": "hf-assisted-diagnostic",
                "root_cause": "unresolved",
                "statement": (
                    "Margins are observations on replayed frozen prefixes. Small margins are consistent "
                    "with near-tie resolution on the runtime verify path but do not establish it as the "
                    "root cause of the vLLM bridge divergence, and traced cases do not generalise to the "
                    "untraced flips. The token-exact gate remains BLOCKED; same-engine evidence is missing."
                ),
            },
        },
        "same_engine_diagnostics": {
            "status": "missing",
            "required": [
                "target-only and dflash verify logits collected inside the same vLLM process/build",
                "per-request accepted/draft positions from the dflash proposer",
                "KV-cache / position metadata at the first divergence",
            ],
            "protocol": (
                "Instrument vLLM 0.22.1's DFlashProposer/verify path (or an equivalent same-engine dump) to "
                "record per-step candidate logits and acceptance positions for the same 32 prompts, then "
                "compare against this HF-side replay. Tracked as GPU_PENDING (review R5)."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="run dir with config.json/contract.json")
    parser.add_argument("--bridge-json", required=True, help="vllm_bridge_check.py result")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bridge_path = Path(args.bridge_json)
    bridge = read_json(bridge_path)
    cfg = read_json(run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    rows, media_dir = load_eval_rows(None, run_dir)
    rows = rows[: args.num_prompts]

    model, processor = load_target(cfg, contract)
    result = run_probe(
        model,
        processor,
        rows,
        media_dir,
        bridge,
        max_cases=args.max_cases,
        top_k=args.top_k,
        run_dir=run_dir,
        bridge_path=bridge_path,
        contract=contract,
    )
    write_json(Path(args.out), result)
    summary = result["summary"]
    print(
        f"causal cases: {summary['causal_cases']}; excluded: {summary['excluded_cases']}; "
        f"attribution: {summary['attribution']['root_cause']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
