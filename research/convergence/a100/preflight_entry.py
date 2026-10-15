#!/usr/bin/env python3
"""Per-entry preflight for the A100 matrix runner (runs before the train stage).

Checks, in order:
1. the pinned effective config resolves to the expected drafter algorithm and
   passes that algorithm's own ``validate_config``;
2. the target contract resolves for the run dir (``ensure_contract``);
3. the bound train store (``<run>/features``) carries the contract's capture
   layers, the contract's source identity, and — when the config asks for it —
   the target's final hidden states;
4. the bound val split (``<run>/data/val.jsonl``) loads through the same code
   path the evaluator uses and has at least ``--min-prompts`` prompts.

Writes ``<run-dir>/preflight.json`` (status ok/failed). Exit code 0 only when
every check passed; the runner parks an entry whose preflight fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _fail(run_dir: Path, payload: dict, message: str) -> int:
    payload["status"] = "failed"
    payload["error"] = message
    try:
        (run_dir / "preflight.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    except OSError:
        pass
    print(f"preflight: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expect-algorithm", default=None)
    parser.add_argument("--store", default=None, help="declared train store (must be bound at <run>/features)")
    parser.add_argument("--val", default=None, help="declared val split (must be bound at <run>/data/val.jsonl)")
    parser.add_argument("--min-prompts", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    run_dir = Path(args.run_dir)
    payload: dict = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(Path(args.config)),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "checks": [],
    }

    from vlm_dflash.algorithms import algorithm_name, get_algorithm
    from vlm_dflash.config import load_config
    from vlm_dflash.contract import contract_capture_layer_ids, ensure_contract
    from vlm_dflash.data.sharegpt import load_sharegpt
    from vlm_dflash.dflash.features import capture_layer_layout
    from vlm_dflash.provenance import assert_source_identity, identity_from_contract

    cfg = load_config(args.config)
    name = algorithm_name(cfg.get("algorithm"))
    payload["algorithm"] = name
    payload["expected_algorithm"] = args.expect_algorithm
    if args.expect_algorithm and name != args.expect_algorithm:
        return _fail(run_dir, payload, f"config resolves algorithm {name!r}, expected {args.expect_algorithm!r}")
    try:
        get_algorithm(cfg.get("algorithm")).validate_config(cfg)
    except Exception as exc:  # noqa: BLE001
        return _fail(run_dir, payload, f"algorithm.validate_config failed: {exc}")
    payload["checks"].append("algorithm_name+validate_config")

    contract = ensure_contract(cfg, run_dir)
    payload["contract_sha256"] = hashlib.sha256((run_dir / "contract.json").read_bytes()).hexdigest()
    payload["checks"].append("target_contract")

    if args.store:
        store = Path(args.store).resolve()
        bound = run_dir / "features"
        if not bound.exists():
            return _fail(run_dir, payload, f"train store not bound at {bound}")
        if Path(bound).resolve() != store:
            return _fail(run_dir, payload, f"{bound} resolves to {Path(bound).resolve()}, expected {store}")
        manifest_path = bound / "manifest.json"
        index_path = bound / "index.jsonl"
        if not manifest_path.exists() or not index_path.exists():
            return _fail(run_dir, payload, f"feature store incomplete: {bound}")
        manifest = json.loads(manifest_path.read_text())
        try:
            assert_source_identity(identity_from_contract(contract), manifest.get("identity"), "preflight vs store")
        except Exception as exc:  # noqa: BLE001
            return _fail(run_dir, payload, f"store identity: {exc}")
        contract_capture = contract_capture_layer_ids(contract)
        store_capture, _ = capture_layer_layout(
            manifest, contract_hidden_size=int(contract["text"]["hidden_size"]))
        if store_capture != contract_capture:
            return _fail(
                run_dir, payload,
                f"feature source holds layers {store_capture} but the contract declares {contract_capture}; "
                "set drafter.capture_layer_ids to the store's list (or recapture) before training")
        if bool((cfg.get("capture") or {}).get("store_final_hidden", False)) and not bool(
                manifest.get("stores_final_hidden", False)):
            return _fail(
                run_dir, payload,
                "config requires capture.store_final_hidden=true but the store does not carry the target's "
                "final hidden states (stores_final_hidden=false); recapture or point at a matching store")
        rows = sum(1 for line in index_path.read_text().splitlines() if line.strip())
        payload["store"] = {"path": str(store), "rows": rows,
                            "capture_layer_ids": list(store_capture),
                            "stores_final_hidden": bool(manifest.get("stores_final_hidden", False))}
        payload["checks"].append("store_identity+layers+final_hidden")

    if args.val:
        val = Path(args.val).resolve()
        bound = run_dir / "data" / "val.jsonl"
        if not bound.exists():
            return _fail(run_dir, payload, f"val split not bound at {bound}")
        if Path(bound).resolve() != val:
            return _fail(run_dir, payload, f"{bound} resolves to {Path(bound).resolve()}, expected {val}")
        try:
            rows, media_dir = load_sharegpt(bound)
        except Exception as exc:  # noqa: BLE001
            return _fail(run_dir, payload, f"val split unreadable through the eval loader: {exc}")
        if not media_dir.exists():
            return _fail(run_dir, payload, f"val media dir missing: {media_dir}")
        if args.min_prompts and len(rows) < args.min_prompts:
            return _fail(run_dir, payload,
                         f"val split has {len(rows)} prompts, need >= {args.min_prompts}")
        payload["val"] = {"path": str(val), "rows": len(rows), "media_dir": str(media_dir)}
        payload["checks"].append("val_split+media")

    payload["status"] = "ok"
    (run_dir / "preflight.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"preflight ok: algorithm={name} checks={payload['checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
