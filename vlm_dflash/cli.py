"""Unified command line: ``vlm-dflash {doctor,train,eval,export}``.

The five stage scripts remain the advanced interface; the CLI is the happy
path: ``doctor`` preflights an environment/target, ``train`` runs the whole
staged pipeline (prepare -> regen -> capture -> train) from a target + dataset
with sensible presets, ``eval`` scores a finished run, and ``export`` writes
the self-describing portable checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .config import apply_overrides, dump_effective_config, load_config, resolve_run_dir

_PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"
_PRESET_NAMES = ("smoke", "single-gpu-24g", "single-gpu-48g")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_preset(name_or_path: str) -> dict:
    candidate = Path(name_or_path)
    if not candidate.exists():
        candidate = _PRESETS_DIR / f"{name_or_path}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Preset {name_or_path!r} not found (looked for {name_or_path}.yaml in {_PRESETS_DIR}); "
            f"known presets: {', '.join(_PRESET_NAMES)}."
        )
    return load_config(candidate)


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _slug(text: str) -> str:
    stem = Path(str(text)).name if not str(text).startswith(("Qwen/", "Qwen3")) else str(text).split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-").lower()


def _load_run_config(run_dir: Path, config_path: str | None) -> dict:
    if config_path:
        cfg = load_config(config_path)
        cfg.setdefault("run", {})
        cfg["run"]["dir"] = str(run_dir)
        return cfg
    effective = run_dir / "config.json"
    if not effective.exists():
        raise FileNotFoundError(
            f"{run_dir} has no config.json (was it created by this CLI?); pass --config explicitly."
        )
    with open(effective, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

def _parse_constraints() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "constraints-tested.txt"
    pins: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                pins[name.strip().lower()] = version.strip()
    return pins


def _version_tuple(value: str) -> tuple[int, ...]:
    base = value.split("+", 1)[0]
    parts = []
    for piece in base.split("."):
        match = re.match(r"(\d+)", piece)
        parts.append(int(match.group(1)) if match else 0)
    return tuple(parts)


def _package_version(name: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:
        return None


def collect_doctor_report(target: str | None, mask_token_id: int | None, samples: int, seq_length: int) -> dict:
    report: dict = {"ok": True, "checks": [], "warnings": [], "errors": []}

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["ok"] = False
            report["errors"].append(f"{name}: {detail}")

    report["python"] = sys.version.split()[0]
    check("python>=3.10", sys.version_info >= (3, 10), report["python"])

    pins = _parse_constraints()
    for package in ("torch", "transformers", "tokenizers", "torchvision", "safetensors"):
        version = _package_version(package)
        if version is None:
            if package == "torchvision":
                check(f"{package} importable", False, "missing (required by Qwen-VL processors)")
            continue
        pinned = pins.get(package)
        if pinned is None:
            continue
        same = _version_tuple(version)[:2] == _version_tuple(pinned)[:2]
        detail = f"{version} (tested: {pinned})"
        if same:
            report["checks"].append({"name": f"{package} version", "ok": True, "detail": detail})
        else:
            report["warnings"].append(f"{package} {version} differs from tested {pinned}")

    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            devices = []
            for index in range(count):
                free, total = torch.cuda.mem_get_info(index)
                devices.append(
                    {"index": index, "name": torch.cuda.get_device_name(index), "free_gb": round(free / 1e9, 1), "total_gb": round(total / 1e9, 1)}
                )
            report["cuda"] = {"available": True, "devices": devices}
        else:
            report["cuda"] = {"available": False}
            check("cuda available", False, "torch.cuda.is_available() is False")
    except Exception as error:  # pragma: no cover - environment dependent
        check("torch import", False, str(error))

    if target:
        try:
            from .adapters import adapter_from_config_path

            adapter = adapter_from_config_path(target)
            report["adapter"] = {"name": adapter.info.name, "family": adapter.info.family, "notes": adapter.info.notes}
            from .contract import resolve_contract

            defaults = {"mask_token_id": mask_token_id} if mask_token_id is not None else {}
            contract = resolve_contract(target, defaults, out_path=None)
            report["contract"] = {
                "model_type": contract["target"]["model_type"],
                "model_fingerprint": contract["target"]["model_fingerprint"],
                "mask_token_id": contract["mask"]["token_id"],
                "mrope_sections": contract["rope"]["sections"],
                "draft_layer_ids": contract["layers"]["draft_layer_ids"],
                "image_pixels": [contract["image"]["min_pixels"], contract["image"]["max_pixels"]],
                "num_target_layers": contract["text"]["num_target_layers"],
                "hidden_size": contract["text"]["hidden_size"],
                "tie_word_embeddings": contract.get("head", {}).get("tie_word_embeddings"),
            }
            hidden = int(contract["text"]["hidden_size"])
            layers = int(contract["layers"]["num_draft_layers"])
            per_sample_gb = layers * hidden * seq_length * 2 / 1e9
            report["estimates"] = {
                "feature_disk_gb_per_1k_samples": round(per_sample_gb * samples, 2),
                "seq_length": seq_length,
                "dtype_bytes": 2,
            }
            weights = sum(item.stat().st_size for item in Path(target).glob("*.safetensors"))
            if weights:
                report["estimates"]["target_weights_gb"] = round(weights / 1e9, 2)
        except Exception as error:
            check("target contract", False, f"{type(error).__name__}: {error}")

    return report


def cmd_doctor(args) -> int:
    report = collect_doctor_report(args.target, args.mask_token_id, args.samples, args.seq_length)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"vlm-dflash doctor — python {report['python']}, ok={report['ok']}")
        for entry in report["checks"]:
            print(f"  [{'ok' if entry['ok'] else 'FAIL'}] {entry['name']}: {entry['detail']}")
        for warning in report["warnings"]:
            print(f"  [warn] {warning}")
        for error in report["errors"]:
            print(f"  [error] {error}")
        if "adapter" in report:
            print(f"  adapter: {report['adapter']['name']} ({report['adapter']['family']})")
        if "contract" in report:
            contract = report["contract"]
            print(
                "  contract: mask={mask_token_id} mrope={mrope_sections} layers={draft_layer_ids} "
                "pixels={image_pixels} depth={num_target_layers} hidden={hidden_size} "
                "tied_head={tie_word_embeddings}".format(**contract)
            )
        if "estimates" in report:
            print(f"  estimates: {report['estimates']}")
    return 0 if report["ok"] else 2


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def build_train_config(args) -> dict:
    cfg = load_preset(args.preset)
    if args.config:
        cfg = deep_merge(cfg, load_config(args.config))
    if args.target:
        cfg.setdefault("target", {})["model"] = args.target
    if not cfg.get("target", {}).get("model"):
        raise SystemExit("train: --target (local model directory) or --config with target.model is required.")

    dataset = cfg.setdefault("dataset", {})
    if args.dataset:
        path = Path(args.dataset)
        if path.suffix != ".jsonl":
            raise SystemExit(f"--dataset expects a .jsonl file, got {args.dataset!r}; use --dataset-source for other sources.")
        dataset.update({"source": "jsonl", "path": str(path)})
    if args.hf_dataset:
        dataset.update({"source": "hf", "hf": {"path": args.hf_dataset}})
        if args.hf_split:
            dataset["hf"]["split"] = args.hf_split
    if args.dataset_source:
        dataset["source"] = args.dataset_source
        if args.annotation:
            dataset["annotation"] = args.annotation
    if args.images_dir:
        dataset["images_dir"] = args.images_dir
        dataset.setdefault("jsonl_images_dir", args.images_dir)
    if args.limit is not None:
        dataset["slice"] = int(args.limit)
    if not dataset.get("source"):
        raise SystemExit(
            "train: pass --dataset <file.jsonl>, --hf-dataset <id> or --config with a dataset block "
            "(the smoke preset ships its own COCO slice)."
        )
    run = cfg.setdefault("run", {})
    if not run.get("dir"):
        name = f"{_slug(cfg['target']['model'])}-dflash"
        run.update({"name": name, "dir": f"runs/{name}"})
    return apply_overrides(cfg, args.set_list)


def _stage_present(run_dir: Path, relative: str) -> bool:
    return (run_dir / relative).exists()


def cmd_train(args) -> int:
    from .contract import ensure_contract
    from .utils import log

    cfg = build_train_config(args)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and not (run_dir / "train" / "ckpt_last.pt").exists():
        raise SystemExit(f"train --resume: no checkpoint at {run_dir / 'train' / 'ckpt_last.pt'} to resume from.")

    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)

    limit = args.limit

    if args.force or not _stage_present(run_dir, "data/data.jsonl"):
        from .data.prepare import build_dataset

        build_dataset(cfg, run_dir / "data", limit=limit)
    else:
        log("train: data stage present, skipping (use --force to rebuild)")

    if args.force or not _stage_present(run_dir, "regen/target_data.jsonl"):
        from .target.regen import run_regen

        run_regen(cfg, run_dir, contract, limit=limit)
    else:
        log("train: regen stage present, skipping (use --force to rerun)")

    if args.force or not _stage_present(run_dir, "features/index.jsonl"):
        from .target.capture import run_capture

        run_capture(cfg, run_dir, contract, limit=limit)
    else:
        log("train: capture stage present, skipping (use --force to recapture)")

    from .dflash.train import train_dflash

    summary = train_dflash(cfg, run_dir, contract, steps=args.steps, limit=limit, resume=args.resume)
    log(f"train: done with steps={summary['steps']} final_loss={summary['final_loss']}")
    return 0


# --------------------------------------------------------------------------- #
# eval / export
# --------------------------------------------------------------------------- #

def cmd_eval(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    cfg = _load_run_config(run_dir, args.config)
    eval_cfg = cfg.setdefault("eval", {})
    if args.num_prompts is not None:
        eval_cfg["num_prompts"] = int(args.num_prompts)
    if args.max_new_tokens is not None:
        eval_cfg["max_new_tokens"] = int(args.max_new_tokens)
    if args.checkpoints:
        eval_cfg["checkpoints"] = [item for item in args.checkpoints.split(",") if item]
    if args.parallel_diagnostic is not None:
        eval_cfg["parallel_diagnostic_prompts"] = int(args.parallel_diagnostic)
    if args.verification is not None:
        eval_cfg["verification"] = args.verification
    if args.strict_margin is not None:
        eval_cfg["strict_parallel_margin"] = float(args.strict_margin)

    from .contract import ensure_contract
    from .eval.run import evaluate

    contract = ensure_contract(cfg, run_dir)
    result = evaluate(
        cfg,
        run_dir,
        contract,
        num_prompts=eval_cfg.get("num_prompts"),
        max_new_tokens=eval_cfg.get("max_new_tokens"),
        tag=args.tag,
    )
    for name, run in result["runs"].items():
        print(
            f"{name}: AL={run['mean_al']:.3f} speedup_vs_target_only="
            f"{run.get('speedup_vs_target_only')} token_exact={result['equivalence_check']['runs'][name]['token_exact']}"
        )
    return 0


def cmd_export(args) -> int:
    from .export.hf import export_run

    run_dir = Path(args.run_dir).resolve()
    if args.format != "hf":
        raise SystemExit(f"export: unknown --format {args.format!r}; v0.2 supports 'hf'.")
    export_dir = export_run(run_dir, checkpoint=args.checkpoint, out_dir=args.out_dir)
    print(f"exported to {export_dir}")
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-dflash", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="preflight environment + target model contract")
    doctor.add_argument("--target", default=None, help="local target model directory")
    doctor.add_argument("--mask-token-id", type=int, default=151669)
    doctor.add_argument("--samples", type=int, default=1000, help="feature-disk estimate sample count")
    doctor.add_argument("--seq-length", type=int, default=2048)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    train = sub.add_parser("train", help="prepare -> regen -> capture -> train in one command")
    train.add_argument("--target", default=None, help="local target model directory")
    train.add_argument("--dataset", default=None, help="JSONL of multimodal rows")
    train.add_argument("--dataset-source", default=None, choices=["llava_instruct", "coco_caption", "jsonl", "hf"])
    train.add_argument("--annotation", default=None, help="annotation file for llava_instruct/coco_caption")
    train.add_argument("--hf-dataset", default=None)
    train.add_argument("--hf-split", default=None)
    train.add_argument("--images-dir", default=None)
    train.add_argument("--preset", default="single-gpu-48g", choices=list(_PRESET_NAMES))
    train.add_argument("--config", default=None, help="advanced: YAML config (merged over the preset)")
    train.add_argument("--run-dir", default=None)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--force", action="store_true", help="rebuild completed stages")
    train.add_argument("--resume", action="store_true", help="continue training from train/ckpt_last.pt")
    train.add_argument("--set", dest="set_list", action="append", default=[], help="dotted key=value override")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("eval", help="acceptance + token-exact evaluation for a finished run")
    evaluate.add_argument("run_dir")
    evaluate.add_argument("--config", default=None)
    evaluate.add_argument("--num-prompts", type=int, default=None)
    evaluate.add_argument("--max-new-tokens", type=int, default=None)
    evaluate.add_argument("--checkpoints", default=None, help="comma-separated checkpoint file names")
    evaluate.add_argument("--parallel-diagnostic", type=int, default=None)
    evaluate.add_argument(
        "--verification",
        default=None,
        choices=["sequential", "parallel", "strict_parallel"],
        help="target verification mode: parity sequential (default), raw parallel, or margin-guarded parallel",
    )
    evaluate.add_argument(
        "--strict-margin",
        type=float,
        default=None,
        help="top-1/top-2 logit gap below which strict_parallel re-verifies a round sequentially",
    )
    evaluate.add_argument("--tag", default=None, help="write eval_result_<tag>.json instead of overwriting")
    evaluate.set_defaults(func=cmd_eval)

    export = sub.add_parser("export", help="write the self-describing portable checkpoint")
    export.add_argument("run_dir")
    export.add_argument("--format", default="hf")
    export.add_argument("--checkpoint", default="ckpt_final.pt")
    export.add_argument("--out-dir", default=None)
    export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
