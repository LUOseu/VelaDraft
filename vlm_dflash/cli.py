"""Unified command line: ``vlm-dflash {doctor,demo,train-demo,train,eval,export}``.

The five stage scripts remain the advanced interface; the CLI is the happy
path: ``doctor`` preflights an environment/target, ``train`` runs the whole
staged pipeline (prepare -> regen -> capture -> train) from a target + dataset
with sensible presets, ``eval`` scores a finished run, and ``export`` writes
the self-describing portable checkpoint.

``demo`` is the five-minute first run (tiny synthetic sample, token-exact
evaluation, no training; see :mod:`vlm_dflash.demo`); ``train-demo`` runs the
complete staged loop on that same sample.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import resources as importlib_resources
from pathlib import Path

from . import __version__
from .config import apply_overrides, dump_effective_config, load_config, resolve_run_dir

_PRESET_NAMES = ("smoke", "single-gpu-24g", "single-gpu-48g")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_preset(name_or_path: str) -> dict:
    """Load a preset by name (packaged resource) or by YAML path.

    The three named presets ship inside the wheel
    (``vlm_dflash/presets/*.yaml``); any existing filesystem path is loaded
    as-is for custom presets.
    """
    candidate = Path(name_or_path)
    if candidate.exists():
        return load_config(candidate)
    name = candidate.stem if candidate.suffix in (".yaml", ".yml") else name_or_path
    if name in _PRESET_NAMES:
        with importlib_resources.as_file(
            importlib_resources.files("vlm_dflash") / "presets" / f"{name}.yaml"
        ) as path:
            return load_config(path)
    raise FileNotFoundError(
        f"Preset {name_or_path!r} not found; known presets: {', '.join(_PRESET_NAMES)} "
        "(or pass a path to a YAML file)."
    )


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
    """Read the tested version pins shipped inside the package."""
    try:
        text = (
            importlib_resources.files("vlm_dflash") / "resources" / "constraints-tested.txt"
        ).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return {}
    pins: dict[str, str] = {}
    for line in text.splitlines():
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


def collect_doctor_report(
    target: str | None,
    mask_token_id: int | None,
    samples: int,
    seq_length: int,
    algorithm: str | None = None,
) -> dict:
    report: dict = {"ok": True, "checks": [], "warnings": [], "errors": []}

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["ok"] = False
            report["errors"].append(f"{name}: {detail}")

    from .algorithms import UnknownAlgorithmError, available_algorithms, get_algorithm

    report["python"] = sys.version.split()[0]
    check("python>=3.10", sys.version_info >= (3, 10), report["python"])

    report["algorithms"] = {"available": available_algorithms()}
    if algorithm:
        try:
            selected = get_algorithm(algorithm)
            report["algorithms"]["selected"] = selected.metadata.as_dict()
            check(f"algorithm {algorithm!r} registered", True, selected.metadata.description)
            if selected.name == "dflash2":
                # DFlash2 retains the full verifier vocabulary by construction:
                # the draft samples through the target's own lm_head, and the
                # selector reranks candidates *before* any vocab mapping would
                # happen. Pruned draft vocabularies are not supported.
                check(
                    "algorithm dflash2 full-vocab head required",
                    True,
                    "draft head = target lm_head (full verifier vocabulary); pruned draft vocab is rejected",
                )
        except UnknownAlgorithmError as error:
            check(f"algorithm {algorithm!r} registered", False, str(error))

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
    report = collect_doctor_report(
        args.target, args.mask_token_id, args.samples, args.seq_length, algorithm=args.algorithm
    )
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
        print(f"  algorithms: available={report['algorithms']['available']}")
        selected = report["algorithms"].get("selected")
        if selected:
            print(f"  algorithm selected: {selected['name']} — maturity={selected['maturity']}")
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

FEATURE_MODES = ("offline", "online")


def resolve_feature_mode(cfg: dict, cli_value: str | None = None) -> str:
    """Resolve ``train.feature_mode`` after the final config merge.

    Priority: explicit CLI ``--mode`` / ``--feature-mode`` > dotted ``--set`` >
    YAML/preset > default ``offline``. Validation runs on the merged config so
    an override can never be validated away. ``--features`` reuse is an offline
    concept; combining it with online is refused in :func:`cmd_train` before
    any expensive stage.
    """
    if cli_value is not None:
        cfg.setdefault("train", {})["feature_mode"] = str(cli_value).lower()
    mode = str((cfg.get("train") or {}).get("feature_mode", "offline")).lower()
    if mode not in FEATURE_MODES:
        raise SystemExit(f"train: feature_mode must be one of {FEATURE_MODES}, got {mode!r}.")
    return mode


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
    if getattr(args, "algorithm", None):
        if isinstance(cfg.get("algorithm"), dict):
            cfg["algorithm"]["name"] = str(args.algorithm)
        else:
            cfg["algorithm"] = str(args.algorithm)
    from .algorithms import get_algorithm

    # Merge preset + YAML + dotted overrides first, then resolve the algorithm
    # and feature mode on the config training will actually run with, so an
    # override can never be validated away (or fail validation it satisfied).
    cfg = apply_overrides(cfg, args.set_list)
    algorithm = get_algorithm(cfg.get("algorithm"))
    algorithm.validate_config(cfg)
    feature_mode = resolve_feature_mode(cfg, getattr(args, "feature_mode", None))
    run = cfg.setdefault("run", {})
    if not run.get("dir"):
        explicit_name = run.get("name")
        name = str(explicit_name) if explicit_name else f"{_slug(cfg['target']['model'])}-{algorithm.name}"
        if feature_mode == "online" and not explicit_name:
            name = f"{name}-online"
        run.update({"name": name, "dir": f"runs/{name}"})
    return cfg


def _stage_manifest_path(run_dir: Path, stage: str) -> Path:
    return run_dir / stage / "stage_manifest.json"


def _dataset_file_fingerprint(cfg: dict) -> dict:
    """Size/mtime of the local dataset files a stage will read (cheap staleness check)."""
    dataset = cfg.get("dataset", {})
    out: dict = {}
    for key in ("path", "annotation"):
        value = dataset.get(key)
        if not value or not isinstance(value, str) or "${" in value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            stat = path.stat()
            out[key] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
    return out


def stage_digest_payload(
    cfg: dict, contract: dict, stage: str, upstream_digest: str | None, limit: int | None = None
) -> dict:
    """The inputs that decide a cached stage's outputs (config subset + upstream + target)."""
    from .provenance import identity_from_contract

    key = {"data": "dataset", "regen": "regen", "capture": "capture"}[stage]
    identity = identity_from_contract(contract)
    # Audit-only fields never change the outputs: the same checkpoint copied to
    # another directory must keep cached stages reusable.
    identity.pop("target_model_path", None)
    payload = {
        "stage": stage,
        "config": cfg.get(key, {}),
        "limit": limit,
        "target_identity": identity,
        "upstream_digest": upstream_digest,
    }
    if stage == "data":
        payload["dataset_files"] = _dataset_file_fingerprint(cfg)
    return payload


def stage_status(run_dir: Path, stage: str, marker: str, digest: str, force: bool) -> str:
    """Decide whether a cached stage can be reused: 'run' or 'skip'.

    Outputs with a missing or mismatching recorded inputs digest are refused
    before any expensive compute (no silent reuse of stale artifacts).
    """
    present = (run_dir / marker).exists()
    if force or not present:
        return "run"
    manifest_path = _stage_manifest_path(run_dir, stage)
    if not manifest_path.exists():
        raise SystemExit(
            f"train: stage '{stage}' has outputs at {run_dir / marker} but no recorded inputs digest "
            f"({manifest_path.name} is missing - the run predates stage manifests). "
            "Rerun with --force to rebuild the stage, or use a new run directory."
        )
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded.get("inputs_digest") != digest:
        raise SystemExit(
            f"train: stage '{stage}' was produced from different inputs (recorded digest "
            f"{(recorded.get('inputs_digest') or '?')[:12]} != current {digest[:12]}); refusing to "
            "silently reuse it. Rerun with --force to rebuild, or use a new run directory."
        )
    return "skip"


def write_stage_manifest(run_dir: Path, stage: str, payload: dict, digest: str) -> None:
    from .utils import now_iso

    manifest = {
        "stage": stage,
        "inputs_digest": digest,
        "payload": payload,
        "recorded_at": now_iso(),
    }
    path = _stage_manifest_path(run_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def stage_digest(
    cfg: dict, contract: dict, stage: str, upstream_digest: str | None, limit: int | None = None
) -> str:
    from .utils import sha256_json

    return sha256_json(stage_digest_payload(cfg, contract, stage, upstream_digest, limit=limit))


def _require_stage(run_dir: Path, stage: str, marker: str, digest: str) -> None:
    """Resume preflight: the upstream stage must exist AND still match its inputs.

    A resume must never quietly rebuild (and thereby change) the data a
    checkpoint was trained from; a missing marker or a changed inputs digest is
    refused here, before anything under ``run_dir`` is written.
    """
    try:
        status = stage_status(run_dir, stage, marker, digest, False)
    except SystemExit as exc:
        raise SystemExit(
            f"train --resume: upstream '{stage}' stage was produced from different inputs ({exc}). "
            "Exact resume needs the original stages; start a new run directory."
        ) from exc
    if status != "skip":
        raise SystemExit(
            f"train --resume: upstream '{stage}' outputs ({marker}) are missing; cannot resume exactly."
        )


def _preflight_resume(cfg: dict, run_dir: Path, args, feature_mode: str, external_features: Path | None) -> dict:
    """Read-only exact-resume gate, run before ``cmd_train`` writes anything.

    Verifies the upstream stages, then the mode / algorithm / feature-source
    digest / training plan / identity of ``train/ckpt_last.pt``. Loading the
    target is not required (the online digest comes from the consumed rows and
    image bytes); every failure raises ``SystemExit`` with the run directory
    untouched.
    """
    from .contract import validate_yaml_against_contract
    from .utils import log, read_json

    ckpt_path = run_dir / "train" / "ckpt_last.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"train --resume: no checkpoint at {ckpt_path} to resume from.")
    contract_path = run_dir / "contract.json"
    if not contract_path.exists():
        raise SystemExit(
            f"train --resume: {contract_path.name} is missing for this run; cannot preflight an exact "
            "resume. Start a new run directory, or run without --resume."
        )
    contract = read_json(contract_path)
    if contract["target"]["model_path"] != str(Path(cfg["target"]["model"]).resolve()):
        raise SystemExit(
            f"train --resume: run directory was created for target {contract['target']['model_path']} "
            f"but the config points at {cfg['target']['model']}; use a new run directory when switching "
            "targets."
        )
    validate_yaml_against_contract(cfg, contract, "config")
    limit = args.limit
    # Stage dependencies follow the feature source actually used by the run
    # (mirroring cmd_train): online and internal-offline builds own their
    # regen stage; `--features <store>` runs never built regen/capture, so
    # demanding them here would wrongly reject a legal external-store resume.
    data_digest = stage_digest(cfg, contract, "data", None, limit=limit)
    _require_stage(run_dir, "data", "data/data.jsonl", data_digest)
    if external_features is None:
        regen_digest = stage_digest(cfg, contract, "regen", data_digest, limit=limit)
        _require_stage(run_dir, "regen", "regen/target_data.jsonl", regen_digest)
        if feature_mode == "offline":
            capture_digest = stage_digest(cfg, contract, "capture", regen_digest, limit=limit)
            _require_stage(run_dir, "capture", "features/index.jsonl", capture_digest)
    from .dflash.train import preflight_resume

    try:
        result = preflight_resume(
            cfg,
            run_dir,
            contract,
            steps=args.steps,
            limit=limit,
            features_dir=external_features,
            feature_mode=feature_mode,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"train --resume: {exc}") from exc
    log(
        f"train: resume preflight passed (step {result['step']}/{result['total_steps']}, "
        f"plan {result['plan_digest'][:12]}, source {result['feature_source_digest'][:12]})"
    )
    return result


def cmd_train(args) -> int:
    from .contract import ensure_contract
    from .utils import log

    cfg = build_train_config(args)
    feature_mode = str((cfg.get("train") or {}).get("feature_mode", "offline")).lower()
    external_features = Path(args.features).expanduser().resolve() if args.features else None
    if feature_mode == "online" and external_features is not None:
        raise SystemExit(
            "train: --mode online extracts teacher features live and cannot reuse a captured store; "
            "drop --features (or use --mode offline to reuse one)."
        )
    if feature_mode == "online":
        # Rejected before any stage runs (prepare / regen / capture): the target
        # cannot be copied into DataLoader workers, so the whole run would be
        # wasted compute.
        num_workers = int((cfg.get("train") or {}).get("num_workers", 0))
        if num_workers != 0:
            raise SystemExit(
                "train: feature_mode=online extracts target features in the main process and requires "
                f"train.num_workers=0 (got train.num_workers={num_workers}); the target cannot be copied "
                "into DataLoader workers."
            )
    run_dir = resolve_run_dir(cfg, args.run_dir)
    if args.resume:
        if args.force:
            raise SystemExit(
                "train --resume cannot be combined with --force: --force rebuilds upstream stages, which "
                "an exact resume must refuse. Drop one of the two flags."
            )
        # Read-only preflight before the first write (config.json below): a
        # rejected resume must leave the original config, checkpoints, logs,
        # provenance and diagnostics byte-identical.
        _preflight_resume(cfg, run_dir, args, feature_mode, external_features)
    run_dir.mkdir(parents=True, exist_ok=True)

    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)

    limit = args.limit
    if external_features is not None:
        for required in ("index.jsonl", "manifest.json"):
            if not (external_features / required).exists():
                raise SystemExit(
                    f"train: --features {external_features} is not a captured feature store "
                    f"({required} missing)."
                )

    data_digest = stage_digest(cfg, contract, "data", None, limit=limit)
    if stage_status(run_dir, "data", "data/data.jsonl", data_digest, args.force) == "run":
        from .data.prepare import build_dataset

        build_dataset(cfg, run_dir / "data", limit=limit)
        write_stage_manifest(
            run_dir, "data", stage_digest_payload(cfg, contract, "data", None, limit=limit), data_digest
        )
    else:
        log("train: data stage present with unchanged inputs, skipping (use --force to rebuild)")

    if external_features is not None:
        log(f"train: --features {external_features}; regen + capture stages skipped (reuse recorded in provenance)")
    else:
        regen_digest = stage_digest(cfg, contract, "regen", data_digest, limit=limit)
        if stage_status(run_dir, "regen", "regen/target_data.jsonl", regen_digest, args.force) == "run":
            from .target.regen import run_regen

            run_regen(cfg, run_dir, contract, limit=limit)
            write_stage_manifest(
                run_dir, "regen", stage_digest_payload(cfg, contract, "regen", data_digest, limit=limit), regen_digest
            )
        else:
            log("train: regen stage present with unchanged inputs, skipping (use --force to rerun)")

        if feature_mode == "online":
            log(
                "train: feature_mode=online; skipping the offline capture stage "
                "(target features are extracted live by the trainer)"
            )
            if (run_dir / "features" / "index.jsonl").exists():
                log(
                    "train: note: a staged capture exists in this run dir; online training ignores it "
                    "(run with --mode offline to train from it)"
                )
        else:
            capture_digest = stage_digest(cfg, contract, "capture", regen_digest, limit=limit)
            if stage_status(run_dir, "capture", "features/index.jsonl", capture_digest, args.force) == "run":
                from .target.capture import run_capture

                run_capture(cfg, run_dir, contract, limit=limit)
                write_stage_manifest(
                    run_dir,
                    "capture",
                    stage_digest_payload(cfg, contract, "capture", regen_digest, limit=limit),
                    capture_digest,
                )
            else:
                log("train: capture stage present with unchanged inputs, skipping (use --force to recapture)")

    from .dflash.train import train_dflash

    summary = train_dflash(
        cfg,
        run_dir,
        contract,
        steps=args.steps,
        limit=limit,
        resume=args.resume,
        features_dir=external_features,
        feature_mode=feature_mode,
    )
    log(f"train: done with steps={summary['steps']} final_loss={summary['final_loss']}")
    return 0


# --------------------------------------------------------------------------- #
# eval / export
# --------------------------------------------------------------------------- #

def cmd_eval(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    cfg = _load_run_config(run_dir, args.config)
    from .algorithms import get_algorithm

    get_algorithm(cfg.get("algorithm"))
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
# demo / train-demo
# --------------------------------------------------------------------------- #

def cmd_demo(args) -> int:
    from .demo import run_demo

    return run_demo(args)


def cmd_train_demo(args) -> int:
    from .demo import run_train_demo

    return run_train_demo(args)


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-dflash", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="preflight environment + target model contract")
    doctor.add_argument("--target", default=None, help="local target model directory")
    doctor.add_argument("--mask-token-id", type=int, default=151669)
    doctor.add_argument("--algorithm", default=None, help="check a drafter algorithm name against the registry")
    doctor.add_argument("--samples", type=int, default=1000, help="feature-disk estimate sample count")
    doctor.add_argument("--seq-length", type=int, default=2048)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    def add_demo_args(parser: argparse.ArgumentParser, *, with_steps: bool) -> None:
        parser.add_argument("--target", default=None, help="local target model directory (or set VLM_DFLASH_TARGET)")
        parser.add_argument(
            "--algorithm",
            default=None,
            help="drafter algorithm name (registered; default: dflash, or the --drafter's recorded algorithm)",
        )
        parser.add_argument("--dataset", default=None, help="multimodal jsonl pool to use instead of the synthetic sample")
        parser.add_argument("--images-dir", default=None, help="image root for --dataset (default: its sibling meta.json)")
        parser.add_argument("--images", type=int, default=8, help="synthetic sample size (>= 2)")
        parser.add_argument("--run-dir", default=None)
        parser.add_argument("--num-prompts", type=int, default=4)
        parser.add_argument("--max-new-tokens", type=int, default=64)
        parser.add_argument("--set", dest="set_list", action="append", default=[], help="dotted key=value override")
        if with_steps:
            parser.add_argument("--steps", type=int, default=80, help="tiny training budget for train-demo")

    demo = sub.add_parser("demo", help="five-minute first run: synthetic sample + token-exact eval, no training")
    add_demo_args(demo, with_steps=False)
    demo.add_argument("--drafter", default=None, help="run dir, checkpoint file or exported zoo dir with drafter weights")
    demo.set_defaults(func=cmd_demo)

    train_demo = sub.add_parser(
        "train-demo",
        help="full staged pipeline (prepare -> regen -> capture -> train -> eval -> export) on the synthetic sample",
    )
    add_demo_args(train_demo, with_steps=True)
    train_demo.set_defaults(func=cmd_train_demo)

    train = sub.add_parser("train", help="prepare -> regen -> capture -> train in one command")
    train.add_argument("--target", default=None, help="local target model directory")
    train.add_argument("--dataset", default=None, help="JSONL of multimodal rows")
    train.add_argument("--dataset-source", default=None, choices=["llava_instruct", "coco_caption", "jsonl", "hf"])
    train.add_argument("--annotation", default=None, help="annotation file for llava_instruct/coco_caption")
    train.add_argument("--hf-dataset", default=None)
    train.add_argument("--hf-split", default=None)
    train.add_argument("--images-dir", default=None)
    train.add_argument("--preset", default="single-gpu-48g", choices=list(_PRESET_NAMES))
    train.add_argument(
        "--algorithm",
        default=None,
        help="drafter algorithm name (registered; default: config value or dflash)",
    )
    train.add_argument("--config", default=None, help="advanced: YAML config (merged over the preset)")
    train.add_argument("--run-dir", default=None)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument(
        "--mode",
        "--feature-mode",
        dest="feature_mode",
        default=None,
        choices=list(FEATURE_MODES),
        help=(
            "target-feature source: offline (default; train from a pre-captured feature store) or "
            "online (extract teacher-forced hidden states live from the frozen target during training)"
        ),
    )
    train.add_argument("--force", action="store_true", help="rebuild completed stages")
    train.add_argument("--resume", action="store_true", help="continue training from train/ckpt_last.pt")
    train.add_argument(
        "--features",
        default=None,
        help=(
            "reuse an existing captured feature store (directory with index.jsonl + manifest.json); "
            "skips regen + capture so several algorithms can train from one capture"
        ),
    )
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
