"""Five-minute demo paths: ``vlm-dflash demo`` and ``vlm-dflash train-demo``.

``demo`` needs no trained checkpoint and no external dataset: it generates a
tiny license-free synthetic sample (coloured shapes on a light background),
loads the target and scores the drafter through the normal token-exact
evaluation path. Without ``--drafter`` a seeded random-init drafter runs as the
acceptance baseline; pass a run directory, a checkpoint file or an exported
model-zoo directory to score real weights.

``train-demo`` runs the entire staged loop on the same sample -- prepare ->
target-native regen -> hidden-state capture -> tiny training -> eval -> export
-- so a fresh clone can prove the complete multimodal pipeline end to end.

When ``--drafter`` points at a checkpoint trained with different image pixel
bounds than the demo defaults, the demo aligns its target-side bounds with the
drafter's recorded identity so the provenance gate accepts the pair, and uses a
drafter-specific run directory (the frozen contract must not be overloaded).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import apply_overrides, dump_effective_config, resolve_run_dir
from .utils import ensure_dir, log, write_jsonl

DEMO_QUESTION = "Please describe this image in detail."

_PALETTE = ((214, 69, 65), (54, 110, 208), (62, 158, 96), (232, 168, 44))
_SHAPES = ("circle", "square", "triangle")


def _draw_demo_image(index: int):
    """Deterministic license-free image: 1-4 coloured shapes on a light background."""
    from PIL import Image, ImageDraw

    size = 768
    image = Image.new("RGB", (size, size), (246, 246, 244))
    draw = ImageDraw.Draw(image)
    count = 1 + index % 4
    shape = _SHAPES[index % 3]
    centres = ((240, 240), (528, 240), (240, 528), (528, 528))
    for position in range(count):
        colour = _PALETTE[(index + position) % len(_PALETTE)]
        centre_x, centre_y = centres[position]
        if shape == "circle":
            draw.ellipse((centre_x - 88, centre_y - 88, centre_x + 88, centre_y + 88), fill=colour)
        elif shape == "square":
            draw.rectangle((centre_x - 80, centre_y - 80, centre_x + 80, centre_y + 80), fill=colour)
        else:
            draw.polygon(
                ((centre_x, centre_y - 92), (centre_x - 92, centre_y + 76), (centre_x + 92, centre_y + 76)),
                fill=colour,
            )
    return image


def write_demo_source(run_dir: str | Path, num_images: int) -> tuple[Path, Path]:
    """Generate the synthetic sample; returns ``(pool_jsonl, images_dir)``."""
    if num_images < 2:
        raise ValueError(f"demo needs at least 2 images (train/val split), got {num_images}.")
    source_dir = ensure_dir(Path(run_dir) / "demo_source")
    images_dir = ensure_dir(source_dir / "images")
    rows = []
    for index in range(num_images):
        name = f"demo_{index:04d}.png"
        image_path = images_dir / name
        if not image_path.exists():
            _draw_demo_image(index).save(image_path)
        rows.append({"id": f"demo-{index:04d}", "question": DEMO_QUESTION, "answer": "", "image": name})
    pool = source_dir / "pool.jsonl"
    write_jsonl(pool, rows)
    return pool, images_dir


def build_demo_config(
    *,
    target: str,
    algorithm: str,
    run_dir: Path,
    pool_path: Path,
    images_dir: Path,
    num_train: int,
    num_val: int,
    num_prompts: int,
    max_new_tokens: int,
    steps: int,
    source_desc: str,
) -> dict:
    """The demo recipe: tiny synthetic sample, staged pipeline, tiny training budget."""
    return {
        "algorithm": algorithm,
        "run": {"name": run_dir.name, "dir": str(run_dir)},
        "target": {
            "model": target,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "device": "cuda:0",
            "local_files_only": True,
            # Uniform small images keep the demo fast; both bounds are
            # validated against the processor by the model contract.
            "image_min_pixels": 200704,
            "image_max_pixels": 200704,
        },
        "dataset": {
            "source": "jsonl",
            "path": str(pool_path),
            "jsonl_images_dir": str(images_dir),
            "slice": num_train,
            "val_size": num_val,
            "seed": 42,
            "demo_source": source_desc,
        },
        "regen": {"max_new_tokens": 64, "min_response_tokens": 16, "batch_size": 4, "temperature": 0.0},
        "capture": {
            "seq_length": 1024,
            "shard_size": 8,
            "dtype": "bfloat16",
            "visual_path_check": True,
            "visual_check_samples": 2,
            "store_final_hidden": True,
        },
        "drafter": {
            "num_hidden_layers": 3,
            "block_size": 8,
            "num_anchors": 64,
            "loss_decay_gamma": 7.0,
            # Family default; the resolved contract validates it against the
            # target tokenizer and fails loudly on a mismatch.
            "mask_token_id": 151669,
            "attention_backend": "flex_attention",
            "spatial_rope_enabled": True,
            "spatial_rope_mode": "replace",
        },
        "train": {
            "micro_batch_size": 4,
            "grad_accumulation_steps": 1,
            "num_epochs": 1,
            "lr": 6.0e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_ratio": 0.04,
            "min_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
            "log_every_steps": 5,
            "seed": 42,
            "device": "cuda:0",
            # The demo is the acceptance baseline + one tiny run; the overfit
            # sanity gate belongs to real training configs.
            "overfit_sanity": {"enabled": False},
        },
        "eval": {
            "verification": "sequential",
            "strict_parallel_margin": 2.0,
            "num_prompts": num_prompts,
            "max_new_tokens": max_new_tokens,
            "sequential_target_verification": True,
            "verify_sdpa_backend": "parity",
            "parallel_diagnostic_prompts": 0,
            "checkpoints": ["ckpt_init.pt"],
        },
        # Recorded so the printed recipe (and config.json) states the budget.
        "demo": {"steps": steps, "num_train": num_train, "num_val": num_val},
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    import torch

    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def random_init_checkpoint(cfg: dict, contract: dict, algorithm, out_path: str | Path) -> None:
    """Seeded random-init draft weights saved in the normal checkpoint format."""
    from .provenance import identity_from_contract
    from .utils import set_seed

    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    draft_config = algorithm.build_draft_config(contract, cfg)
    model = algorithm.build_inference_module(
        draft_config.to_dict(), str(cfg["drafter"].get("attention_backend", "flex_attention"))
    )
    tied = bool(contract.get("head", {}).get("tie_word_embeddings", True))
    payload = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": algorithm.name,
        **algorithm.checkpoint_metadata(contract, cfg, draft_config),
        "head": "tied_embedding" if tied else "target_lm_head",
        "identity": identity_from_contract(contract),
        "step": 0,
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    _save_checkpoint(Path(out_path), payload)


def checkpoint_from_export(export_dir: str | Path, out_path: str | Path) -> str:
    """Convert a self-describing zoo export into a run checkpoint; returns the algorithm name."""
    from safetensors.torch import load_file

    from .export.hf import EXPORT_FORMAT
    from .utils import read_json

    export_dir = Path(export_dir)
    export_config = read_json(export_dir / "config.json")
    if export_config.get("format") != EXPORT_FORMAT:
        raise ValueError(
            f"{export_dir} is not a vlm-dflash export (format={export_config.get('format')!r})."
        )
    drafter = export_config["drafter"]
    method = str(export_config.get("method", "dflash"))
    payload = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": method,
        "draft_config": drafter["draft_config"],
        "attention_backend": drafter.get("attention_backend", "flex_attention"),
        "mask_token_id": int(drafter["mask_token_id"]),
        "block_size": int(drafter["block_size"]),
        "num_anchors": int(drafter.get("num_anchors", 256)),
        "loss_decay_gamma": drafter.get("loss_decay_gamma"),
        "target_layer_ids": [int(x) for x in drafter.get("target_layer_ids", [])],
        "head": (export_config.get("target") or {}).get("head", {}).get("trained_with", "tied_embedding"),
        "identity": export_config.get("identity"),
        "step": 0,
    }
    method_details = export_config.get("method_details")
    if isinstance(method_details, dict):
        payload[method] = method_details
    payload["model"] = dict(load_file(export_dir / "model.safetensors"))
    _save_checkpoint(Path(out_path), payload)
    return method


def attach_drafter(drafter_path: str | Path, train_dir: Path) -> str:
    """Place drafter weights at ``<train_dir>/ckpt_final.pt``; returns a description."""
    drafter_path = Path(drafter_path).expanduser().resolve()
    if not drafter_path.exists():
        raise SystemExit(f"--drafter {drafter_path} does not exist.")
    link = train_dir / "ckpt_final.pt"
    if drafter_path.is_dir():
        run_checkpoint = drafter_path / "train" / "ckpt_final.pt"
        export_files = (drafter_path / "config.json").exists() and (drafter_path / "model.safetensors").exists()
        if run_checkpoint.exists():
            source, description = run_checkpoint, f"trained checkpoint from run {drafter_path}"
        elif export_files:
            method = checkpoint_from_export(drafter_path, link)
            return f"exported checkpoint {drafter_path} (algorithm={method})"
        else:
            raise SystemExit(
                f"--drafter {drafter_path} is a directory but has neither train/ckpt_final.pt "
                "nor an exported config.json + model.safetensors."
            )
    elif drafter_path.suffix == ".pt":
        source, description = drafter_path, f"trained checkpoint {drafter_path}"
    else:
        raise SystemExit(
            "--drafter must be a run directory, a .pt checkpoint file or an exported zoo directory."
        )
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        os.symlink(source, link)
    except OSError:  # pragma: no cover - platform dependent (e.g. Windows)
        shutil.copy2(source, link)
    return description


def drafter_identity(drafter_path: str | Path) -> dict:
    """Read the target identity recorded in a drafter (run dir, export dir or .pt file)."""
    import torch

    from .utils import read_json

    path = Path(drafter_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"--drafter {path} does not exist.")
    if path.is_dir():
        run_checkpoint = path / "train" / "ckpt_final.pt"
        export_config = path / "config.json"
        if run_checkpoint.exists():
            meta = torch.load(run_checkpoint, map_location="cpu", weights_only=False)
            return dict(meta.get("identity") or {})
        if export_config.exists():
            return dict(read_json(export_config).get("identity") or {})
        raise SystemExit(
            f"--drafter {path} is a directory but has neither train/ckpt_final.pt "
            "nor an exported config.json."
        )
    if path.suffix == ".pt":
        meta = torch.load(path, map_location="cpu", weights_only=False)
        return dict(meta.get("identity") or {})
    raise SystemExit(
        "--drafter must be a run directory, a .pt checkpoint file or an exported zoo directory."
    )


def drafter_run_suffix(drafter_path: str | Path) -> str:
    """Default run-dir suffix for a drafter-scored demo (keeps frozen contracts separate)."""
    name = Path(drafter_path).expanduser().resolve().name
    return name[:-3] if name.endswith(".pt") else name


def _algorithm_name(value) -> str | None:
    """Normalize a config/checkpoint ``algorithm`` field (string or {name: ...})."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def detect_drafter_algorithm(drafter_path: str | Path) -> str | None:
    """Best-effort read of the algorithm a drafter was trained/exported with.

    Lets ``demo --drafter <export dir>`` follow the checkpoint's own algorithm
    instead of assuming the CLI default (scoring a Domino export as DFlash used
    to fail loudly, which is correct but a usability trap).
    """
    from .utils import read_json

    path = Path(drafter_path).expanduser().resolve()
    if not path.exists():
        return None
    try:
        if path.is_dir():
            algorithm_json = path / "algorithm.json"
            if algorithm_json.exists():
                data = read_json(algorithm_json)
                name = _algorithm_name(data.get("algorithm")) if isinstance(data, dict) else None
                if name:
                    return name
            config_json = path / "config.json"
            if config_json.exists():
                config = read_json(config_json)
                name = _algorithm_name(config.get("method")) or _algorithm_name(config.get("algorithm"))
                if name:
                    return name
            run_checkpoint = path / "train" / "ckpt_final.pt"
            if run_checkpoint.exists():
                import torch

                meta = torch.load(run_checkpoint, map_location="cpu", weights_only=False)
                return _algorithm_name(meta.get("algorithm"))
        elif path.suffix == ".pt":
            import torch

            meta = torch.load(path, map_location="cpu", weights_only=False)
            return _algorithm_name(meta.get("algorithm"))
    except Exception:
        return None
    return None


def _demo_setup(args, kind: str, *, target_overrides: dict | None = None, run_name_suffix: str | None = None):
    """Shared setup: resolve target/run dir, generate the sample, write config + contract."""
    from .algorithms import get_algorithm
    from .contract import ensure_contract
    from .data.prepare import build_dataset
    from .utils import read_json

    target = args.target or os.environ.get("VLM_DFLASH_TARGET")
    if not target:
        raise SystemExit(f"{kind}: pass --target <model_dir> (or set VLM_DFLASH_TARGET).")
    algorithm = get_algorithm(args.algorithm)
    default_name = f"runs/demo_{algorithm.name}"
    if run_name_suffix:
        default_name = f"{default_name}_{run_name_suffix}"
    run_dir = resolve_run_dir({"run": {"dir": default_name}}, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    num_images = int(getattr(args, "images", 8))
    num_train = max(1, num_images // 2)
    num_val = num_images - num_train
    if getattr(args, "dataset", None):
        pool_path = Path(args.dataset).expanduser().resolve()
        if not pool_path.exists():
            raise SystemExit(f"{kind}: --dataset {pool_path} does not exist.")
        images_dir = getattr(args, "images_dir", None)
        if images_dir:
            images_dir = Path(images_dir).expanduser().resolve()
        else:
            meta_file = pool_path.parent / "meta.json"
            if meta_file.exists():
                media_dir = Path(read_json(meta_file)["media_dir"])
                images_dir = media_dir if media_dir.is_absolute() else (meta_file.parent / media_dir).resolve()
            else:
                raise SystemExit(f"{kind}: pass --images-dir (no sibling meta.json next to {pool_path.name}).")
        source_desc = f"{pool_path} ({num_images} rows requested)"
    else:
        pool_path, images_dir = write_demo_source(run_dir, num_images)
        source_desc = f"synthetic shapes sample ({num_images} license-free generated images)"

    cfg = build_demo_config(
        target=target,
        algorithm=algorithm.name,
        run_dir=run_dir,
        pool_path=pool_path,
        images_dir=images_dir,
        num_train=num_train,
        num_val=num_val,
        num_prompts=int(args.num_prompts),
        max_new_tokens=int(args.max_new_tokens),
        steps=int(getattr(args, "steps", 0)),
        source_desc=source_desc,
    )
    for key, value in (target_overrides or {}).items():
        if value is not None:
            cfg["target"][key] = int(value)
    cfg = apply_overrides(cfg, getattr(args, "set_list", None))
    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    build_dataset(cfg, run_dir / "data", limit=num_train)
    return cfg, contract, algorithm, run_dir, source_desc


def _print_report(result: dict, run_dir: Path, drafter_desc: str, source_desc: str) -> None:
    run_name = next(iter(result["runs"]))
    summary = result["runs"][run_name]
    equivalence = result["equivalence_check"]["runs"][run_name]
    slots = ", ".join("n/a" if value is None else f"{value:.2f}" for value in summary["per_slot_acceptance"])
    identity = result["identity"]
    exact = equivalence["num_prompts"] - len(equivalence["failed_ids"])
    print()
    print("vlm-dflash demo")
    print(f"  Algorithm:        {result['algorithm']}")
    print(f"  Target:           {identity['target_model_path']}")
    print(f"                    (fingerprint {identity['model_fingerprint'][:12]}, mask_token={identity['mask_token_id']})")
    print(f"  Drafter:          {drafter_desc}")
    print(f"  Sample:           {source_desc}")
    print(f"  Prompts:          {result['settings']['num_prompts']} (max_new_tokens={result['settings']['max_new_tokens']})")
    print(f"  AL:               {summary['mean_al']:.3f} tokens per target forward")
    print(f"  Acceptance rate:  {summary['acceptance_rate']:.4f}")
    print(
        f"  Token exactness:  {exact}/{equivalence['num_prompts']} prompts token-identical "
        f"to target-only greedy ({result['equivalence_check']['verification']} verification)"
    )
    print(f"  Per-slot accept:  [{slots}] (draft slots 0..{len(summary['per_slot_acceptance']) - 1})")
    speedup = summary.get("speedup_vs_target_only")
    if speedup:
        print(f"  Speed vs target:  {speedup:.2f}x wall-clock on this machine")
    print(f"  Results:          {run_dir}/eval/eval_result.json")


def run_demo(args) -> int:
    """``vlm-dflash demo``: token-exact eval on the tiny sample; no training."""
    from .eval.run import evaluate

    if getattr(args, "drafter", None) and not getattr(args, "algorithm", None):
        detected = detect_drafter_algorithm(args.drafter)
        if detected:
            args.algorithm = detected
            log(f"demo: --drafter records algorithm={detected}; evaluating with it (pass --algorithm to override)")

    target_overrides: dict[str, int] = {}
    run_name_suffix = None
    if getattr(args, "drafter", None):
        identity = drafter_identity(args.drafter)
        for key in ("image_min_pixels", "image_max_pixels"):
            value = identity.get(key)
            if value is not None:
                target_overrides[key] = int(value)
        run_name_suffix = drafter_run_suffix(args.drafter)
    cfg, contract, algorithm, run_dir, source_desc = _demo_setup(
        args, "demo", target_overrides=target_overrides, run_name_suffix=run_name_suffix
    )
    train_dir = ensure_dir(run_dir / "train")
    if getattr(args, "drafter", None):
        drafter_desc = attach_drafter(args.drafter, train_dir)
        cfg["eval"]["checkpoints"] = ["ckpt_final.pt"]
    else:
        random_init_checkpoint(cfg, contract, algorithm, train_dir / "ckpt_init.pt")
        drafter_desc = "random-init drafter (acceptance baseline; pass --drafter to score trained weights)"
    dump_effective_config(cfg, run_dir / "config.json")

    result = evaluate(
        cfg, run_dir, contract, num_prompts=int(args.num_prompts), max_new_tokens=int(args.max_new_tokens)
    )
    _print_report(result, run_dir, drafter_desc, source_desc)
    return 0


def run_train_demo(args) -> int:
    """``vlm-dflash train-demo``: the full staged loop on the tiny sample."""
    from .dflash.train import train_dflash
    from .eval.run import evaluate
    from .export.hf import export_run
    from .target.capture import run_capture
    from .target.regen import run_regen

    cfg, contract, algorithm, run_dir, source_desc = _demo_setup(args, "train-demo")
    cfg["eval"]["checkpoints"] = ["ckpt_final.pt", "ckpt_init.pt"]
    dump_effective_config(cfg, run_dir / "config.json")

    run_regen(cfg, run_dir, contract)
    run_capture(cfg, run_dir, contract)
    summary = train_dflash(cfg, run_dir, contract, steps=int(args.steps))
    log(f"train-demo: {summary['steps']} steps, final loss {summary['final_loss']:.4f}")
    result = evaluate(
        cfg, run_dir, contract, num_prompts=int(args.num_prompts), max_new_tokens=int(args.max_new_tokens)
    )
    export_dir = export_run(run_dir)
    _print_report(result, run_dir, "trained drafter (ckpt_final) vs random-init baseline (ckpt_init)", source_desc)
    print(f"  Export:           {export_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("vlm_dflash.demo is a library module; use the vlm-dflash CLI.")
