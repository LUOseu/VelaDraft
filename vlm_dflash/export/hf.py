"""HF-style, self-describing checkpoint export.

The export directory is the portable asset: draft weights in ``safetensors``
plus a ``config.json`` that pins the target identity, adapter family, routing,
MRoPE, mask token, method hyper-parameters, training/eval evidence, and a
rendered ``model_card.md``. ``load_export`` reconstructs the draft model from
the directory alone (no run directory needed).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from ..utils import ensure_dir, log, now_iso, read_json, sha256_file, write_json

EXPORT_FORMAT = "vlm-dflash-export-v1"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def build_export_config(run_dir: Path, checkpoint: dict, checkpoint_name: str) -> dict:
    contract = read_json(run_dir / "contract.json")
    train_summary = read_json(run_dir / "train" / "train_summary.json") if (run_dir / "train" / "train_summary.json").exists() else None
    manifest = read_json(run_dir / "data" / "manifest.json") if (run_dir / "data" / "manifest.json").exists() else None
    eval_result = read_json(run_dir / "eval" / "eval_result.json") if (run_dir / "eval" / "eval_result.json").exists() else None
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    provenance = read_json(run_dir / "provenance.json") if (run_dir / "provenance.json").exists() else {}
    drafter_cfg = config.get("drafter", {})
    rope_cfg = {
        "sections": contract["rope"]["sections"],
        "mode": drafter_cfg.get("spatial_rope_mode", "replace"),
        "enabled": bool(drafter_cfg.get("spatial_rope_enabled", True)),
    }
    data_block = None
    if manifest is not None:
        data_block = {
            "source": manifest.get("source"),
            "num_train": manifest.get("num_train"),
            "num_val": manifest.get("num_val"),
            "rows_sha256": manifest.get("rows_sha256"),
        }
    return {
        "format": EXPORT_FORMAT,
        "created": now_iso(),
        "method": "dflash",
        "checkpoint": checkpoint_name,
        "code_commit": _git_commit(),
        "adapter": contract["target"]["model_type"],
        "target": {
            "model_path": contract["target"]["model_path"],
            "model_type": contract["target"]["model_type"],
            "model_fingerprint": contract["target"]["model_fingerprint"],
            "processor_fingerprint": contract["target"]["processor_fingerprint"],
            "image_min_pixels": contract["image"]["min_pixels"],
            "image_max_pixels": contract["image"]["max_pixels"],
            # Which frozen output head the draft logits were trained against: the
            # embedding matrix (tied) or the target's own lm_head (untied). Serving
            # must sample through the same head or acceptance collapses.
            "head": {
                "trained_with": checkpoint.get("head", "tied_embedding"),
                "tie_word_embeddings": contract.get("head", {}).get("tie_word_embeddings"),
            },
            "special_tokens": contract["tokens"],
        },
        "drafter": {
            "num_hidden_layers": contract["layers"]["num_draft_layers"],
            "block_size": int(checkpoint["block_size"]),
            "num_anchors": int(checkpoint["num_anchors"]),
            "loss_decay_gamma": float(checkpoint["loss_decay_gamma"]),
            "attention_backend": checkpoint["attention_backend"],
            "mask_token_id": int(checkpoint["mask_token_id"]),
            "target_layer_ids": [int(x) for x in checkpoint["target_layer_ids"]],
            "spatial_rope": rope_cfg,
            "draft_config": checkpoint["draft_config"],
        },
        "training": None if train_summary is None else {
            key: train_summary.get(key)
            for key in ("steps", "final_loss", "first10_mean_loss", "last10_mean_loss", "nan_seen", "wall_seconds", "peak_gpu_mem_gb")
        },
        "data": data_block,
        "eval": None if eval_result is None else {
            "num_prompts": eval_result.get("settings", {}).get("num_prompts"),
            "token_exact": eval_result.get("equivalence_check", {}).get("token_exact"),
            "runs": {
                name: {
                    key: summary.get(key)
                    for key in ("mean_al", "acceptance_rate", "per_slot_acceptance", "role")
                }
                for name, summary in (eval_result.get("runs") or {}).items()
            },
        },
        "provenance_stages": sorted((provenance.get("stages") or {}).keys()),
        "identity": checkpoint.get("identity"),
    }


def render_model_card(export_config: dict) -> str:
    target = export_config["target"]
    drafter = export_config["drafter"]
    training = export_config["training"] or {}
    data = export_config["data"] or {}
    eval_block = export_config["eval"] or {}
    lines = [
        f"# vlm-dflash drafter — {Path(target['model_path']).name}",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| method | {export_config['method']} (block drafter) |",
        f"| target | `{target['model_path']}` ({target['model_type']}) |",
        f"| target fingerprint | `{target['model_fingerprint'][:16]}…` |",
        f"| adapter | {export_config['adapter']} |",
        f"| block size | {drafter['block_size']} |",
        f"| anchors / layers | {drafter['num_anchors']} / {drafter['num_hidden_layers']} |",
        f"| mask token | {drafter['mask_token_id']} |",
        f"| routed target layers | {drafter['target_layer_ids']} |",
        f"| MRoPE sections | {drafter['spatial_rope']['sections']} ({drafter['spatial_rope']['mode']}) |",
        f"| code commit | `{export_config.get('code_commit') or 'n/a'}` |",
    ]
    if data:
        lines += [
            f"| training data | {data.get('num_train')} train / {data.get('num_val')} val from `{data.get('source')}` |",
            f"| data digest | `{str(data.get('rows_sha256'))[:16]}…` |",
        ]
    if training:
        lines += [
            f"| training steps | {training.get('steps')} |",
            f"| block CE (first10 → last10) | {training.get('first10_mean_loss')} → {training.get('last10_mean_loss')} |",
            f"| peak VRAM / wall | {training.get('peak_gpu_mem_gb')} GB / {training.get('wall_seconds')} s |",
        ]
    if eval_block:
        lines += [
            "",
            "## Quality (greedy, exactness-checked)",
            "",
            f"- token-exact vs target-only: **{eval_block.get('token_exact')}** over {eval_block.get('num_prompts')} prompts",
        ]
        for name, summary in (eval_block.get("runs") or {}).items():
            lines.append(
                f"- `{name}`: AL {summary.get('mean_al')}, acceptance rate {summary.get('acceptance_rate')}, "
                f"per-slot {summary.get('per_slot_acceptance')} ({summary.get('role')})"
            )
    lines += [
        "",
        "## Scope / limitations",
        "",
        "- Single-image image-text prompts; greedy decoding only.",
        "- Acceptance numbers depend on the exact target revision and processor recorded above.",
        "- Exported for evaluation/training interop; serving integration is separate roadmap.",
        "",
    ]
    return "\n".join(lines)


def export_run(run_dir: str | Path, *, checkpoint: str = "ckpt_final.pt", out_dir: str | Path | None = None) -> Path:
    """Export one trained draft checkpoint as a self-describing directory."""
    run_dir = Path(run_dir).resolve()
    ckpt_path = run_dir / "train" / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint {ckpt_path}; run the train stage first.")
    checkpoint_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint_obj.get("format") != "vlm-dflash-ckpt-v1":
        raise ValueError(f"Unrecognized checkpoint format: {checkpoint_obj.get('format')!r}")

    export_dir = ensure_dir(Path(out_dir) if out_dir is not None else run_dir / "export")
    from safetensors.torch import save_file

    state = {key: value.contiguous() for key, value in checkpoint_obj["model"].items()}
    save_file(state, str(export_dir / "model.safetensors"), metadata={"format": EXPORT_FORMAT})

    export_config = build_export_config(run_dir, checkpoint_obj, checkpoint)
    write_json(export_dir / "config.json", export_config)
    (export_dir / "model_card.md").write_text(render_model_card(export_config), encoding="utf-8")
    write_json(
        export_dir / "checksums.json",
        {
            "model.safetensors": sha256_file(export_dir / "model.safetensors"),
            "config.json": sha256_file(export_dir / "config.json"),
            "model_card.md": sha256_file(export_dir / "model_card.md"),
        },
    )
    log(f"export: wrote {export_dir} (format={EXPORT_FORMAT}, checkpoint={checkpoint})")
    return export_dir


def load_export(export_dir: str | Path):
    """Rebuild the draft model from an exported directory (no run dir needed)."""
    from safetensors.torch import load_file
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from ..dflash.draft_model import Qwen3DFlashDraftModel

    export_dir = Path(export_dir)
    export_config = read_json(export_dir / "config.json")
    if export_config.get("format") != EXPORT_FORMAT:
        raise ValueError(f"Unrecognized export format in {export_dir}: {export_config.get('format')!r}")
    qwen3_config = Qwen3Config.from_dict(export_config["drafter"]["draft_config"])
    qwen3_config._attn_implementation = export_config["drafter"].get("attention_backend", "flex_attention")
    model = Qwen3DFlashDraftModel(qwen3_config)
    model.load_state_dict(load_file(export_dir / "model.safetensors"))
    model.eval()
    return model, export_config
