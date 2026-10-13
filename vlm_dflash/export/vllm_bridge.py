"""vLLM bridge: emit a speculators-format DFlash draft from a repo export.

vLLM (0.22.1+) ships a native ``dflash`` speculative-decoding method whose
draft directory follows the ``vllm-project/speculators`` layout: a
``config.json`` carrying ``speculators_model_type: "dflash"``, the target
verifier path, the draft layer config, the mask token, and the auxiliary
hidden-state layer ids, next to the draft weights. This module converts a
vlm-dflash export directory into that layout.

The weights are copied unchanged. The vlm-dflash export already stores the
draft in the exact tensor names the vLLM ``DFlashDraftModel`` loads
(``fc.weight``, ``hidden_norm.weight``, ``norm.weight``, ``layers.*``), and it
does not carry embedding / LM-head copies: vLLM shares the target's
``embed_tokens`` and ``lm_head`` with the draft at runtime when the checkpoint
omits them, which is the same convention this repo's serving uses.

Layer-id convention: vlm-dflash captures target features from
``output_hidden_states[layer_id + 1]`` (post-block hidden states; index 0 is
the embedding output) and records those ``target_layer_ids`` in the export.
The speculators format records the same indices as
``aux_hidden_state_layer_ids`` (vLLM's aux index equals the HF
``output_hidden_states`` index). vLLM then derives its internal
``dflash_config.target_layer_ids`` by subtracting one, reproducing the
vlm-dflash ids verbatim. The mapping here is ``aux = layer_id + 1``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..utils import ensure_dir, log, read_json, sha256_file, write_json

VLLM_PIN = "0.22.1"


def expected_weight_keys(num_layers: int) -> set[str]:
    """Tensor names one vlm-dflash DFlash export carries for ``num_layers`` layers."""

    keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    for layer in range(num_layers):
        prefix = f"layers.{layer}"
        keys.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    return keys


def build_speculators_config(
    export_config: dict,
    *,
    target_path: str | None = None,
    speculative_tokens: int | None = None,
) -> dict:
    """Map a vlm-dflash export config onto the vLLM speculators draft layout."""

    if export_config.get("format") != "vlm-dflash-export-v1":
        raise ValueError(
            f"unexpected export format {export_config.get('format')!r}; "
            "expected 'vlm-dflash-export-v1'"
        )
    method = str(export_config.get("method", ""))
    if method != "dflash":
        raise ValueError(
            f"vLLM's native dflash spec-decode path serves DFlash block drafters only; "
            f"this export records method {method!r}. Use the in-repo verifier or a "
            "runtime/algorithm pair with a registered vLLM integration."
        )
    drafter = export_config.get("drafter") or {}
    draft_config = dict(drafter.get("draft_config") or {})
    if not draft_config:
        raise ValueError("export config has no 'drafter.draft_config' block")
    if bool(draft_config.get("attention_bias")):
        raise ValueError("the vLLM DFlash draft path assumes attention_bias=false")

    block_size = int(drafter["block_size"])
    if speculative_tokens is None:
        num_spec = block_size - 1
    else:
        num_spec = int(speculative_tokens)
    if num_spec <= 0:
        raise ValueError(f"speculative_tokens must be positive, got {num_spec}")

    layer_ids = [int(x) for x in (drafter.get("target_layer_ids") or [])]
    if not layer_ids:
        raise ValueError("export config records no 'drafter.target_layer_ids'")
    if sorted(layer_ids) != layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"target_layer_ids must be unique and ascending, got {layer_ids}")

    source_target = str((export_config.get("target") or {}).get("model_path") or "")
    target = str(target_path or source_target)
    if not target:
        raise ValueError("no target path: pass --target or use an export that records one")

    layer_config = {
        key: value for key, value in draft_config.items() if key not in ("architectures", "dflash_config")
    }
    vocab_size = int(layer_config.get("vocab_size") or 0)
    hidden_size = int(layer_config.get("hidden_size") or 0)
    if vocab_size <= 0 or hidden_size <= 0:
        raise ValueError("draft config must record positive vocab_size and hidden_size")

    config = {
        "speculators_model_type": "dflash",
        "speculators_config": {
            "proposal_methods": [
                {
                    "speculative_tokens": num_spec,
                    "draft_arch": "qwen3-dflash-block",
                }
            ],
            "verifier": {"name_or_path": target},
        },
        "transformer_layer_config": layer_config,
        "draft_vocab_size": vocab_size,
        "target_hidden_size": hidden_size,
        "mask_token_id": int(drafter["mask_token_id"]),
        # vLLM aux index == HF output_hidden_states index == vlm-dflash layer_id + 1.
        "aux_hidden_state_layer_ids": [layer + 1 for layer in layer_ids],
    }
    return config


def check_weight_file(export_dir: Path, draft_config: dict, *, num_features: int) -> dict:
    """Validate the safetensors layout against what the vLLM drafter loads."""

    from safetensors import safe_open

    weights_path = export_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing weights: {weights_path}")
    with safe_open(weights_path, framework="pt") as handle:
        keys = set(handle.keys())
        fc_shape = tuple(handle.get_slice("fc.weight").get_shape()) if "fc.weight" in keys else ()
    hidden = int(draft_config.get("hidden_size") or 0)
    num_layers = int(draft_config.get("num_hidden_layers") or 0)
    expected = expected_weight_keys(num_layers)
    missing = sorted(expected - keys)
    if missing:
        raise ValueError(f"weights are missing expected tensors: {missing}")
    own_embed = "embed_tokens.weight" in keys
    own_head = "lm_head.weight" in keys or "model.lm_head.weight" in keys
    unexpected = sorted(
        key
        for key in keys - expected
        if key not in ("embed_tokens.weight", "lm_head.weight")
    )
    if unexpected:
        raise ValueError(f"weights carry tensors the bridge does not understand: {unexpected}")
    if fc_shape != (hidden, hidden * num_features):
        raise ValueError(
            f"fc.weight has shape {fc_shape}; expected {(hidden, hidden * num_features)} "
            f"(hidden={hidden}, features={num_features})"
        )
    return {
        "num_tensors": len(keys),
        "own_embed_tokens": own_embed,
        "own_lm_head": own_head,
        "fc_shape": list(fc_shape),
    }


def export_vllm_draft(
    export_dir: str | Path,
    out_dir: str | Path,
    *,
    target_path: str | None = None,
    speculative_tokens: int | None = None,
    force: bool = False,
    copy_weights: bool = True,
) -> dict:
    """Convert ``export_dir`` (vlm-dflash export) into ``out_dir`` (vLLM draft)."""

    export_dir = Path(export_dir)
    out_dir = Path(out_dir)
    export_config = read_json(export_dir / "config.json")
    config = build_speculators_config(
        export_config,
        target_path=target_path,
        speculative_tokens=speculative_tokens,
    )
    drafter = export_config["drafter"]
    draft_config = drafter["draft_config"]
    num_features = len(config["aux_hidden_state_layer_ids"])
    weight_report = check_weight_file(export_dir, draft_config, num_features=num_features)

    out_dir = ensure_dir(out_dir)
    write_json(out_dir / "config.json", config)
    if copy_weights:
        source = export_dir / "model.safetensors"
        shutil.copyfile(source, out_dir / "model.safetensors")

    spatial_rope = dict(drafter.get("spatial_rope") or {})
    rope_note = (
        f"enabled={bool(spatial_rope.get('enabled'))} mode={spatial_rope.get('mode')} "
        f"sections={spatial_rope.get('sections')}"
    )
    weights_sha = sha256_file(out_dir / "model.safetensors") if copy_weights else None
    summary = {
        "format": "vlm-dflash-vllm-draft-v1",
        "vllm_pin": VLLM_PIN,
        "source_export": str(export_dir),
        "source_method": export_config.get("method"),
        "source_created": export_config.get("created"),
        "weights_sha256": weights_sha,
        "block_size": int(drafter["block_size"]),
        "speculative_tokens": config["speculators_config"]["proposal_methods"][0]["speculative_tokens"],
        "mask_token_id": config["mask_token_id"],
        "target_layer_ids": [int(x) for x in drafter["target_layer_ids"]],
        "aux_hidden_state_layer_ids": config["aux_hidden_state_layer_ids"],
        "spatial_rope": spatial_rope,
        "weights": weight_report,
    }
    write_json(out_dir / "bridge.json", summary)

    note_lines = [
        "# vLLM DFlash draft (converted from a vlm-dflash export)",
        "",
        f"- vLLM pin: `{VLLM_PIN}` (native `dflash` speculative-decoding method, speculators layout)",
        f"- source export: `{export_dir}` (`created={export_config.get('created')}`)",
        f"- mask token: `{config['mask_token_id']}`; block size {drafter['block_size']}"
        f" -> `speculative_tokens={summary['speculative_tokens']}`",
        f"- target layers: vlm-dflash `{summary['target_layer_ids']}`"
        f" -> aux `{summary['aux_hidden_state_layer_ids']}` (HF `output_hidden_states` indices)",
        f"- spatial rope in the source export: {rope_note}",
        f"- weights: {weight_report['num_tensors']} tensors"
        f" (own embed_tokens: {weight_report['own_embed_tokens']}, own lm_head: {weight_report['own_lm_head']};"
        " omitted copies are shared from the verifier at runtime)",
        "",
        "## Serving",
        "",
        "```console",
        f"vllm serve {config['speculators_config']['verifier']['name_or_path']} \\",
        f"    --speculative-config '{{\"method\": \"dflash\", \"model\": \"{out_dir}\", "
        f"\"num_speculative_tokens\": {summary['speculative_tokens']}}}'",
        "```",
        "",
        "The draft directory alone is also self-describing: `LLM(model=\"<this dir>\")`"
        " flips to the recorded verifier and builds the speculative config from it.",
        "",
        "## Caveats",
        "",
        "- vLLM's DFlash proposer applies a 1-D RoPE over the target positions; it does"
        " not implement the MRoPE (`spatial_rope`) rotary the in-repo serving path uses"
        " when the source export has `spatial_rope.enabled=true`. Acceptance measured"
        " under vLLM must therefore be compared against the in-repo 1-D-convention run,"
        " not against the MRoPE-convention AL from the model card.",
        "- Token-exactness of the deployment is enforced by the verifier: compare greedy"
        " output against target-only greedy in the same runtime before quoting speed.",
    ]
    (out_dir / "VLLM_DRAFT.md").write_text("\n".join(note_lines) + "\n")
    log(f"vllm bridge: wrote {out_dir} (weights_sha256={str(weights_sha)[:12]}...)")
    return summary
