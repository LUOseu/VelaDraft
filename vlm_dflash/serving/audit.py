"""Zero-GPU capability audit against the *installed* vLLM source tree."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    INCLUDE_IN_PAPER,
    PROJECT_SCOPE,
    SCHEMA_VERSION,
    sha256_file,
    write_json_atomic,
)
from .profile import describe_profile

VLLM_PIN = "0.22.1"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _line_of(path: Path, pattern: str) -> Optional[int]:
    text = _read(path)
    for index, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            return index
    return None


def _evidence(path: Path, pattern: str, symbol: str, note: str) -> dict[str, Any]:
    line = _line_of(path, pattern)
    return {
        "path": str(path),
        "line": line,
        "symbol": symbol,
        "note": note,
        "found": line is not None,
    }


def _git_commit_from_dist() -> Optional[str]:
    for candidate in (
        Path(sys.prefix) / "vllm-0.22.1.dist-info" / "direct_url.json",
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "vllm-0.22.1.dist-info" / "direct_url.json",
    ):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text())
                if data.get("vcs_info", {}).get("commit_id"):
                    return str(data["vcs_info"]["commit_id"])
            except Exception:
                pass
    return None


def _gpu_probe() -> dict[str, Any]:
    probe: dict[str, Any] = {}
    try:
        import torch

        probe["torch"] = torch.__version__
        probe["torch_cuda"] = torch.version.cuda
        probe["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            probe["gpu_name"] = props.name
            probe["compute_capability"] = f"{props.major}.{props.minor}"
            probe["total_memory_mib"] = int(props.total_memory // (1024 * 1024))
    except Exception as exc:
        probe["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).strip()
        probe["nvidia_smi"] = output.splitlines()
    except Exception as exc:
        probe["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return probe


def _package_versions() -> dict[str, Any]:
    import importlib.metadata as metadata

    versions = {}
    for name in ("vllm", "torch", "transformers", "tokenizers", "numpy", "Pillow", "openai", "fastapi", "uvicorn"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version
    versions["python_executable"] = sys.executable
    versions["platform"] = platform.platform()
    return versions


def _model_identity(model_path: str) -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "chat_template.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    ):
        path = root / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    shards = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        for name in sorted(set(index.get("weight_map", {}).values())):
            path = root / name
            if path.is_file():
                shards[name] = {"size": path.stat().st_size}
    return {
        "model": str(root),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "dtype": config.get("torch_dtype"),
        "vocab_size": config.get("vocab_size"),
        "image_token_id": config.get("image_token_id"),
        "rope_scaling": config.get("rope_scaling"),
        "files": files,
        "shards": shards,
        "complete": bool(files) and bool(shards),
    }


def collect_audit(
    *,
    model_path: str,
    source_commit: Optional[str] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(environment, capabilities)`` without touching a GPU.

    ``evidence`` can carry paths to already executed GPU smoke/measurement
    results; it only changes maturity labels, never invents a supported status.
    """

    import vllm

    env = _package_versions()
    env["vllm_file"] = vllm.__file__
    env["vllm_version"] = getattr(vllm, "__version__", None)
    env["gpu"] = _gpu_probe()
    env["model"] = _model_identity(model_path)
    env["source_commit"] = source_commit or _git_commit_from_dist() or "unknown-wheel-build"
    env["source_commit_note"] = (
        "The installed wheel does not record a VCS commit; source_commit is the "
        "upstream v0.22.1 tag commit supplied by the auditor."
        if (source_commit or _git_commit_from_dist()) is None
        else "VCS commit from direct_url.json or auditor pin."
    )

    root = Path(vllm.__file__).resolve().parent
    evidence_by_id = evidence or {}

    def evidence_list(*items: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for item in items:
            if item:
                out.append(item)
        return out

    spec_conf = root / "config" / "speculative.py"
    ngram = root / "v1" / "spec_decode" / "ngram_proposer.py"
    runner = root / "v1" / "worker" / "gpu_model_runner.py"
    metrics = root / "v1" / "spec_decode" / "metrics.py"
    qwen = root / "model_executor" / "models" / "qwen2_5_vl.py"
    registry = root / "model_executor" / "models" / "registry.py"
    chat_protocol = root / "entrypoints" / "openai" / "chat_completion" / "protocol.py"

    source_evidence = {
        "speculative_config": evidence_list(
            _evidence(spec_conf, r"class SpeculativeConfig", "SpeculativeConfig", "method/model/prompt_lookup fields"),
            _evidence(spec_conf, r'"ngram"', "SpeculativeMethod.ngram", "native ngram method is registered"),
        ),
        "ngram_proposer": evidence_list(
            _evidence(ngram, r"class NgramProposer", "NgramProposer", "CPU N-Gram proposer; no draft model forward"),
            _evidence(ngram, r"_find_longest_matched_ngram", "_find_longest_matched_ngram_and_propose_tokens", "longest suffix n-gram match, k tokens proposed"),
        ),
        "runner_hook": evidence_list(
            _evidence(runner, r"NgramProposer\(", "gpu_model_runner: NgramProposer", "engine constructs the proposer from SpeculativeConfig"),
        ),
        "metrics": evidence_list(
            _evidence(metrics, r"spec_decode_num_drafts", "spec_decode_num_drafts", "aggregate draft/verify counters"),
            _evidence(metrics, r"spec_decode_num_accepted_tokens", "spec_decode_num_accepted_tokens", "accepted token counter"),
        ),
        "qwen2_5_vl": evidence_list(
            _evidence(registry, r"Qwen2_5_VLForConditionalGeneration", "registry: Qwen2_5_VL", "target architecture is registered"),
            _evidence(qwen, r"class Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "multimodal conditional generation"),
            _evidence(qwen, r"def get_mrope_input_positions", "get_mrope_input_positions", "MRoPE positions for image/text tokens"),
            _evidence(qwen, r"def recompute_mrope_positions", "recompute_mrope_positions", "incremental MRoPE recomputation"),
        ),
        "return_token_ids": evidence_list(
            _evidence(chat_protocol, r"return_token_ids", "ChatCompletionRequest.return_token_ids", "vLLM extension returns token IDs over HTTP"),
        ),
    }

    gpu_smoke = evidence_by_id.get("gpu_smoke", {})
    lifecycle = evidence_by_id.get("lifecycle", {})
    correctness = evidence_by_id.get("correctness", {})
    dflash = evidence_by_id.get("dflash", {})

    def _load_evidence(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict) or not item.get("path"):
            return {}
        try:
            return json.loads(Path(str(item["path"])).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    ar_doc = _load_evidence(gpu_smoke.get("ar"))
    ngram_doc = _load_evidence(gpu_smoke.get("ngram"))
    correctness_doc = _load_evidence(correctness)
    lifecycle_doc = _load_evidence(lifecycle.get("stream") or lifecycle.get("cancel"))
    ar_summary = ar_doc.get("summary", {}) if isinstance(ar_doc, dict) else {}
    ngram_summary = ngram_doc.get("summary", {}) if isinstance(ngram_doc, dict) else {}
    ar_full = int(ar_summary.get("num_requests") or 0) >= 40 and int(ar_summary.get("num_errors") or 0) == 0
    ngram_full = int(ngram_summary.get("num_requests") or 0) >= 40 and int(ngram_summary.get("num_errors") or 0) == 0
    correctness_verified = bool(correctness_doc.get("completion_levels", {}).get("SD_CORRECTNESS_PASS"))
    lifecycle_verified = bool(lifecycle_doc.get("completion_levels", {}).get("SERVING_LIFECYCLE_PASS"))
    lifecycle_present = bool(lifecycle.get("stream") or lifecycle.get("cancel"))

    gpu_evidence = evidence_by_id.get("gpu_smoke") or {}
    lifecycle_evidence = evidence_by_id.get("lifecycle") or {}
    ar_maturity = "validated" if ar_full else ("gpu_smoke" if gpu_evidence.get("ar") else "cpu_tested")
    ngram_maturity = "validated" if ngram_full and correctness_verified else ("gpu_smoke" if gpu_evidence.get("ngram") else "cpu_tested")
    lifecycle_maturity = "validated" if lifecycle_verified else ("gpu_smoke" if lifecycle_present else "cpu_tested")
    token_maturity = "validated" if lifecycle_verified and (ar_full or ngram_full) else ("gpu_smoke" if (ar_full or ngram_full) else "cpu_tested")
    metrics_maturity = "validated" if ngram_full and correctness_verified else ("gpu_smoke" if ngram_full else "cpu_tested")

    capabilities = {
        "ar_image": {
            "status": "supported",
            "maturity": ar_maturity,
            "version": VLLM_PIN,
            "evidence": evidence_list(source_evidence["qwen2_5_vl"], gpu_smoke.get("ar")),
            "limits": [
                "TP=1, one A6000, BF16, greedy, prefix caching disabled in the validated config",
                "Qwen2.5-VL image processor is loaded as the fast processor in vLLM 0.22.1",
            ],
        },
        "ngram_image": {
            "status": "supported",
            "maturity": ngram_maturity,
            "version": VLLM_PIN,
            "evidence": evidence_list(source_evidence["speculative_config"], source_evidence["ngram_proposer"], source_evidence["runner_hook"], gpu_smoke.get("ngram")),
            "limits": [
                "CPU N-Gram proposer matches token-id suffixes; it does not use image features directly",
                "N-Gram forces synchronous scheduling in vLLM 0.22.1 (observed startup warning)",
                "draft proposals are verified by the same target model; no lossless claim beyond the measured greedy gate",
            ],
        },
        "dflash_image": {
            "status": "experimental",
            "maturity": "blocked",
            "version": VLLM_PIN,
            "evidence": evidence_list(dflash.get("wp5"), dflash.get("source")),
            "limits": [
                "WP5 reports AL 1.4059 in vLLM vs 1.6866/1.6924 in-repo (-17% gap), unresolved",
                "vLLM DFlash draft path uses 1-D RoPE while the source export may use spatial MRoPE",
                "same-engine reruns are not token-exact (12/32), so a token-exact serving gate is not established",
                "DFlash is out of scope for the first N-Gram delivery",
            ],
        },
        "stream": {
            "status": "supported",
            "maturity": lifecycle_maturity,
            "version": VLLM_PIN,
            "evidence": evidence_list(lifecycle.get("stream"), source_evidence["return_token_ids"]),
            "limits": ["SSE chunk timing is a client-side boundary; engine token timestamps are not exposed"],
        },
        "cancel": {
            "status": "experimental" if not lifecycle.get("cancel") else "supported",
            "maturity": lifecycle_maturity,
            "version": VLLM_PIN,
            "evidence": evidence_list(lifecycle.get("cancel")),
            "limits": ["client disconnect is the cancellation signal; per-request engine abort logs depend on server log level"],
        },
        "round_trace": {
            "status": "unsupported",
            "maturity": "implemented",
            "version": VLLM_PIN,
            "evidence": evidence_list(source_evidence["metrics"]),
            "limits": ["vLLM exposes aggregate draft/accept counters, not a public per-round propose/commit hook"],
        },
        "return_token_ids": {
            "status": "supported",
            "maturity": token_maturity,
            "version": VLLM_PIN,
            "evidence": evidence_list(source_evidence["return_token_ids"]),
            "limits": ["vLLM extension; standard OpenAI clients ignore the extra field"],
        },
        "metrics": {
            "status": "supported",
            "maturity": metrics_maturity,
            "version": VLLM_PIN,
            "evidence": evidence_list(source_evidence["metrics"]),
            "limits": ["counters are global to the engine/session window; no request tag on aggregate counters"],
        },
        "wire_profile": {
            "status": "supported",
            "maturity": "cpu_tested",
            "version": "1.0",
            "evidence": evidence_list(
                {"path": "vlm_dflash/serving/profile.py", "note": "explicit serving wire profile and normalizer"},
                {"path": "tests/serving/fixtures/wire_profile_record.json", "note": "real-shaped CPU fixture"},
            ),
            "limits": [
                "profile id veladraft-serving-wire is not interchangeable with the advisor profile despite the shared schema name",
                "request/window/global counters keep their recorded scope; unavailable counters remain null",
            ],
        },
    }
    capabilities_doc = {
        "schema_version": SCHEMA_VERSION,
        "project_scope": PROJECT_SCOPE,
        "include_in_paper": INCLUDE_IN_PAPER,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).date().isoformat(),
        "vllm_pin": VLLM_PIN,
        "environment": env,
        "source_evidence": source_evidence,
        "correctness_evidence": correctness,
        "capabilities": capabilities,
        "serving_profile": describe_profile(),
    }
    return env, capabilities_doc


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit the installed vLLM for serving capabilities")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--gpu-smoke-ar", default=None)
    parser.add_argument("--gpu-smoke-ngram", default=None)
    parser.add_argument("--lifecycle", default=None)
    parser.add_argument("--correctness", default=None)
    parser.add_argument("--dflash-wp5", default=None)
    args = parser.parse_args(argv)
    evidence = {
        "gpu_smoke": {
            "ar": {"path": args.gpu_smoke_ar, "note": "AR GPU smoke"} if args.gpu_smoke_ar else None,
            "ngram": {"path": args.gpu_smoke_ngram, "note": "N-Gram GPU smoke"} if args.gpu_smoke_ngram else None,
        },
        "lifecycle": {"stream": {"path": args.lifecycle}, "cancel": {"path": args.lifecycle}} if args.lifecycle else {},
        "correctness": {"path": args.correctness} if args.correctness else {},
        "dflash": {"wp5": {"path": args.dflash_wp5}} if args.dflash_wp5 else {},
    }
    _, doc = collect_audit(model_path=args.model, source_commit=args.source_commit, evidence=evidence)
    write_json_atomic(args.out, doc)
    print(json.dumps({"out": args.out, "vllm": doc["vllm_pin"], "capabilities": {k: v["status"] for k, v in doc["capabilities"].items()}}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
