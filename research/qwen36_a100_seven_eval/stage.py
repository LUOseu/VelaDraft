"""One fresh vLLM 0.32 engine for one stage of one job, on the GPU the lane leased.

Stages: ``reference`` (untimed native AR, natural stop; supplies multi-turn history and the frozen
prompt token ids), ``ar_r<k>`` / ``sd_r<k>`` (timed), ``stats`` (SD with official counters enabled,
one untimed pass for native acceptance statistics). Rows are written per MMSpec topic shard:
``raw/mmspec/<owner>/<protocol>__<stage>__<topic>.jsonl`` appears only when the whole shard is done,
so a resumed stage re-runs exactly the missing topics on a fresh engine.

Timing comes from ``vlm_dflash.eval.native_runtime.request``: image read and RGB decode, template,
processing, vision, prefill, decode and detokenization; engine load, compile and warmup are excluded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path

from research.qwen36_a100_seven_eval.settings import (
    CATEGORIES,
    SPEC_METHOD,
    inference_settings,
    read_json,
    save_json,
)

ROW_DROP = ("prompt", "messages", "prompt_token_ids")


def shard_path(run: Path, owner: str, protocol: str, stage: str, topic: str) -> Path:
    return (
        Path(run)
        / "raw"
        / "mmspec"
        / owner
        / f"{protocol}__{stage}__{topic.replace(' ', '_')}.jsonl"
    )


def owner_of(method: str, stage: str) -> str:
    return "_reference" if stage == "reference" else method


def missing_topics(
    run: Path, sessions: list[dict], owner: str, protocol: str, stage: str
) -> list[str]:
    topics = [t for t in CATEGORIES if any(s["topic"] == t for s in sessions)]
    return [
        t for t in topics if not shard_path(run, owner, protocol, stage, t).exists()
    ]


def load_reference(run: Path, sessions: list[dict]) -> dict:
    rows = {}
    for topic in {s["topic"] for s in sessions}:
        path = shard_path(run, "_reference", "natural", "reference", topic)
        if not path.exists():
            raise FileNotFoundError(f"reference shard missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[(row["id"], row["turn"])] = row
    return rows


def environment_for(manifest: dict, method: str, arm: str, protocol: str) -> dict:
    """Process environment of the engine (also exported before vLLM is imported)."""
    settings = inference_settings()
    engine = manifest["engine"]
    if arm == "ar" and method == "ar_default":
        return {
            "unset": [
                "Q36_V032_ARM",
                "Q36_MOE_POLICY",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_GDN_DECODE_KERNEL",
                "VLLM_TUNED_CONFIG_FOLDER",
            ],
            "set": {"VLLM_BATCH_INVARIANT": "0"},
        }
    if arm == "ar":
        ref = settings["ar_reference"]
        values = dict(
            ref["environment"], VLLM_GDN_DECODE_KERNEL=engine["gdn_decode_kernel"]
        )
        return {"set": values, "unset": ref["unset_environment"]}
    values = {
        k: v
        for k, v in settings["environment"].items()
        if k != "VLLM_TUNED_CONFIG_FOLDER"
    }
    values.update(settings["methods"][method].get("environment", {}))
    values["VLLM_GDN_DECODE_KERNEL"] = engine["gdn_decode_kernel"]
    values["Q36_MOE_POLICY"] = engine["moe_policy"]
    unset = ["Q36_DYNAMIC_SD"]
    if engine.get("tuned_config_folder"):
        values["VLLM_TUNED_CONFIG_FOLDER"] = engine["tuned_config_folder"]
    else:
        unset.append("VLLM_TUNED_CONFIG_FOLDER")
    if "Q36_SWEEP_K" in values:
        values["Q36_SWEEP_K"] = str(manifest["protocol"]["draft_length"])
    return {"set": values, "unset": unset}


def engine_kwargs(
    manifest: dict, method: str, arm: str, *, stats: bool = False
) -> dict:
    settings = inference_settings()
    engine, protocol = manifest["engine"], manifest["protocol"]
    kwargs = {
        "model": manifest["target"]["path"],
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": protocol["max_model_len"],
        "max_num_seqs": 1,
        "max_num_batched_tokens": engine["max_num_batched_tokens"],
        "gpu_memory_utilization": engine["gpu_memory_utilization"],
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "limit_mm_per_prompt": {"image": 1},
        "enforce_eager": False,
        "generation_config": "vllm",
        "disable_log_stats": not stats,
        "mm_processor_kwargs": {
            "min_pixels": protocol["min_pixels"],
            "max_pixels": protocol["max_pixels"],
        },
    }
    if arm == "ar" and method == "ar_default":
        return (
            kwargs  # stock vLLM defaults: automatic attention backend, default graphs
        )
    kwargs["attention_config"] = {"backend": "FLASH_ATTN"}
    if arm == "ar":
        kwargs["worker_extension_cls"] = settings["ar_reference"]["worker"]
        return kwargs
    spec = settings["methods"][method]
    k = protocol["draft_length"]
    kwargs["worker_extension_cls"] = spec["worker"]
    kwargs["speculative_config"] = {
        "model": manifest["exports"][method]["path"],
        "method": SPEC_METHOD[method],
        "num_speculative_tokens": k,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN",
    }
    kwargs["compilation_config"] = {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [k + 1],
        "max_cudagraph_capture_size": k + 1,
    }
    return kwargs


def sampling(manifest: dict, protocol: str, SamplingParams):
    p = manifest["protocol"]
    if protocol == "natural":
        return SamplingParams(
            temperature=0,
            max_tokens=p["max_new_tokens"],
            seed=p["seed"],
            stop_token_ids=manifest["target"]["stop_token_ids"],
            ignore_eos=False,
        )
    if protocol == "fixed512":
        # Explicit stop tokens removed and EOS ignored: every output is exactly fixed_tokens long and may
        # continue past EOS; never used for natural-answer quality.
        return SamplingParams(
            temperature=0, max_tokens=p["fixed_tokens"], seed=p["seed"], ignore_eos=True
        )
    raise ValueError(protocol)


def compact(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k not in ROW_DROP}
    out["prompt_tokens"] = len(row["prompt_token_ids"])
    out["prompt_token_ids_sha256"] = hashlib.sha256(
        json.dumps(row["prompt_token_ids"]).encode()
    ).hexdigest()
    out["output_tokens"] = len(row["tokens"])
    return out


def run_stage(args) -> int:
    run = Path(args.run_dir)
    manifest = read_json(run / "manifest.json")
    sessions = read_json(run / "sessions.json")
    arm = "ar" if args.stage == "reference" or args.stage.startswith("ar") else "sd"
    if args.method == "ar_default" and arm != "ar":
        raise ValueError("ar_default has AR stages only")
    owner = owner_of(args.method, args.stage)
    protocol = "natural" if args.stage in ("reference", "stats") else args.protocol
    topics = missing_topics(run, sessions, owner, protocol, args.stage)
    meta_path = run / "stages" / owner / f"{protocol}__{args.stage}.json"
    if not topics:
        print("STAGE_ALREADY_COMPLETE", owner, protocol, args.stage, flush=True)
        return 0
    meta = {
        "status": "running",
        "method": args.method,
        "arm": arm,
        "stage": args.stage,
        "protocol": protocol,
        "topics": topics,
        "gpu": os.environ.get("LANE_GPU"),
        "started": time.strftime("%F %T"),
        "attempts": [],
    }
    previous = read_json(meta_path, {})
    meta["attempts"] = previous.get("attempts", []) + [
        {"started": meta["started"], "topics": topics}
    ]
    save_json(meta_path, meta)
    env = environment_for(manifest, args.method, arm, protocol)
    for key in env["unset"]:
        os.environ.pop(key, None)
    os.environ.update(env["set"])
    llm = None
    lease = None
    windows = []
    staged = []
    entered = False
    try:
        for package, expected_version in manifest["versions"].items():
            if importlib.metadata.version(package) != expected_version:
                raise ValueError(f"{package} version differs from the frozen manifest")
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        from vlm_dflash.eval.native_resources import GPULease
        from vlm_dflash.eval.native_runtime import metric_delta, metrics, request

        if arm == "sd":
            hook = inference_settings()["methods"][args.method].get(
                "install_before_engine"
            )
            if hook:
                module, _, name = hook.rpartition(".")
                getattr(importlib.import_module(module), name)()
        stats = args.stage == "stats"
        kwargs = engine_kwargs(manifest, args.method, arm, stats=stats)
        meta.update(
            engine_args=copy.deepcopy(kwargs),
            environment=env,
            versions={
                n: importlib.metadata.version(n)
                for n in ("vllm", "torch", "transformers")
            },
        )
        save_json(meta_path, meta)
        gpu = int(os.environ["LANE_GPU"])
        lease = GPULease(
            gpu,
            run
            / "resources"
            / f"{owner}__{protocol}__{args.stage}__{int(time.time())}.jsonl",
        )
        lease.path.parent.mkdir(parents=True, exist_ok=True)
        lease.__enter__()
        entered = True
        gpu_id = hashlib.sha256(lease.uuid.encode()).hexdigest()
        binding = run / "bindings" / f"{owner}__{protocol}.json"
        previous_binding = read_json(binding)
        if previous_binding and previous_binding["gpu_id"] != gpu_id:
            raise ValueError("resume requires the original physical GPU for this job")
        save_json(binding, {"gpu_id": gpu_id, "index": gpu})
        llm = LLM(**copy.deepcopy(kwargs))
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            meta["worker"] = llm.collective_rpc("backend_metadata", timeout=180)[0]
        except Exception as error:
            # Only stock AR intentionally has no audit worker.
            if args.method != "ar_default":
                raise
            meta["worker"] = {"unavailable": str(error)[:300]}
        if arm == "sd":
            meta["method_metadata"] = llm.collective_rpc(
                "method_metadata", timeout=180
            )[0]
            got = meta.get("worker", {}).get("speculative_tokens")
            if got != manifest["protocol"]["draft_length"]:
                raise ValueError(
                    f"engine mounted K={got}, expected {manifest['protocol']['draft_length']}"
                )
        save_json(meta_path, meta)
        processor = AutoProcessor.from_pretrained(
            manifest["target"]["path"],
            local_files_only=True,
            min_pixels=manifest["protocol"]["min_pixels"],
            max_pixels=manifest["protocol"]["max_pixels"],
        )
        params = sampling(manifest, protocol, SamplingParams)
        plan = {
            "system_prompt": manifest["protocol"]["system_prompt"],
            "chat_template_kwargs": manifest["protocol"]["chat_template_kwargs"],
        }
        reference = {} if args.stage == "reference" else load_reference(run, sessions)
        todo = [s for s in sessions if s["topic"] in topics]
        if args.stage != "reference":
            warm = []
            for topic in topics:
                warm += [s for s in todo if s["topic"] == topic][
                    : manifest["protocol"]["warmup_sessions_per_topic"]
                ]
            for session in warm:
                for turn in range(len(session["questions"])):
                    request(
                        llm,
                        processor,
                        params,
                        torch,
                        Image,
                        session,
                        turn,
                        reference,
                        plan,
                    )
        for topic in topics:
            rows = []
            began = time.time_ns()
            for session in [s for s in todo if s["topic"] == topic]:
                for turn in range(len(session["questions"])):
                    before = metrics(llm) if stats else None
                    row = request(
                        llm,
                        processor,
                        params,
                        torch,
                        Image,
                        session,
                        turn,
                        reference,
                        plan,
                    )
                    if stats:
                        row["official_counters"] = metric_delta(before, metrics(llm))
                    if args.stage == "reference":
                        reference[(session["id"], turn)] = row
                    elif (
                        row["prompt_token_ids"]
                        != reference[(session["id"], turn)]["prompt_token_ids"]
                    ):
                        raise ValueError(
                            f"{session['id']}/{turn}: physical prompt ids differ from the reference"
                        )
                    if protocol == "fixed512" and (
                        len(row["tokens"]) != manifest["protocol"]["fixed_tokens"]
                        or row["finish_reason"] != "length"
                    ):
                        raise ValueError("fixed-length contract failed")
                    full = dict(
                        row,
                        topic=topic,
                        stage=args.stage,
                        protocol=protocol,
                        method=args.method,
                        arm=arm,
                        gpu_id=gpu_id,
                    )
                    rows.append(full if args.stage == "reference" else compact(full))
            windows.append((began, time.time_ns()))
            path = shard_path(run, owner, protocol, args.stage, topic)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            partial.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            staged.append((partial, path))
            meta.setdefault("completed_topics", []).append(topic)
            save_json(meta_path, meta)
        meta["status"] = "complete"
    except BaseException as error:  # noqa: BLE001 - record failed attempts, including interrupts
        meta.update(
            status="failed",
            error=f"{type(error).__name__}: {error}"[:2000],
            traceback=traceback.format_exc()[-6000:],
        )
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as error:  # noqa: BLE001 - still release the GPU lease
                meta.update(status="failed", shutdown_error=type(error).__name__)
        if entered:
            try:
                lease.__exit__(None, None, None)
                meta["resources"] = lease.validate(timed_windows=windows or None)
            except Exception as error:  # noqa: BLE001 - reject evidence on any monitor failure
                meta.update(status="invalid_resources", resources_error=str(error))
        # Only validated, successful attempts become visible to readers and publishers.
        for partial, path in staged:
            if meta["status"] == "complete":
                os.replace(partial, path)
            else:
                partial.rename(path.with_name(path.name + f".invalid{time.time_ns()}"))
        meta["ended"] = time.strftime("%F %T")
        save_json(meta_path, meta)
    print("STAGE_DONE", owner, protocol, args.stage, meta["status"], flush=True)
    return 0 if meta["status"] == "complete" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--protocol", choices=("natural", "fixed512"), default="natural"
    )
    parser.add_argument("--stage", required=True)
    return run_stage(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
