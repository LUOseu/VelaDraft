"""Diagnostic (not a measurement): stats-enabled SD engine on a few smoke prompts.

Prints official acceptance counters (overall and per position) and the actual
runtime wiring read from the workers: draft rotary module, KV cache groups and
block sizes, target auxiliary hidden-state layers.

  Q36_DIAG_BOOTSTRAP=<bootstrap_<method>.py> PYTHONPATH=<repo>:<this dir>/diag_bootstrap:<this dir> \
  CUDA_VISIBLE_DEVICES=0,1 python diag_acceptance.py --draft <export> --target <Qwen3.6> \
      --source <smoke jsonl> --images-dir <images> [--sessions 6] [--max-tokens 256]
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import contract


class Q36Diag:
    def q36_diag(self):
        runner = self.model_runner
        drafter = getattr(runner, "drafter", None)
        info = {}
        model = runner.model
        lm = model.get_language_model() if hasattr(model, "get_language_model") else model
        inner = getattr(lm, "model", None)
        info["target_aux_layers"] = list(getattr(inner, "aux_hidden_state_layers", ()) or ())
        groups = []
        for group in runner.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            groups.append({"spec": type(spec).__name__, "block_size": getattr(spec, "block_size", None),
                           "layers": len(group.layer_names), "first": group.layer_names[:2], "last": group.layer_names[-2:]})
        info["kv_groups"] = groups
        if drafter is not None:
            rotary = drafter.model.model.layers[0].self_attn.rotary_emb
            info["drafter"] = {"class": type(drafter).__name__, "model": type(drafter.model).__name__,
                               "block_size": drafter.block_size, "uses_mrope": drafter.uses_mrope,
                               "rotary": type(rotary).__name__, "rotary_dim": getattr(rotary, "rotary_dim", None),
                               "head_size": getattr(rotary, "head_size", None),
                               "mrope_section": getattr(rotary, "mrope_section", None),
                               "mrope_interleaved": getattr(rotary, "mrope_interleaved", None),
                               "is_neox": getattr(rotary, "is_neox_style", None),
                               "attn_layer_names": sorted(getattr(drafter, "_draft_attn_layer_names", ()))[:4],
                               "parallel_drafting_token_id": getattr(drafter, "parallel_drafting_token_id", None),
                               "embed_shared": drafter.model.model.embed_tokens is inner.embed_tokens,
                               "head_shared": drafter.model.lm_head is lm.lm_head}
        return info


def main():
    parser = argparse.ArgumentParser()
    for name in ("draft", "target", "source", "images-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--system-prompt")
    parser.add_argument("--graph", choices=["piecewise8", "default", "full_decode_only", "piecewise1_8"],
                        default="piecewise8")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--ar", action="store_true", help="no speculative config (AR timing baseline)")
    parser.add_argument("--repeat", type=int, default=2, help="passes over the sessions; the first is warmup")
    args = parser.parse_args()
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    tp = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    kwargs = dict(model=args.target, dtype="bfloat16", tensor_parallel_size=tp, max_model_len=8192, max_num_seqs=1,
                  max_num_batched_tokens=4096, gpu_memory_utilization=0.85, enable_prefix_caching=False,
                  mm_processor_cache_gb=0, limit_mm_per_prompt={"image": 1}, enforce_eager=False,
                  generation_config="vllm", disable_log_stats=False,
                  mm_processor_kwargs=dict(min_pixels=contract.IMAGE_MIN_PIXELS, max_pixels=contract.IMAGE_MAX_PIXELS),
                  worker_extension_cls="diag_acceptance.Q36Diag")
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    graphs = {"piecewise8": dict(mode=3, cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[8],
                                 max_cudagraph_capture_size=8),
              "piecewise1_8": dict(mode=3, cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[1, 8],
                                   max_cudagraph_capture_size=8),
              "full_decode_only": dict(mode=3, cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=[1, 8],
                                       max_cudagraph_capture_size=8)}
    if args.graph != "default":
        kwargs["compilation_config"] = graphs[args.graph]
    if not args.ar:
        kwargs.update(attention_config={"backend": "FLASH_ATTN"},
                      speculative_config=dict(model=args.draft, method="dflash",
                                              num_speculative_tokens=contract.SPECULATIVE_TOKENS,
                                              draft_tensor_parallel_size=tp, attention_backend="FLASH_ATTN",
                                              use_local_argmax_reduction=False))
        if args.block_size:
            kwargs["block_size"] = args.block_size
    llm = LLM(**kwargs)
    print("DIAG_WIRING " + json.dumps(llm.collective_rpc("q36_diag")[0], default=str), flush=True)
    processor = AutoProcessor.from_pretrained(args.target, min_pixels=contract.IMAGE_MIN_PIXELS,
                                              max_pixels=contract.IMAGE_MAX_PIXELS)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, stop_token_ids=contract.TARGET_STOP_TOKEN_IDS)
    rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()][:args.sessions]

    def counters():
        out = {}
        for metric in llm.get_metrics():
            if metric.name.startswith("vllm:spec_decode"):
                out[metric.name] = getattr(metric, "values", None) or getattr(metric, "value", None)
        return out

    import time
    for row in rows * args.repeat:
        question = row["conversations"][0]["value"]
        messages = ([{"role": "system", "content": [{"type": "text", "text": args.system_prompt}]}]
                    if args.system_prompt else [])
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]})
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               **contract.TARGET_CHAT_TEMPLATE_KWARGS)
        image = Image.open(io.BytesIO((Path(args.images_dir) / row["images"][0]).read_bytes())).convert("RGB")
        before = counters()
        start = time.perf_counter()
        output = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}], params, use_tqdm=False)[0]
        wall = time.perf_counter() - start
        after = counters()
        delta = {}
        for key, value in after.items():
            old = before.get(key)
            if isinstance(value, list):
                delta[key] = [a - (old[i] if old else 0) for i, a in enumerate(value)]
            elif isinstance(value, (int, float)):
                delta[key] = value - (old or 0)
        drafts = delta.get("vllm:spec_decode_num_drafts", 0)
        accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
        print("DIAG_ROW " + json.dumps({"id": row["id"], "wall": round(wall, 3),
                                        "tokens": len(output.outputs[0].token_ids),
                                        "drafts": drafts, "accepted": accepted,
                                        "acceptance_length": 1 + accepted / drafts if drafts else None,
                                        "per_position": delta.get("vllm:spec_decode_num_accepted_tokens_per_pos"),
                                        "text": output.outputs[0].text[:120]}), flush=True)


if __name__ == "__main__":
    main()
