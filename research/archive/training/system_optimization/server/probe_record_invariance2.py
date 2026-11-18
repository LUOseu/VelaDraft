"""Probe 2: characterize the batched-vs-single extraction difference.

Ran after probe 1 showed:
  - same call repeated: bitwise identical; reversed order: identical;
  - 8-row group vs mixed-composition 8-row group: identical;
  - 8-row group vs single-row fetch: hidden differs (max_abs ~48).

This probe measures:
  A) padded length per fetch call (prepare -> input length), for several groups;
  B) row0 extracted alone vs in groups whose *padded max length* differs a lot;
  C) a detailed diff report (count / first index / values) for one pair.
"""
import json
import os
import sys
from pathlib import Path

import torch

REPO = "/home/lyh/VelaDraft-systems"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("VLM_DFLASH_TARGET", "/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct")
os.environ.setdefault("VLM_DFLASH_LLAVA_JSON", "/home/lyh/vlm-systems/data/llava_v1safe.json")
os.environ.setdefault("VLM_DFLASH_COCO_IMAGES", "/home/dhz/Data/COCO-caption/train2017")

from vlm_dflash.config import load_config
from vlm_dflash.target.online import OnlineFeatureSource

RUN = Path("/home/lyh/vlm-systems/runs/s1b/real256")


def diff_report(tag, x, y):
    a, b = x["hidden"], y["hidden"]
    if torch.equal(a, b):
        print(f"  {tag}: hidden bitwise identical", flush=True)
        return
    d = (a.float() - b.float()).abs()
    neq = int((a != b).sum())
    flat = d.flatten()
    idx = int(flat.argmax())
    row, col = divmod(idx, d.shape[1])
    print(f"  {tag}: hidden DIFF elems={neq}/{a.numel()} max_abs={flat[idx]:.4e} "
          f"at (pos={row}, dim={col}) values {float(a[row, col]):.4f} vs {float(b[row, col]):.4f}",
          flush=True)
    # per-position diff profile
    per_pos = d.max(dim=1).values
    nz = (per_pos > 0).nonzero().flatten()
    if nz.numel():
        print(f"     differing positions: {nz[:5].tolist()} ... {nz[-5:].tolist()} "
              f"(n={int(nz.numel())} of {d.shape[0]})", flush=True)
    # do input ids/loss masks differ?
    for k in ("input_ids", "loss_mask", "position3"):
        if k in x and not torch.equal(x[k], y[k]):
            print(f"     field {k} also differs", flush=True)


def main():
    cfg = load_config(str(Path(REPO) / "configs/qwen2_5_vl_3b_2k.yaml"))
    contract = json.load(open(RUN / "contract.json"))
    src = OnlineFeatureSource(cfg, contract, RUN, limit=256)
    ds = src.dataset()

    # pick a target row and a very short row by prepared length
    lens = {}
    for i in (0, 1, 20, 40, 60, 80, 100, 120):
        lens[i] = int(src.prepare(i)["inputs"]["input_ids"].numel())
    print("prepared lengths:", lens, flush=True)
    target = 0
    short = min(lens, key=lambda i: lens[i] if i != target else 10**9)
    long_ = max(lens, key=lambda i: lens[i])
    print(f"target={target} (len {lens.get(target)}) short={short} (len {lens[short]}) long={long_} (len {lens[long_]})", flush=True)

    only = ds.fetch_batch([target])[0]
    pair_short = ds.fetch_batch([target, short])[target if target < short else 0] if False else ds.fetch_batch([target, short])[0]
    pair_long = ds.fetch_batch([target, long_])[0]
    full = ds.fetch_batch(list(range(8)))[0]

    print("== target alone vs target+short ==", flush=True)
    diff_report("alone vs +short", only, pair_short)
    print("== target alone vs target+long ==", flush=True)
    diff_report("alone vs +long", only, pair_long)
    print("== target alone vs 8-group ==", flush=True)
    diff_report("alone vs g8", only, full)
    print("== target+short vs 8-group ==", flush=True)
    diff_report("+short vs g8", pair_short, full)
    src.close()


if __name__ == "__main__":
    main()
