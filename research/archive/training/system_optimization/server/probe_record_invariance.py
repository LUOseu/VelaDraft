"""Probe: are online extraction records invariant to batch grouping/order?

Extracts the same rows in several groupings via the real OnlineFeatureDataset
path (master real256 slice) and compares every tensor field bitwise:
  g1      rows[0..7] in order, one batched forward
  g1'     same call repeated (determinism control)
  single  each of rows[0..3] fetched alone (group of 1)
  rev     rows[7..0] as a group (order sensitivity)
  mixed   rows[0..3]+rows[120..123] (composition/padding sensitivity)

If records are grouping-invariant, every comparison is bitwise identical and
the OFF/ON divergence must come from elsewhere; if not, the cache pins the
epoch-1 grouping's numeric image of each row.
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


def tensors(record):
    return {k: v for k, v in record.items() if torch.is_tensor(v)}


def compare(tag, x, y):
    same = True
    for key in tensors(x):
        a, b = x[key], y[key]
        if a.shape != b.shape or not torch.equal(a, b):
            same = False
            if a.shape == b.shape:
                delta = (a.float() - b.float()).abs().max().item()
            else:
                delta = float("nan")
            print(f"  {tag} [{key}]: DIFF max_abs {delta:.3e} shapes {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
    if same:
        print(f"  {tag}: bitwise identical ({len(tensors(x))} fields)", flush=True)
    return same


def main():
    cfg = load_config(str(Path(REPO) / "configs/qwen2_5_vl_3b_2k.yaml"))
    contract = json.load(open(RUN / "contract.json"))
    src = OnlineFeatureSource(cfg, contract, RUN, limit=256)
    dataset = src.dataset()
    print("rows:", len(src.rows), "online_batch_size:", src.online_batch_size, flush=True)
    g1 = dataset.fetch_batch(list(range(8)))
    g1b = dataset.fetch_batch(list(range(8)))
    singles = [dataset.fetch_batch([i])[0] for i in range(4)]
    rev = dataset.fetch_batch(list(reversed(range(8))))
    mixed = dataset.fetch_batch([0, 1, 2, 3, 120, 121, 122, 123])
    print("== g1 vs repeat ==", flush=True)
    for i in range(8):
        compare(f"row{i} g1-vs-g1'", g1[i], g1b[i])
    print("== g1 vs single ==", flush=True)
    for i in range(4):
        compare(f"row{i} g1-vs-single", g1[i], singles[i])
    print("== g1 vs reversed order ==", flush=True)
    for i in range(8):
        compare(f"row{i} g1-vs-rev", g1[i], rev[7 - i])
    print("== g1 vs mixed composition ==", flush=True)
    for i in range(4):
        compare(f"row{i} g1-vs-mixed", g1[i], mixed[i])
    src.close()


if __name__ == "__main__":
    main()
