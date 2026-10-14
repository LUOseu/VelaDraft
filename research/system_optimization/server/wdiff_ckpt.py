"""Diff two drafter checkpoints (ckpt_final.pt) tensor by tensor.

Usage: python wdiff_ckpt.py A.pt B.pt
Prints tensor count, total elements, differing elements, max |delta| and the
five tensors with the most differing elements.
"""
import sys

import torch


def model_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def main():
    a = model_state(sys.argv[1])
    b = model_state(sys.argv[2])
    keys = [key for key in a if key in b]
    total = diff = 0
    max_abs = 0.0
    worst = None
    rows = []
    for key in keys:
        x, y = a[key], b[key]
        if not torch.is_tensor(x) or not torch.is_tensor(y) or x.shape != y.shape:
            print(f"  skip {key}: shape/type mismatch")
            continue
        d = (x.float() - y.float()).abs()
        nd = int((d > 0).sum())
        mabs = float(d.max()) if d.numel() else 0.0
        total += x.numel()
        diff += nd
        if mabs > max_abs:
            max_abs, worst = mabs, key
        if nd:
            rows.append((nd, key, mabs))
    rows.sort(reverse=True)
    print(f"tensors={len(keys)} elements={total} differing={diff} ({100.0 * diff / max(1, total):.3f}%)")
    print(f"max|delta|={max_abs:.6g} worst_tensor={worst}")
    for nd, key, mabs in rows[:5]:
        print(f"  {key}: {nd} elems, max|delta|={mabs:.6g}")
    missing = [key for key in a if key not in b]
    if missing:
        print("missing in b:", missing[:5])


if __name__ == "__main__":
    main()
