"""Per-element ULP-normalised drift between two drafter checkpoints.

Usage: python ulp_drift.py A.pt B.pt [--json out.json]

For every float tensor: d = |a - b| in float64; the bf16 grid step at the
reference value a is ulp(a) = 2**(floor(log2|a|) - 7) for normal values
(bf16: 8 significand bits), flattened to the smallest normal otherwise.
Reports the distribution of ratio = d / ulp(a) (how many bf16 grid steps the
two runs sit apart), the relative error d/|a|, and per-tensor/per-category
shares so "rounding scale" can be stated precisely.

a is the reference (cache-OFF / canonical leg).
"""
import json
import math
import sys

import torch

BF16_MIN_NORMAL = 2.0 ** -126
BF16_SUBNORMAL_ULP = 2.0 ** -133  # 2**(1-7-127)


def model_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def ulp_bf16(x):
    ax = x.abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(ax)) - 7.0)
    return torch.where(ax >= BF16_MIN_NORMAL, ulp,
                       torch.full_like(x, BF16_SUBNORMAL_ULP))


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]
    a, b = model_state(pa), model_state(pb)
    keys = [k for k in a if k in b and torch.is_tensor(a[k]) and torch.is_tensor(b[k])
            and a[k].shape == b[k].shape and a[k].is_floating_point()]
    edges = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    buckets = [f"({lo},{hi}]" for lo, hi in zip([0] + edges, edges)] + [">16"]
    total = diff = 0
    gmax_ratio = 0.0
    gmax_ratio_t = None
    gmax_rel = 0.0
    gmax_rel_t = None
    gmax_abs = 0.0
    gmax_abs_t = None
    rows = []
    for k in keys:
        x = a[k].double()
        y = b[k].double()
        d = (x - y).abs()
        nd = int((d > 0).sum())
        ulp = ulp_bf16(x)
        ratio = d / ulp
        rmax = float(ratio.max())
        total += x.numel()
        diff += nd
        rel = float((d / x.abs().clamp_min(1e-30)).max())
        amax = float(d.max()) if d.numel() else 0.0
        if rmax > gmax_ratio:
            gmax_ratio, gmax_ratio_t = rmax, k
        if rel > gmax_rel:
            gmax_rel, gmax_rel_t = rel, k
        if amax > gmax_abs:
            gmax_abs, gmax_abs_t = amax, k
        rows.append((rmax, k, nd, amax, rel))
    # proper histogram over all tensors in one pass
    hist = [0] * (len(edges) + 1)
    for k in keys:
        x = a[k].double()
        y = b[k].double()
        d = (x - y).abs()
        ratio = d / ulp_bf16(x)
        nz = ratio > 0
        prev = 0
        for i, e in enumerate(edges):
            m = nz & (ratio > prev) & (ratio <= e)
            hist[i] += int(m.sum())
            prev = e
        hist[-1] += int((nz & (ratio > edges[-1])).sum())
    print(f"tensors={len(keys)} elements={total} differing={diff} ({100.0*diff/max(1,total):.3f}%)")
    print(f"max|delta|={gmax_abs:.6g} @ {gmax_abs_t}")
    print(f"max d/|a| ={gmax_rel:.6g} @ {gmax_rel_t}  (bf16 eps=2^-8={2**-8:.6g})")
    print(f"max d/ulp(a)={gmax_ratio:.6g} @ {gmax_ratio_t}")
    print("ratio=d/ulp buckets (share of all elements):")
    prev = 0
    for name, c, lo, hi in zip(buckets, hist, [0] + edges, edges + [math.inf]):
        print(f"  {lo:>4} < ratio <= {hi if hi != math.inf else 'inf':<4}: {c} ({100.0*c/max(1,total):.3f}%)")
    rows.sort(reverse=True)
    print("top-5 by max d/ulp:")
    for rmax, k, nd, amax, rel in rows[:5]:
        print(f"  {k}: max d/ulp={rmax:.4g} differing={nd} max|d|={amax:.6g} max rel={rel:.4g}")
    if out_json:
        with open(out_json, "w") as fh:
            json.dump({
                "a": pa, "b": pb, "tensors": len(keys), "elements": total,
                "differing": diff,
                "max_abs": gmax_abs, "max_abs_tensor": gmax_abs_t,
                "max_rel": gmax_rel, "max_rel_tensor": gmax_rel_t,
                "max_ulp_ratio": gmax_ratio, "max_ulp_ratio_tensor": gmax_ratio_t,
                "ulp_buckets": {name: c for name, c in zip(buckets, hist)},
                "per_tensor": [{"key": k, "max_ulp_ratio": rmax, "differing": nd,
                                "max_abs": amax, "max_rel": rel} for rmax, k, nd, amax, rel in rows],
            }, fh, indent=2)


if __name__ == "__main__":
    main()
