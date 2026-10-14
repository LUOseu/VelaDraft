"""Harvest the S3 canonical 2K A/B pair into report-ready evidence.

Run on the server from the lane root (/home/lyh/vlm-systems). Reads
runs/s3/{a_off,b_on}/train/{train_summary.json,train_log.jsonl},
logs/s3_ckpt_hashes.txt and logs/s3_{leg}.time; writes runs/s3/harvest.json
and prints a compact summary (walls, extraction counters, cache stats,
loss-sequence equality, grad_norm drift bound, RSS, ckpt hash verdict).
"""
import json
import re
from pathlib import Path

LANE = Path("/home/lyh/vlm-systems")
LEGS = {"a_off": 0, "b_on": 16}


def leg_summary(leg):
    d = json.load(open(LANE / f"runs/s3/{leg}/train/train_summary.json"))
    fs = d["feature_source"]
    return {
        "wall_seconds": d["wall_seconds"],
        "steps": d["steps"],
        "peak_gpu_gb": d["peak_gpu_mem_gb"],
        "extraction_wall_s": fs["extraction_wall_seconds_total"],
        "extraction_prepare_s": fs["extraction_prepare_seconds_total"],
        "extraction_forward_s": fs["extraction_forward_seconds_total"],
        "target_forwards": fs["target_forwards_total"],
        "target_samples": fs["target_samples_total"],
        "feature_cache": fs["feature_cache"],
        "digest": fs["digest"],
    }


def load_log(leg):
    return {int(json.loads(line)["step"]): json.loads(line)
            for line in open(LANE / f"runs/s3/{leg}/train/train_log.jsonl")}


def rss_gib(leg):
    text = (LANE / f"logs/s3_{leg}.time").read_text()
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    return round(int(match.group(1)) / 2**20, 3) if match else None


def main():
    out = {"legs": {}, "loss_equality": {}, "declared_cache_gib": LEGS}
    for leg in LEGS:
        out["legs"][leg] = leg_summary(leg)
        out["legs"][leg]["host_rss_peak_gib"] = rss_gib(leg)

    a, b = load_log("a_off"), load_log("b_on")
    steps = sorted(set(a) & set(b))
    loss_equal = all(a[s]["loss"] == b[s]["loss"] for s in steps)
    valid_equal = all(a[s]["valid_tokens"] == b[s]["valid_tokens"] for s in steps)
    grad_max = max(abs(float(a[s]["grad_norm"]) - float(b[s]["grad_norm"])) for s in steps)
    acc_max = max(abs(float(a[s]["accuracy"]) - float(b[s]["accuracy"])) for s in steps)
    out["loss_equality"] = {
        "steps_compared": len(steps),
        "loss_identical": loss_equal,
        "valid_tokens_identical": valid_equal,
        "grad_norm_max_abs_delta": grad_max,
        "accuracy_max_abs_delta": acc_max,
    }

    hashes = {}
    for line in open(LANE / "logs/s3_ckpt_hashes.txt"):
        h, p = line.split()
        hashes[p] = h
    finals = {p: h for p, h in hashes.items() if p.endswith("ckpt_final.pt")}
    out["ckpt_final_hashes"] = finals
    out["ckpt_final_match"] = len(set(finals.values())) == 1 and len(finals) == 2

    (LANE / "runs/s3/harvest.json").write_text(json.dumps(out, indent=2))

    print("== S3 harvest ==")
    for leg, s in out["legs"].items():
        cache = s["feature_cache"]
        print(f"{leg}: wall {s['wall_seconds']:.1f}s steps {s['steps']} "
              f"extract {s['extraction_wall_s']:.1f}s fwd {s['target_forwards']} "
              f"cache hits {cache.get('hits')} miss {cache.get('misses')} "
              f"evict {cache.get('evictions')} peakMB {int(cache.get('peak_bytes', 0)) // 2**20} "
              f"rss {s['host_rss_peak_gib']} GiB peakGPU {s['peak_gpu_gb']:.1f}")
    eq = out["loss_equality"]
    print(f"loss_equality: steps {eq['steps_compared']} loss_identical {eq['loss_identical']} "
          f"valid_identical {eq['valid_tokens_identical']} "
          f"grad_max_delta {eq['grad_norm_max_abs_delta']:.2e} "
          f"acc_max_delta {eq['accuracy_max_abs_delta']:.2e}")
    print(f"ckpt_final_match: {out['ckpt_final_match']}")
    for p, h in finals.items():
        print(f"  {h[:16]}  {p}")


if __name__ == "__main__":
    main()
