import json, sys, statistics
from transformers import AutoTokenizer

BASE_TOK = "/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE_TOK, use_fast=False)

def load(fp):
    rows = []
    for line in open(fp):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

base_fp, spec_fp = sys.argv[1], sys.argv[2]
b = load(base_fp); s = load(spec_fp)
bs = {r["question_id"]: r for r in b}
ss = {r["question_id"]: r for r in s}
common = [q for q in bs if q in ss]
print("questions baseline/spec/common:", len(b), len(s), len(common))

def base_counts(row):
    c = row["choices"][0]
    n = 0
    for t in c["turns"]:
        n += max(len(tok(t).input_ids) - 1, 0)
    return n, sum(c["wall_time"]), c["turns"][0]

b_tps = []; b_wall = 0.0; b_tok = 0; b_out = {}
for q in common:
    n, t, text = base_counts(bs[q])
    b_tps.append(n / t if t > 0 else 0.0)
    b_wall += t; b_tok += n; b_out[q] = text

s_tps = []; s_wall = 0.0; s_tok = 0; s_out = {}
taus = []
for q in common:
    c = ss[q]["choices"][0]
    n = sum(c["new_tokens"]); t = sum(c["wall_time"])
    s_tps.append(n / t if t > 0 else 0.0)
    s_wall += t; s_tok += n; s_out[q] = c["turns"][0]
    taus += c["acceptance_length"]

print("baseline: %d retok tokens, %d wall-time total, %.1f s/ q" % (b_tok, round(b_wall), b_wall/len(common)))
print("spec    : %d new tokens, %d wall-time total, %.1f s/ q" % (s_tok, round(s_wall), s_wall/len(common)))
print("baseline mean per-q tok/s: %.2f" % statistics.mean(b_tps))
print("spec     mean per-q tok/s: %.2f" % statistics.mean(s_tps))
print("official-style speedup (mean tok/s ratio): %.3f" % (statistics.mean(s_tps)/statistics.mean(b_tps)))
print("pooled tok/s ratio: %.3f" % ((s_tok/s_wall)/(b_tok/b_wall)))
wr = []
for q in common:
    tb = sum(bs[q]["choices"][0]["wall_time"]); ts = sum(ss[q]["choices"][0]["wall_time"])
    wr.append(tb / ts if ts > 0 else 0.0)
print("paired wall ratio mean %.3f median %.3f" % (statistics.mean(wr), statistics.median(wr)))
print("tau pooled mean over %d rounds: %.4f" % (len(taus), statistics.mean(taus)))
eq = sum(1 for q in common if b_out[q].strip() == s_out[q].strip())
print("exact-output-match prompts: %d/%d" % (eq, len(common)))
