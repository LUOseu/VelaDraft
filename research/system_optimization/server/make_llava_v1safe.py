"""Build /home/lyh/vlm-systems/data/llava_v1safe.json for the lane runs.

At HEAD e15256d the prepare stage rejects LLaVA-Instruct rows whose <image>
placeholder is not leading (vlm_dflash/data/schema.py check_v1_question_layout,
added by review-remediation 47b0075). The 2026-10-11 canonical run predates the
check, so the canonical recipe cannot be rebuilt from the unfiltered pool.

This script writes the subset of rows the check accepts (exactly one <image>
marker, leading, image name present) so prepare runs unchanged on current code.
The pool delta (78752/157712 rows, 49.9% dropped) is recorded in the lane report;
downstream slice/seed selection is untouched.
"""
import json

SRC = "/home/dhz/Data/llava_instruct_150k.json"
DST = "/home/lyh/vlm-systems/data/llava_v1safe.json"

with open(SRC, encoding="utf-8") as fh:
    raw = json.load(fh)

kept = []
dropped = {"short_conv": 0, "not_one_marker": 0, "text_before": 0, "no_image_name": 0}
for row in raw:
    conversations = row.get("conversations") or []
    if len(conversations) < 2:
        dropped["short_conv"] += 1
        continue
    question = str(conversations[0].get("value", ""))
    markers = question.count("<image>")
    if markers != 1:
        dropped["not_one_marker"] += 1
        continue
    if question.split("<image>", 1)[0].strip():
        dropped["text_before"] += 1
        continue
    if not isinstance(row.get("image"), str) or not row.get("image"):
        dropped["no_image_name"] += 1
        continue
    kept.append(row)

with open(DST, "w", encoding="utf-8") as fh:
    json.dump(kept, fh, ensure_ascii=False, separators=(",", ":"))

print(f"src={len(raw)} kept={len(kept)} dropped={dropped}")
