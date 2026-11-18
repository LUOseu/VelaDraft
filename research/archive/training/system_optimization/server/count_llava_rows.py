"""Count LLaVA-150k rows by v1 question layout (for the filtered-annotation decision)."""
import json

raw = json.load(open("/home/dhz/Data/llava_instruct_150k.json"))
n = ok = noimg = oneconv = multi = notstart = 0
examples = []
for row in raw:
    n += 1
    conv = row.get("conversations") or []
    if len(conv) < 2:
        oneconv += 1
        continue
    q = conv[0].get("value", "") or ""
    m = q.count("<image>")
    if m == 0:
        noimg += 1
    elif m > 1:
        multi += 1
        examples.append((str(row.get("id")), "multi", m))
    elif q.split("<image>", 1)[0].strip():
        notstart += 1
        if len(examples) < 5:
            examples.append((str(row.get("id")), "text-before", repr(q.split("<image>", 1)[0].strip()[:36])))
    else:
        ok += 1
print(f"rows={n} image_first={ok} no_image={noimg} single_conv={oneconv} multi_img={multi} text_before_img={notstart}")
print("examples:", examples[:5])
