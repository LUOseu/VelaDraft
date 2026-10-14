"""CPU check: train-only overrides must not change data/regen stage digests."""
import copy
import json
import sys

from vlm_dflash.cli import stage_digest
from vlm_dflash.config import load_config

cfg = load_config("configs/qwen2_5_vl_3b_2k.yaml")
contract = json.load(open("/home/lyh/vlm-dflash/runs/qwen25vl3b_2k/contract.json"))
d1 = stage_digest(cfg, contract, "data", None, limit=256)
cfg2 = copy.deepcopy(cfg)
cfg2["train"].update({
    "online_feature_cache_gb": 8,
    "online_batch_size": 8,
    "overfit_sanity": {"enabled": False},
})
d2 = stage_digest(cfg2, contract, "data", None, limit=256)
print("data digest master:", d1[:16])
print("data digest leg   :", d2[:16], "EQUAL" if d1 == d2 else "MISMATCH!!")
d3 = stage_digest(cfg, contract, "data", None, limit=None)   # the old S2 bug
print("data digest a no-limit leg:", d3[:16], "differs" if d3 != d1 else "same (fix would be unneeded!)")
r1 = stage_digest(cfg, contract, "regen", d1, limit=256)
r2 = stage_digest(cfg2, contract, "regen", d2, limit=256)
print("regen digest pair :", "EQUAL" if r1 == r2 else "MISMATCH!!")
ok = (d1 == d2) and (d3 != d1) and (r1 == r2)
print("CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
