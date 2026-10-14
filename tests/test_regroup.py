"""D1 regroup integration tests: planner math, weighted-parity oracle, forward API.

Ports the D1 lane's ``oracle_regroup_parity.py`` + ``test_bucketing_pipeline.py``
(oracle math unchanged) onto the integrated opt-in API:

* planner / Z / weight helpers: :mod:`vlm_dflash.dflash.regroup` (D1 port);
* ``DFlashTrainerModule.forward(anchor_table=..., sample_weights=...)``;
* ``DFlashDecayLoss(weight_dtype="fp32")``.

The equivalence target is the canonical loop semantics: per-canonical-microbatch
decay-weighted MEAN, averaged over the accumulation window
(``L = (1/A) * sum_m S_m / Z_m``); the regrouped path replays one frozen anchor
table per logical batch with preserved per-sample weights
``g_i = 1 / (A * Z_m(i))`` applied to the block mask and a raw weighted-sum loss
(``normalize="tokens"``, ``num_tokens=None``).
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.decay_loss import DFlashDecayLoss
from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.dflash.regroup import (
    COST_ANCHOR_WEIGHT,
    SampleStat,
    canonical_microbatches,
    modeled_cost,
    plan_groups,
    row_weights,
    sample_z_rows,
    split_evenly,
)
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule

VOCAB, HID, LAYERS, HEADS, KVHEADS, HEADDIM = 96, 64, 2, 4, 2, 16
BLOCK, NUM_ANCHORS, GAMMA = 4, 8, 2.0
MASK_ID = 5
SGD_LR = 0.1


# --------------------------------------------------------------------------- #
# 1) planner unit tests (ported from the D1 pipeline test)
# --------------------------------------------------------------------------- #

def _mixed_stats():
    cells = [(900, 4, 2.0), (100, 1, 1.0), (500, 8, 4.0), (120, 2, 1.5), (880, 3, 2.5), (130, 6, 3.0)]
    return [SampleStat(key=f"s{i}", C=c, A=a, Z=z, index=i) for i, (c, a, z) in enumerate(cells)]


@pytest.mark.parametrize("mode,size", [("token", 2), ("token", 3), ("dual", 2), ("cost", 4),
                                       ("grid", 2), ("grid", 3), ("grid", 4), ("canonical", 2)])
def test_plan_invariants(mode, size):
    stats = _mixed_stats()
    groups = plan_groups(stats, mode, size)
    flat = sorted(i for group in groups for i in group)
    assert flat == list(range(len(stats)))
    assert all(1 <= len(group) <= size for group in groups)


def test_plan_token_partitions_by_context():
    stats = _mixed_stats()
    groups = plan_groups(stats, "token", sub_batch_size=3)
    lows, highs = groups[0], groups[1]
    assert max(stats[i].C for i in lows) <= min(stats[i].C for i in highs)


def test_plan_folds_zero_anchor_rows_into_last_group():
    stats = _mixed_stats()
    stats[1] = SampleStat(key="z", C=100, A=0, Z=0.0, index=1)
    groups = plan_groups(stats, "token", sub_batch_size=2)
    flat = sorted(i for group in groups for i in group)
    assert flat == list(range(len(stats)))
    assert 1 in groups[-1]


def test_split_evenly_and_modeled_cost():
    assert split_evenly([0, 1, 2, 3, 4], 2) == [[0, 1, 2], [3, 4]]
    stats = _mixed_stats()
    token_groups = plan_groups(stats, "token", 3)
    cost_groups = plan_groups(stats, "cost", 3)
    assert modeled_cost(stats, token_groups) < modeled_cost(stats, [list(range(6))])
    assert modeled_cost(stats, cost_groups) <= modeled_cost(stats, [list(range(6))])
    assert COST_ANCHOR_WEIGHT > 0


def test_row_weights_preserve_microbatch_means():
    canonical = [[0, 1], [2, 3], [4, 5]]
    z = [2.0, 1.0, 4.0, 1.5, 2.5, 0.0]
    weights = row_weights([str(i) for i in range(6)], canonical, z, accumulation=3)
    assert weights["0"] == pytest.approx(1.0 / (3 * (2.0 + 1.0)))
    assert weights["1"] == weights["0"]
    assert weights["4"] == pytest.approx(1.0 / (3 * (2.5 + 0.0)))
    assert weights["5"] == 0.0  # Z=0 samples drop out


def test_sample_z_rows_matches_decay_mask_sum():
    loss_mask = torch.ones(2, 8)
    anchors = torch.tensor([[0], [4]])
    keep = torch.ones(2, 1, dtype=torch.bool)
    # uniform decay: slots 1..3 per block -> 3 supervised positions per row
    z_uniform = sample_z_rows(loss_mask, anchors, keep, BLOCK, 1, None)
    assert z_uniform == pytest.approx([3.0, 3.0])
    # decay gamma=2.0: w_k = exp(-k/2), k = 0..2
    expected = float(torch.exp(-torch.arange(3, dtype=torch.float64) / 2.0).sum())
    z_decay = sample_z_rows(loss_mask, anchors, keep, BLOCK, 1, 2.0)
    assert z_decay == pytest.approx([expected, expected])
    # masked-out labels drop out of Z
    loss_mask[0, 2] = 0.0
    z_masked = sample_z_rows(loss_mask, anchors, keep, BLOCK, 1, None)
    assert z_masked[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 2) oracle fixture (tiny real module, mirrored from the D1 lane oracle)
# --------------------------------------------------------------------------- #

def _make_config():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    return Qwen3Config(
        hidden_size=HID, intermediate_size=128, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=KVHEADS, head_dim=HEADDIM,
        vocab_size=VOCAB, max_position_embeddings=512, rms_norm_eps=1e-6,
        attention_bias=False, attention_dropout=0.0, rope_theta=10000.0,
        num_target_layers=4, block_size=BLOCK,
        dflash_config={"mask_token_id": MASK_ID, "target_layer_ids": [0],
                       "spatial_rope_enabled": False, "visual_num_query_tokens": 0},
    )


def _build_module(seed=7, dtype=torch.float64, loss_type="dflash"):
    torch.manual_seed(seed)
    draft = Qwen3DFlashDraftModel(_make_config())
    embed = nn.Embedding(VOCAB, HID, dtype=dtype)
    lm_head = nn.Linear(HID, VOCAB, bias=False, dtype=dtype)
    with torch.no_grad():
        nn.init.normal_(embed.weight, std=0.05)
        nn.init.normal_(lm_head.weight, std=0.05)
    module = DFlashTrainerModule(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=MASK_ID, block_size=BLOCK, attention_backend="sdpa",
        num_anchors=NUM_ANCHORS, loss_decay_gamma=GAMMA, loss_type=loss_type,
    )
    if dtype == torch.float64:
        module.double()
    else:
        module.float()
    module.embed_tokens.to(dtype)
    module.lm_head.to(dtype)
    return module


def _key_of(row_ids: torch.Tensor) -> str:
    ids = row_ids.detach().to(torch.int64).cpu().flatten()
    nz = (ids != 0).nonzero(as_tuple=False)
    if nz.numel():
        ids = ids[: nz[-1, 0] + 1]
    return hashlib.sha1(ids.numpy().tobytes()).hexdigest()


def _make_rows():
    torch.manual_seed(3)
    spans = [5, 4, 6, 3, 5, 6]
    lengths = [9, 11, 8, 10, 12, 9]
    rows = []
    for c, ans in zip(lengths, spans):
        ids = torch.randint(10, VOCAB, (c,), dtype=torch.long)
        lm = torch.zeros(c, dtype=torch.float32)
        lm[c - ans:] = 1.0
        hidden = torch.randn(c, HID, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(100 + c * 31 + ans))
        rows.append({"ids": ids, "lm": lm, "C": c, "hidden": hidden})
    return rows


def _collate(rows, dtype, min_len=None):
    s = max(r["C"] for r in rows)
    if min_len is not None:
        s = max(s, min_len)
    ids = torch.zeros(len(rows), s, dtype=torch.long)
    lm = torch.zeros(len(rows), s, dtype=torch.float32)
    hidden = torch.zeros(len(rows), s, HID, dtype=dtype)
    for i, r in enumerate(rows):
        ids[i, : r["C"]] = r["ids"]
        lm[i, : r["C"]] = r["lm"]
        hidden[i, : r["C"]] = r["hidden"].to(dtype)
    return ids, lm.to(dtype), hidden


def _sample_anchor_table(rows, canonical_split, seed=1000):
    class Shim:
        block_size = BLOCK
        num_anchors = NUM_ANCHORS

    table = {}
    for mbi, members in enumerate(canonical_split):
        s = max(rows[i]["C"] for i in members)
        lm = torch.zeros(len(members), s)
        for j, i in enumerate(members):
            lm[j, : rows[i]["C"]] = rows[i]["lm"]
        torch.manual_seed(seed + mbi)
        anchors, keep = DFlashTrainerModule._sample_anchor_positions(Shim(), s, lm, torch.device("cpu"))
        for j, i in enumerate(members):
            cnt = int(keep[j].sum().item())
            table[_key_of(rows[i]["ids"])] = anchors[j, :cnt].tolist()
    return table


def _replay_anchors(table, batch_ids, num_anchors):
    counts = [len(table[_key_of(b)]) for b in batch_ids]
    max_n = min(num_anchors, max(counts))
    anchors = torch.zeros(len(counts), max_n, dtype=torch.long)
    keep = torch.zeros(len(counts), max_n, dtype=torch.bool)
    for i, b in enumerate(batch_ids):
        pos = table[_key_of(b)]
        anchors[i, : len(pos)] = torch.tensor(pos, dtype=torch.long)
        keep[i, : len(pos)] = True
    return anchors, keep


def _run_arm(base, groups, rows, dtype, table, *, weighted, weights=None, accumulation=1):
    """One A/B arm via the integrated API (anchor_table + optional sample_weights)."""
    module = copy.deepcopy(base)
    if weighted:
        module.loss_fn = DFlashDecayLoss(
            loss_gamma=module.loss_decay_gamma, normalize="tokens",
            first_slot=module.first_supervised_slot,
        )
    module.zero_grad(set_to_none=True)
    total = 0.0
    zs = []
    for members in groups:
        batch_ids = [rows[i]["ids"] for i in members]
        min_len = max(max(table[_key_of(b)]) + BLOCK for b in batch_ids if table[_key_of(b)])
        ids, lm, hidden = _collate([rows[i] for i in members], dtype, min_len=min_len)
        anchors, keep = _replay_anchors(table, batch_ids, NUM_ANCHORS)
        row_w = None
        if weighted:
            row_w = torch.tensor([weights[_key_of(rows[i]["ids"])] for i in members], dtype=torch.float32)
        metrics = module(input_ids=ids, hidden_states=hidden, loss_mask=lm,
                         anchor_table=(anchors, keep), sample_weights=row_w)
        zs.append(float(metrics.loss_weight))
        if weighted:
            total += float(metrics.loss.detach())
            metrics.loss.backward()
        else:
            total += float(metrics.loss.detach()) / accumulation
            (metrics.loss / accumulation).backward()
    grads = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}
    upd = {n: (p.detach() - SGD_LR * p.grad).clone()
           for n, p in module.named_parameters() if p.grad is not None}
    return total, grads, upd, zs


def _analytic_z(rows, table, canonical):
    """Per-row Z via the planner helper, bounded by the row's canonical microbatch pad."""
    zs = []
    for i, r in enumerate(rows):
        mb = next(m for m in canonical if i in m)
        s = max(rows[j]["C"] for j in mb)
        lm = torch.zeros(1, s)
        lm[0, : r["C"]] = r["lm"]
        pos = table[_key_of(r["ids"])]
        anchors = torch.zeros(1, len(pos), dtype=torch.long)
        keep = torch.zeros(1, len(pos), dtype=torch.bool)
        if pos:
            anchors[0] = torch.tensor(pos)
            keep[0] = True
        zs.append(sample_z_rows(lm, anchors, keep, BLOCK, 1, GAMMA)[0])
    return zs


def _assert_arms_match(loss_ref, grads_ref, upd_ref, loss_c, grads_c, upd_c, *, dloss_tol, grel_tol):
    dloss = abs(loss_c - loss_ref)
    assert dloss < dloss_tol, dloss
    for name in grads_ref:
        rel = ((grads_c[name] - grads_ref[name]).abs().max() /
               (grads_ref[name].abs().max() + 1e-30)).item()
        assert rel < grel_tol, (name, rel)
    umax = max((upd_c[n] - upd_ref[n]).abs().max().item() for n in upd_ref)
    assert umax < max(1e-8, grel_tol)


@pytest.mark.parametrize("dtype,name,dloss_tol,grel_tol", [
    (torch.float64, "float64", 1e-5, 1e-5),
    (torch.float32, "float32", 1e-4, 1e-3),
])
def test_weighted_regroup_matches_canonical(dtype, name, dloss_tol, grel_tol):
    torch.set_num_threads(2)
    rows = _make_rows()
    canonical = [[0, 1], [2, 3], [4, 5]]
    a = len(canonical)
    table = _sample_anchor_table(rows, canonical)
    base = _build_module(dtype=dtype)

    loss_ref, grads_ref, upd_ref, zs_ref = _run_arm(
        base, canonical, rows, dtype, table, weighted=False, accumulation=a)
    # Z bookkeeping: the planner's analytic per-row Z must reproduce the module's
    # own decay-weighted mask sums (loss_weight) per canonical microbatch.
    z_rows = _analytic_z(rows, table, canonical)
    for mbi, members in enumerate(canonical):
        assert sum(z_rows[i] for i in members) == pytest.approx(zs_ref[mbi], rel=1e-6, abs=1e-9)

    weights = {_key_of(rows[i]["ids"]): 1.0 / (a * sum(z_rows[j] for j in next(m for m in canonical if i in m)))
               for i in range(len(rows))}

    cases = {
        "same-size [0,3][1,2][4,5]": [[0, 3], [1, 2], [4, 5]],
        "uneven [0,2,4][1][3,5]": [[0, 2, 4], [1], [3, 5]],
        "2 batches [0,1,2,3][4,5]": [[0, 1, 2, 3], [4, 5]],
    }
    for cname, groups in cases.items():
        loss_c, grads_c, upd_c, _ = _run_arm(
            base, groups, rows, dtype, table, weighted=True, weights=weights)
        _assert_arms_match(loss_ref, grads_ref, upd_ref, loss_c, grads_c, upd_c,
                           dloss_tol=dloss_tol, grel_tol=grel_tol)

    # planner-driven modes end-to-end (planner -> weights -> forward)
    stats = [SampleStat(key=_key_of(rows[i]["ids"]), C=rows[i]["C"],
                        A=len(table[_key_of(rows[i]["ids"])]), Z=z_rows[i], index=i)
             for i in range(len(rows))]
    # size=1 = all-singleton groups (= the R4 `--sub-batch 1` boundary config;
    # added per D1's cross-check of 4e1674e, suggestion (a)).
    for mode, size in (("token", 2), ("dual", 2), ("cost", 4), ("grid", 2),
                       ("token", 1), ("cost", 1), ("grid", 1)):
        groups = plan_groups(stats, mode, size)
        loss_c, grads_c, upd_c, _ = _run_arm(
            base, groups, rows, dtype, table, weighted=True, weights=weights)
        _assert_arms_match(loss_ref, grads_ref, upd_ref, loss_c, grads_c, upd_c,
                           dloss_tol=dloss_tol, grel_tol=grel_tol)


# --------------------------------------------------------------------------- #
# 3) forward-API guarantees
# --------------------------------------------------------------------------- #

def _batch(rows=None, dtype=torch.float32):
    rows = rows if rows is not None else _make_rows()
    return _collate(rows, dtype)


def test_anchor_table_replay_is_bitwise_equal_to_sampling():
    torch.set_num_threads(2)
    ids, lm, hidden = _batch()
    s = ids.shape[1]
    mod_a = _build_module(dtype=torch.float32)
    mod_b = _build_module(dtype=torch.float32)
    torch.manual_seed(123)
    metrics_a = mod_a(input_ids=ids, hidden_states=hidden, loss_mask=lm)
    metrics_a.loss.backward()
    torch.manual_seed(123)
    table = mod_b._sample_anchor_positions(s, lm, torch.device("cpu"))
    metrics_b = mod_b(input_ids=ids, hidden_states=hidden, loss_mask=lm, anchor_table=table)
    assert float(metrics_a.loss.detach()) == float(metrics_b.loss.detach())
    metrics_b.loss.backward()
    for name, p in mod_a.named_parameters():
        g_a = p.grad
        g_b = dict(mod_b.named_parameters())[name].grad
        assert (g_a is None) == (g_b is None)
        if g_a is not None:
            assert torch.equal(g_a, g_b), name


def test_sample_weights_noop_and_scaling():
    torch.set_num_threads(2)
    ids, lm, hidden = _batch()
    bsz = ids.shape[0]
    module = _build_module(dtype=torch.float32)
    module.loss_fn = DFlashDecayLoss(loss_gamma=GAMMA, normalize="tokens", first_slot=1)
    torch.manual_seed(9)
    base_loss = module(input_ids=ids, hidden_states=hidden, loss_mask=lm).loss.detach()
    torch.manual_seed(9)
    ones = module(input_ids=ids, hidden_states=hidden, loss_mask=lm,
                  sample_weights=torch.ones(bsz)).loss.detach()
    # weights=1.0 multiply is a bitwise no-op on the mask
    assert torch.equal(base_loss, ones)
    torch.manual_seed(9)
    doubled = module(input_ids=ids, hidden_states=hidden, loss_mask=lm,
                     sample_weights=torch.full((bsz,), 2.0)).loss.detach()
    # raw weighted sum scales exactly with a power-of-two weight
    assert float(doubled) == 2.0 * float(base_loss)


def test_weight_dtype_fp32_precise_reduction():
    torch.manual_seed(3)
    nll = torch.rand(4, 6, dtype=torch.bfloat16) * 8
    mask = (torch.rand(4, 6) > 0.25).float() * 0.375
    auto = DFlashDecayLoss(loss_gamma=2.0, normalize="tokens", first_slot=1)
    precise = DFlashDecayLoss(loss_gamma=2.0, normalize="tokens", first_slot=1, weight_dtype="fp32")
    out_auto = auto._reduce(nll, mask, None, None)
    out_precise = precise._reduce(nll, mask, None, None)
    assert out_auto.total_loss.dtype == torch.bfloat16
    assert out_precise.total_loss.dtype == torch.float32
    w = torch.exp(-torch.arange(6, dtype=torch.float64) / 2.0)
    manual = ((w.unsqueeze(0) * mask.double()) * nll.double()).sum()
    assert float(out_precise.total_loss) == pytest.approx(float(manual), rel=1e-6)
    with pytest.raises(ValueError):
        DFlashDecayLoss(weight_dtype="float64")


def test_forward_api_guards():
    torch.set_num_threads(2)
    ids, lm, hidden = _batch()
    bsz, s = ids.shape
    module = _build_module(dtype=torch.float32)
    ok_keep = torch.ones(bsz, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="seq_len - block_size"):
        module(input_ids=ids, hidden_states=hidden, loss_mask=lm,
               anchor_table=(torch.full((bsz, 1), s - 1, dtype=torch.long), ok_keep))
    with pytest.raises(ValueError, match="sample_weights"):
        module(input_ids=ids, hidden_states=hidden, loss_mask=lm,
               sample_weights=torch.ones(bsz + 1))
    with pytest.raises(ValueError, match="packed inputs"):
        module(input_ids=ids, hidden_states=hidden, loss_mask=lm,
               seq_lens=torch.ones(bsz, 1, dtype=torch.long),
               doc_remaining=torch.ones(bsz, s, dtype=torch.long),
               anchor_table=(torch.zeros(bsz, 1, dtype=torch.long), ok_keep))
    vp = _build_module(dtype=torch.float32, loss_type="variable_prefix")
    with pytest.raises(ValueError, match="variable_prefix"):
        vp(input_ids=ids, hidden_states=hidden, loss_mask=lm,
           anchor_table=(torch.zeros(bsz, 1, dtype=torch.long), ok_keep))


def test_default_loss_settings_unchanged():
    module = _build_module(dtype=torch.float32)
    assert module.loss_fn.normalize == "mean"
    assert module.loss_fn.weight_dtype == "auto"
