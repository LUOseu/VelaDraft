r"""Training objectives and recipes for the DFlash-family block drafters.

The default objective is ``controlled_ce``: the decay-weighted block CE that
every pre-recipe run in this repo used. ``upstream_two_stage`` ports the
upstream AngelSpec DFly training recipe (Tencent/AngelSpec @ d3412bed — the
loss ops are the MIT-licensed LightSeek Foundation code, see
``docs/upstream/dfly.md``):

* stage 1 cold start: the LK acceptance-rate distillation loss with D-PACE
  position weighting and **no CE**;
* stage 2: D-PACE-weighted block CE **plus** the end-to-end multi-step TV term.

Formulas (p = teacher, q = student, per supervised slot; ``sg`` = stop-grad):

.. math::
    \mathrm{TV} = \tfrac12 \sum_i |p_i - q_i|, \qquad
    \mathrm{KL} = \sum_i p_i (\log p_i - \log q_i)

    \mathrm{LK_{hybrid}} = \lambda\,\mathrm{KL} + (1-\lambda)\,\mathrm{TV},
    \quad \lambda = \exp(-\eta \cdot sg(1 - \mathrm{TV}))

    \mathrm{LK_{alpha}} = -\log \sum_i \min(p_i, q_i)

    w_j^{\mathrm{dpace}} = \sum_{t \ge j} \prod_{s \le t}
        \bigl((1-\alpha)\,c_s + \alpha\bigr), \quad c_s = sg\,e^{-\mathrm{CE}_s}

    \alpha_j = \sum_i \min(p_{j,i}, q_{j,i}), \qquad
    L_{\mathrm{e2e}} = 1 - \frac{1}{\Gamma}\sum_{j=1}^{\Gamma}
        \prod_{i \le j} \alpha_i

The recipe is selected with ``algorithm.train_recipe`` (a name or a mapping
``{name, cold_start_fraction}``). It is supported by the algorithms that share
the block-CE trainer core (``dflash``, ``dflare``, ``dfly``); the other
algorithms reject it instead of silently ignoring it. Both upstream stages
consume the target's final hidden states as the teacher signal, so a recipe run
requires ``capture.store_final_hidden=true`` and a feature store with
``stores_final_hidden=true``.

Everything here is intentionally pure math + config resolution; the trainer
module (``vlm_dflash.dflash.trainer_core``) projects the teacher logits lazily
through :class:`TeacherLogitsStream` and applies the loss. All loss arithmetic
runs in fp32 in bounded chunks, so neither the full ``[B, n, S, V]`` teacher
tensor nor the fp32 scores materialize at once (that teacher tensor alone is
8.7 GB at a 16-sample batch of the canonical geometry), and every chunk is
activation-checkpointed so the fp32 probability slabs are recomputed in the
backward instead of being retained across chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "RECIPE_ALGORITHMS",
    "RECIPE_SCHEME_VERSION",
    "RecipeStage",
    "ResolvedRecipe",
    "TeacherLogitsStream",
    "dpace_position_weights",
    "e2e_tv_term",
    "gather_teacher_logits",
    "lk_per_position",
    "lk_term",
    "per_slot_nll",
    "plan_recipe_knobs",
    "recipe_loss",
    "resolve_recipe",
]

#: Version tag for the recipe block inside the training-plan digest.
RECIPE_SCHEME_VERSION = "vlm-dflash-recipe-v1"

#: Algorithms whose trainer module is the shared block-CE core and therefore
#: accepts ``algorithm.train_recipe``. Other algorithms reject the key.
RECIPE_ALGORITHMS = ("dflash", "dflare", "dfly")

# Chunk sizes for the fp32 loss arithmetic (rows of positions / blocks per
# chunk). Bounds the transient fp32 memory at full vocab (~152k) without
# changing any math.
LK_CHUNK_ROWS = 1024
E2E_CHUNK_BLOCKS = 128


@dataclass(frozen=True)
class RecipeStage:
    """One resolved training stage: loss coefficients + position weighting."""

    name: str
    weighting: str  # "decay" | "dpace"
    ce_coef: float
    lk_coef: float
    e2e_tv_coef: float
    lk_loss_type: str = "hybrid"
    lk_eta: float = 3.0
    dpace_alpha: float = 0.5

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "weighting": self.weighting,
            "ce_coef": float(self.ce_coef),
            "lk_coef": float(self.lk_coef),
            "e2e_tv_coef": float(self.e2e_tv_coef),
            "lk_loss_type": self.lk_loss_type,
            "lk_eta": float(self.lk_eta),
            "dpace_alpha": float(self.dpace_alpha),
        }


@dataclass(frozen=True)
class ResolvedRecipe:
    """A resolved recipe: stages plus the stage boundary (as a fraction)."""

    name: str
    stages: tuple[RecipeStage, ...]
    cold_start_fraction: float
    needs_teacher: bool

    def split_step(self, total_steps: int) -> int:
        """Step at which stage 2 starts (0 = skip stage 1, total = skip stage 2)."""
        total = max(1, int(total_steps))
        return min(total, max(0, int(round(self.cold_start_fraction * total))))

    def stage_index(self, step: int, total_steps: int) -> int:
        if len(self.stages) == 1:
            return 0
        return 0 if int(step) < self.split_step(total_steps) else 1

    def stage_of(self, step: int, total_steps: int) -> RecipeStage:
        return self.stages[self.stage_index(step, total_steps)]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "scheme": RECIPE_SCHEME_VERSION,
            "cold_start_fraction": float(self.cold_start_fraction),
            "needs_teacher": bool(self.needs_teacher),
            "stages": [stage.as_dict() for stage in self.stages],
        }


def _controlled_ce() -> ResolvedRecipe:
    return ResolvedRecipe(
        name="controlled_ce",
        stages=(
            RecipeStage(
                name="decayed_ce", weighting="decay", ce_coef=1.0, lk_coef=0.0, e2e_tv_coef=0.0
            ),
        ),
        cold_start_fraction=1.0,
        needs_teacher=False,
    )


def _upstream_two_stage(cold_start_fraction: float) -> ResolvedRecipe:
    if not 0.0 <= cold_start_fraction <= 1.0:
        raise ValueError(
            "algorithm.train_recipe.cold_start_fraction must be in [0, 1], got "
            f"{cold_start_fraction!r}."
        )
    return ResolvedRecipe(
        name="upstream_two_stage",
        stages=(
            RecipeStage(
                name="cold_start_lk_dpace",
                weighting="dpace",
                ce_coef=0.0,
                lk_coef=1.0,
                e2e_tv_coef=0.0,
                lk_loss_type="hybrid",
                lk_eta=3.0,
                dpace_alpha=0.5,
            ),
            RecipeStage(
                name="e2e_tv",
                weighting="dpace",
                ce_coef=1.0,
                lk_coef=0.0,
                e2e_tv_coef=1.0,
                dpace_alpha=0.5,
            ),
        ),
        cold_start_fraction=float(cold_start_fraction),
        needs_teacher=True,
    )


_RECIPE_BUILDERS = {
    "controlled_ce": lambda options: _controlled_ce(),
    "upstream_two_stage": lambda options: _upstream_two_stage(
        float(options.get("cold_start_fraction", 0.5))
    ),
}

#: Option keys a recipe mapping may carry besides ``name``.
_RECIPE_OPTIONS = ("cold_start_fraction",)


def resolve_recipe(cfg: dict, algorithm_name: str) -> ResolvedRecipe | None:
    """Resolve ``algorithm.train_recipe`` for ``algorithm_name`` (or ``None``).

    Bounded schema: unknown recipe names, unknown option keys and recipes on
    algorithms outside :data:`RECIPE_ALGORITHMS` all fail loud, so a config can
    never ask for an objective the run would silently not train.
    """
    algorithm_cfg = cfg.get("algorithm") if isinstance(cfg, dict) else None
    if not isinstance(algorithm_cfg, dict):
        return None
    block = algorithm_cfg.get("train_recipe")
    if block is None:
        return None
    if algorithm_name not in RECIPE_ALGORITHMS:
        raise ValueError(
            f"algorithm.train_recipe is not supported for algorithm={algorithm_name!r}. "
            f"Supported algorithms: {list(RECIPE_ALGORITHMS)} — their trainers share the block-CE "
            "core the recipe objectives attach to; other algorithms define their own objectives."
        )
    if isinstance(block, str):
        name, options = block, {}
    elif isinstance(block, dict):
        options = {key: value for key, value in block.items() if key != "name"}
        name = block.get("name")
    else:
        raise ValueError(
            "algorithm.train_recipe must be a recipe name or a mapping {name, cold_start_fraction}; "
            f"got {type(block).__name__}."
        )
    if not isinstance(name, str) or name not in _RECIPE_BUILDERS:
        raise ValueError(
            f"algorithm.train_recipe has unknown recipe name {name!r}; supported: "
            f"{sorted(_RECIPE_BUILDERS)}."
        )
    unknown = sorted(set(options) - set(_RECIPE_OPTIONS))
    if unknown:
        raise ValueError(
            f"algorithm.train_recipe for {name!r} has unsupported keys {unknown}; supported: "
            f"{sorted(_RECIPE_OPTIONS)}."
        )
    return _RECIPE_BUILDERS[name](options)


def plan_recipe_knobs(cfg: dict, algorithm_name: str) -> dict:
    """Plan-digest fragment for ``algorithm.train_recipe`` ({} when unset).

    The fragment is only emitted when the config selects a recipe, so plans of
    runs that never mention the key keep their pre-recipe digests byte-identical
    (their recorded checkpoints stay exactly resumable).
    """
    recipe = resolve_recipe(cfg, algorithm_name)
    if recipe is None:
        return {}
    return {"train_recipe": recipe.as_dict()}


def assert_recipe_capture(
    recipe: ResolvedRecipe | None, cfg: dict, *, algorithm_name: str
) -> None:
    """Fail loud when a recipe needs the teacher signal but capture is off."""
    if recipe is None or not recipe.needs_teacher:
        return
    capture = cfg.get("capture") or {}
    if not bool(capture.get("store_final_hidden", False)):
        raise ValueError(
            f"algorithm={algorithm_name} train_recipe={recipe.name!r} distills the target "
            "distribution, which requires capture.store_final_hidden=true so the target's final "
            "hidden states (the teacher signal) reach the trainer."
        )


# --------------------------------------------------------------------------- #
# loss math (exact ports of the upstream formulas)
# --------------------------------------------------------------------------- #


def dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Detached D-PACE continuation-value weights from per-position confidences.

    Exact port of ``angelspec.models.dflash._dpace_position_weights``: with
    ``s_j = (1-alpha) c_j + alpha`` the weight of position *j* is the suffix sum
    of the prefix products, ``w_j = Σ_{t ≥ j} Π_{s ≤ t} s_s`` — high weights
    early in the block, decaying with depth.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dpace_alpha must be in [0, 1], got {alpha!r}")
    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1), dims=[-1]
        )
    return weights.to(dtype=confidences.dtype)


def _tv_kl(student_chunk: torch.Tensor, teacher_chunk: torch.Tensor):
    """fp32 ``(tv, kl)`` for one chunk of positions (``[rows, V]``)."""
    t = teacher_chunk.float()
    s = student_chunk.float()
    s_logp = F.log_softmax(s, dim=-1)
    q = s_logp.exp()
    p = F.softmax(t, dim=-1)
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    eps = 1e-9
    kl = (p * (p.clamp_min(eps).log() - s_logp)).sum(dim=-1)
    return tv, kl


def lk_per_position(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    loss_type: str,
    eta: float,
) -> torch.Tensor:
    """Per-position LK loss ``[rows]`` (chunked; fp32 arithmetic)."""
    rows = student_logits.shape[0]
    out = []
    for start in range(0, rows, LK_CHUNK_ROWS):
        stop = min(rows, start + LK_CHUNK_ROWS)
        tv, kl = _tv_kl(student_logits[start:stop], teacher_logits[start:stop])
        if loss_type == "alpha":
            ell = -torch.log((1.0 - tv).clamp_min(1e-10))
        elif loss_type == "hybrid":
            lam = torch.exp(-eta * (1.0 - tv).detach().clamp(0.0, 1.0))
            ell = lam * kl + (1.0 - lam) * tv
        else:
            raise ValueError(f"unknown lk_loss_type {loss_type!r}; expected 'alpha' or 'hybrid'.")
        out.append(ell)
    return torch.cat(out, dim=0) if len(out) > 1 else out[0]


def lk_term(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    *,
    loss_type: str,
    eta: float,
    teacher_chunk_fn=None,
) -> torch.Tensor:
    """Weighted-mean LK term over the flat supervised slots.

    The teacher distribution arrives either as one full ``[rows, V]`` tensor
    or lazily through ``teacher_chunk_fn(start, stop) -> [rows, V]`` (the
    trainer's :class:`TeacherLogitsStream`), so the full teacher tensor never
    has to exist at once. Chunk boundaries are an implementation detail: the
    result is identical either way.

    Every chunk runs under activation checkpointing (when grad is enabled), so
    the fp32 softmax/log-softmax activations of a chunk are recomputed in the
    backward instead of being retained: without it the autocast fp32 copies of
    the full-vocab rows accumulate across all chunks (~35 GB at the canonical
    16-sample sanity batch) and OOM a 48 GB card.
    """
    if (teacher_logits is None) == (teacher_chunk_fn is None):
        raise ValueError("pass exactly one of teacher_logits / teacher_chunk_fn.")
    rows = student_logits.shape[0]
    flat_w = weights.reshape(-1)
    total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, rows, LK_CHUNK_ROWS):
        stop = min(rows, start + LK_CHUNK_ROWS)

        def chunk_term(student_chunk, weight_chunk, start=start, stop=stop):
            teacher_chunk = (
                teacher_chunk_fn(start, stop)
                if teacher_chunk_fn is not None
                else teacher_logits[start:stop]
            )
            ell = lk_per_position(student_chunk, teacher_chunk, loss_type=loss_type, eta=eta)
            return (ell * weight_chunk).sum()

        student_chunk = student_logits[start:stop]
        weight_chunk = flat_w[start:stop]
        if torch.is_grad_enabled() and student_chunk.requires_grad:
            total = total + checkpoint(chunk_term, student_chunk, weight_chunk, use_reentrant=False)
        else:
            total = total + chunk_term(student_chunk, weight_chunk)
    denom = flat_w.sum().clamp_min(1e-6).to(torch.float32)
    return total / denom


def e2e_tv_term(
    student_logits_pb: torch.Tensor,
    teacher_logits_pb: torch.Tensor | None = None,
    valid_mask_pb: torch.Tensor | None = None,
    *,
    teacher_chunk_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end multi-step TV loss + detached accepted length.

    ``student`` / ``teacher`` are ``[B, n, S, V]``; ``valid_mask_pb`` is the
    ``[B, n, S]`` slot mask (masked slots act as identity in the prefix chain,
    so a per-block accepted length only advances over supervised slots).
    Chunked over blocks; returns ``(loss, accept_length)`` scalars. The teacher
    side may also be a lazy ``teacher_chunk_fn(row_start, row_stop)`` over flat
    slot rows (``block * S + slot``), which is how the trainer avoids
    materializing the full teacher tensor.

    Like :func:`lk_term`, every chunk is activation-checkpointed when grad is
    enabled, so the fp32 teacher/student probability slabs are recomputed in
    the backward instead of being retained across the whole run of chunks.
    """
    if (teacher_logits_pb is None) == (teacher_chunk_fn is None):
        raise ValueError("pass exactly one of teacher_logits_pb / teacher_chunk_fn.")
    bsz, nblocks, slots, _ = student_logits_pb.shape
    blocks = bsz * nblocks
    student_flat = student_logits_pb.reshape(blocks, slots, -1)
    teacher_flat = None if teacher_chunk_fn is not None else teacher_logits_pb.reshape(blocks, slots, -1)
    mask_flat = valid_mask_pb.reshape(blocks, slots).float()

    loss_sum = torch.zeros((), device=student_flat.device, dtype=torch.float32)
    accept_sum = torch.zeros((), device=student_flat.device, dtype=torch.float32)
    valid_blocks = torch.zeros((), device=student_flat.device, dtype=torch.float32)
    for start in range(0, blocks, E2E_CHUNK_BLOCKS):
        stop = min(blocks, start + E2E_CHUNK_BLOCKS)

        def chunk_terms(student_chunk, mask_chunk, start=start, stop=stop):
            if teacher_chunk_fn is not None:
                t_chunk = teacher_chunk_fn(start * slots, stop * slots)
                t = torch.softmax(t_chunk.view(stop - start, slots, -1).float(), dim=-1)
            else:
                t = torch.softmax(teacher_flat[start:stop].float(), dim=-1)
            s = torch.softmax(student_chunk.float(), dim=-1)
            alpha = torch.minimum(t, s).sum(dim=-1)  # [chunk, slots], grad via student branch
            alpha_effective = alpha * mask_chunk + (1.0 - mask_chunk)
            prefix_prod = torch.cumprod(alpha_effective, dim=-1)
            gamma_valid = mask_chunk.sum(dim=-1).clamp(min=1.0)
            accept_length_pb = (prefix_prod * mask_chunk).sum(dim=-1)
            e2e_per_block = 1.0 - accept_length_pb / gamma_valid
            block_has_valid = (mask_chunk.sum(dim=-1) > 0).float()
            return (
                (e2e_per_block * block_has_valid).sum(),
                (accept_length_pb.detach() * block_has_valid).sum(),
                block_has_valid.sum(),
            )

        student_chunk = student_flat[start:stop]
        mask_chunk = mask_flat[start:stop]
        if torch.is_grad_enabled() and student_chunk.requires_grad:
            loss_part, accept_part, valid_part = checkpoint(
                chunk_terms, student_chunk, mask_chunk, use_reentrant=False
            )
        else:
            loss_part, accept_part, valid_part = chunk_terms(student_chunk, mask_chunk)
        loss_sum = loss_sum + loss_part
        accept_sum = accept_sum + accept_part
        valid_blocks = valid_blocks + valid_part
    denom = valid_blocks.clamp_min(1.0)
    return loss_sum / denom, accept_sum / denom


def per_slot_nll(
    student_logits: torch.Tensor, target_ids: torch.Tensor, *, chunk_rows: int = LK_CHUNK_ROWS
) -> torch.Tensor:
    """fp32 per-slot NLL ``[B, n, S]`` (chunked over positions)."""
    shape = target_ids.shape
    flat_logits = student_logits.reshape(-1, student_logits.size(-1))
    flat_targets = target_ids.reshape(-1)
    out = []
    for start in range(0, flat_logits.shape[0], chunk_rows):
        stop = min(flat_logits.shape[0], start + chunk_rows)
        out.append(
            F.cross_entropy(
                flat_logits[start:stop].float(), flat_targets[start:stop], reduction="none"
            )
        )
    nll = torch.cat(out, dim=0) if len(out) > 1 else out[0]
    return nll.view(shape)


def recipe_loss(
    stage: RecipeStage,
    *,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    block_mask: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    teacher_chunk_fn=None,
    loss_decay_gamma: float | None,
) -> tuple[torch.Tensor, dict]:
    """Loss for one recipe stage + logging components.

    ``student_logits`` ``[B, n, S, V]`` (supervised slots only), ``target_ids``
    ``[B, n, S]``, ``block_mask`` ``[B, n, S]`` float (no objective weighting),
    and the teacher distribution as either ``teacher_logits`` ``[B, n, S, V]``
    (materialized) or ``teacher_chunk_fn(row_start, row_stop)`` over flat slot
    rows (lazy; the trainer's :class:`TeacherLogitsStream`). Any stage whose
    ``lk_coef`` or ``e2e_tv_coef`` is non-zero requires the teacher; pass
    exactly one of the two forms.
    """
    if stage.ce_coef <= 0.0 and stage.lk_coef <= 0.0 and stage.e2e_tv_coef <= 0.0:
        raise ValueError(f"recipe stage {stage.name!r} has no active loss term.")
    mask = block_mask.to(torch.float32)

    needs_teacher = stage.lk_coef > 0.0 or stage.e2e_tv_coef > 0.0
    if needs_teacher and teacher_logits is None and teacher_chunk_fn is None:
        raise ValueError(
            f"recipe stage {stage.name!r} needs the target's final hidden states as the teacher "
            "signal, but the batch carries none: enable capture.store_final_hidden=true and train "
            "from a feature store with stores_final_hidden=true (or run online capture with it on)."
        )
    if teacher_logits is not None and teacher_chunk_fn is not None:
        raise ValueError("pass exactly one of teacher_logits / teacher_chunk_fn to recipe_loss.")

    nll = None
    if stage.ce_coef > 0.0 or stage.weighting == "dpace":
        nll = per_slot_nll(student_logits, target_ids)

    if stage.weighting == "dpace":
        confidences = torch.exp(-nll.detach())
        position_weights = dpace_position_weights(confidences, stage.dpace_alpha)
        weights = mask * position_weights
    elif stage.weighting == "decay":
        gamma = 7.0 if loss_decay_gamma is None else float(loss_decay_gamma)
        depth = torch.exp(-torch.arange(mask.size(-1), device=mask.device, dtype=torch.float32) / gamma)
        weights = mask * depth.view(1, 1, -1)
    else:
        raise ValueError(f"unknown stage weighting {stage.weighting!r}.")
    denom = weights.sum().clamp_min(1e-6)

    components: dict[str, float] = {}
    total = torch.zeros((), device=student_logits.device, dtype=torch.float32)

    if stage.ce_coef > 0.0:
        ce_term = (nll * weights).sum() / denom
        total = total + float(stage.ce_coef) * ce_term
        components["ce_loss"] = float(ce_term.detach())

    if stage.lk_coef > 0.0:
        lk = lk_term(
            student_logits.reshape(-1, student_logits.size(-1)),
            None if teacher_chunk_fn is not None else teacher_logits.reshape(-1, teacher_logits.size(-1)),
            weights.reshape(-1),
            loss_type=stage.lk_loss_type,
            eta=stage.lk_eta,
            teacher_chunk_fn=teacher_chunk_fn,
        )
        total = total + float(stage.lk_coef) * lk
        components["lk_loss"] = float(lk.detach())

    if stage.e2e_tv_coef > 0.0:
        e2e, accept_len = e2e_tv_term(
            student_logits, teacher_logits, mask, teacher_chunk_fn=teacher_chunk_fn
        )
        total = total + float(stage.e2e_tv_coef) * e2e
        components["e2e_tv_loss"] = float(e2e.detach())
        components["e2e_accept_len"] = float(accept_len)

    return total, components


class TeacherLogitsStream:
    """Lazy teacher next-token logits over the supervised slots.

    The teacher LM at position ``p`` emits the distribution for token ``p+1``,
    so slot ``k`` (predicting the token at ``label_indices[..., k]``) reads the
    teacher hidden state at ``label_indices[..., k] - 1`` — the same convention
    the trainer uses for the teacher-forced previous tokens. Computed under
    ``no_grad`` in the head's dtype (ports upstream's detached teacher).

    Rows are flat ``(B * n) * S + ...`` slots in block-major order: row ``r``
    belongs to block ``r // S`` (batch ``(r // S) // n``) and slot ``r % S``.
    :meth:`logits` projects a bounded slab on demand, so the full
    ``[B, n, S, V]`` teacher tensor never materializes (it is 8.7 GB at the
    canonical 16-sample sanity batch).
    """

    def __init__(
        self,
        lm_head,
        teacher_hidden: torch.Tensor,
        label_indices: torch.Tensor,
        *,
        first_slot: int,
    ) -> None:
        bsz, nblocks, block_size = label_indices.shape
        slots = int(block_size) - int(first_slot)
        if slots < 1:
            raise ValueError(
                f"first_slot={first_slot} leaves no supervised slots in block_size={block_size}."
            )
        seq_len = teacher_hidden.size(1)
        positions = (
            (label_indices - 1).clamp(min=0, max=max(0, seq_len - 1))[:, :, int(first_slot):]
        ).reshape(-1)
        self.num_rows = int(positions.numel())
        self.num_blocks = int(bsz) * int(nblocks)
        self.slots = slots
        self.seq_len = int(seq_len)
        self.hidden_size = int(teacher_hidden.size(-1))
        self._lm_head = lm_head
        self._hidden = teacher_hidden.reshape(-1, self.hidden_size)
        self._positions = positions
        self._batch = (
            torch.arange(self.num_rows, device=positions.device) // (int(nblocks) * slots)
        )

    def logits(self, row_start: int, row_stop: int) -> torch.Tensor:
        """Teacher logits for flat rows ``[row_start, row_stop)``, shape ``[k, V]``."""
        start = max(0, min(int(row_start), self.num_rows))
        stop = max(start, min(int(row_stop), self.num_rows))
        index = self._batch[start:stop] * self.seq_len + self._positions[start:stop]
        rows = self._hidden[index]
        with torch.no_grad():
            return self._lm_head(rows)

    def full(self) -> torch.Tensor:
        """Materialize every row (``[B*n*S, V]``); for tests / small batches."""
        return self.logits(0, self.num_rows)


def gather_teacher_logits(
    lm_head,
    teacher_hidden: torch.Tensor,
    label_indices: torch.Tensor,
    *,
    first_slot: int,
) -> torch.Tensor:
    """Teacher next-token logits for every supervised slot, shape ``[B, n, S, V]``.

    Thin materializing wrapper around :class:`TeacherLogitsStream` for tests and
    small batches; the trainer streams (:meth:`TeacherLogitsStream.logits`)
    instead, which is the only way to stay memory-bounded at canonical batches.
    """
    bsz, nblocks, block_size = label_indices.shape
    stream = TeacherLogitsStream(lm_head, teacher_hidden, label_indices, first_slot=first_slot)
    return stream.full().view(bsz, nblocks, block_size - int(first_slot), -1)
