"""DSpark heads: low-rank Markov logit bias and a per-slot confidence head.

Ported from the official implementations (see ``docs/upstream/dspark.md``):
``vllm-project/speculators`` @ ``55ebbd5b``
``src/speculators/models/dspark/{core,model_definitions}.py`` and
``NVIDIA-NeMo/Automodel`` @ ``bb5b34ce``
``nemo_automodel/components/speculative/dspark/markov_head.py`` (Apache-2.0).

Semantics kept from upstream:

- ``MarkovHead`` learns a low-rank logit bias ``B = W2 @ W1``: ``W1`` embeds the
  previous token (full target vocabulary), ``W2`` projects back to the target
  vocabulary, and the bias is added onto the block-parallel draft logits.
- Three variants: ``vanilla`` (bias from the previous token alone), ``gated``
  (bias gated by the block hidden state), ``rnn`` (a recurrent state carried
  across the block, as in NeMo's ``RNNHead``).
- ``ConfidenceHead`` is a linear->scalar head predicting each draft position's
  acceptance probability.

Deliberate adaptation (recorded in the audit doc): the upstream confidence
target is the analytical acceptance rate ``1 - TV(p_draft, p_target)`` computed
against verifier logits during online distillation. Our pipeline is staged and
trains on cached hard-target trajectories (no verifier logits on disk), so the
confidence target here is the greedy per-slot acceptance indicator (draft
argmax == target token) — the exact event the temperature-0 verification gate
measures.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "ConfidenceHead",
    "GatedMarkovHead",
    "RNNHead",
    "VanillaMarkov",
    "build_markov_head",
]


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Greedy (temperature ~ 0) or temperature sampling over the last dim."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits.view(-1, logits.shape[-1]) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(logits.shape[:-1])


class VanillaMarkov(nn.Module):
    """First-order Markov logit bias: ``W2(W1(prev_token))``."""

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}.")
        self.markov_head_type = "vanilla"
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        # Upstream (speculators) initialises W1 with a small normal; the default
        # N(0, 1) embedding init would inject an O(1) logit bias at step 0.
        nn.init.normal_(self.markov_w1.weight, std=0.01)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def compute_block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [.., block_size]
        hidden_states: torch.Tensor | None,  # [.., block_size, hidden]
        prev_emb: torch.Tensor | None = None,  # [.., block_size, rank]
    ) -> torch.Tensor:
        """Teacher-forced block bias ``[.., block_size, vocab]``."""
        if prev_emb is None:
            prev_emb = self.get_prev_embeddings(prev_token_ids)
        return self.project_bias(prev_emb)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,  # [B, num_blocks, block_size, vocab]
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        return base_logits + self.compute_block_bias(
            prev_token_ids=token_ids, hidden_states=hidden_states
        )

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,  # [B, proposal_len, vocab]
        *,
        first_prev_token_ids: torch.Tensor,  # [B]
        hidden_states: torch.Tensor | None,  # [B, proposal_len, hidden]
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Semi-autoregressive sampling: each slot's bias uses the previous slot's token.

        Returns ``(sampled_tokens [B, proposal_len], corrected_logits [B, proposal_len, vocab])``.
        """
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
            step_logits = base_logits[:, step_idx, :] + self.compute_step_bias(prev_token_ids, step_hidden)
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(step_logits.unsqueeze(1), temperature=temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkov):
    """Markov bias gated by the block hidden state (NeMo ``GatedMarkovHead``)."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_inputs)).to(dtype=prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):
    """Recurrent Markov head: a GRU-like state carried across block positions.

    Training unrolls the state over the whole block (teacher-forced);
    :meth:`sample_block_tokens` carries the state across sampled slots.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [B, N, block_size]
        hidden_states: torch.Tensor | None,
        prev_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden states.")
        leading = prev_token_ids.shape[:-1]
        block_size = prev_token_ids.shape[-1]
        state = torch.zeros(*leading, self.markov_rank, device=prev_token_ids.device, dtype=hidden_states.dtype)
        biases = []
        for k in range(block_size):
            prev_emb_k = self.get_prev_embeddings(prev_token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb_k, hidden_states[..., k, :])
            biases.append(bias)
        return torch.stack(biases, dim=-2)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        state = torch.zeros(batch_size, self.markov_rank, device=base_logits.device, dtype=hidden_states.dtype)
        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx, :])
            step_logits = base_logits[:, step_idx, :] + bias
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = _sample(step_logits.unsqueeze(1), temperature=temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def build_markov_head(
    *,
    vocab_size: int,
    markov_rank: int,
    markov_head_type: str = "vanilla",
    hidden_size: int = 0,
) -> nn.Module | None:
    """Build the configured Markov head (``None`` when ``markov_rank == 0``)."""
    markov_rank = int(markov_rank)
    if markov_rank < 0:
        raise ValueError(f"markov_rank must be >= 0, got {markov_rank}")
    if markov_rank == 0:
        return None
    head_type = str(markov_head_type).lower()
    if head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if head_type == "gated":
        return GatedMarkovHead(vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size)
    if head_type == "rnn":
        return RNNHead(vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size)
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r} (expected vanilla/gated/rnn)")
