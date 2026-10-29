"""
MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DFlash2 local dynamic convolution, ported from the Z-Lab DFlash2 architecture as
vendored by vllm-project/speculators @ 55ebbd5b
(`src/speculators/models/dflash2/model_definitions.py`, itself adapted from
https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py).
Math, initialization, and block-local tap semantics are unchanged; imports and
docstrings are local. See docs/upstream/dflash2.md.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["GroupedDynamicCausalConv", "grouped_dynamic_conv"]


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta_kernel: torch.Tensor,
    base_kernel: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Apply a token-conditioned grouped causal convolution within draft blocks.

    A tap never reads across an anchor-block boundary: training flattens many
    anchored blocks into one token dimension, so each token's in-block offset is
    ``index % block_size`` and tap ``t`` is masked out when ``offset < t``.
    """
    hidden_size = hidden_states.shape[-1]
    if hidden_size % group_size:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size}).")

    kernel_size = base_kernel.shape[0]
    if kernel_size > block_size:
        raise ValueError(f"kernel_size ({kernel_size}) cannot exceed block_size ({block_size}).")
    num_groups = hidden_size // group_size
    if base_kernel.shape != (kernel_size, hidden_size):
        raise ValueError(
            f"base_kernel must have shape [kernel_size, hidden_size], got {tuple(base_kernel.shape)}."
        )
    expected_delta_shape = (*hidden_states.shape[:-1], kernel_size, num_groups)
    if delta_kernel.shape != expected_delta_shape:
        raise ValueError(
            f"delta_kernel must have shape {expected_delta_shape}, got {tuple(delta_kernel.shape)}."
        )

    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, num_groups, group_size)
    flat_delta = delta_kernel.reshape(-1, kernel_size, num_groups)
    positions = torch.arange(flat_hidden.shape[0], device=hidden_states.device).remainder(block_size)
    output = torch.zeros_like(flat_hidden)

    for tap in range(kernel_size):
        if tap == 0:
            shifted = flat_hidden
        else:
            padding = flat_hidden.new_zeros(tap, num_groups, group_size)
            shifted = torch.cat([padding, flat_hidden[:-tap]], dim=0)
        coefficient = (
            base_kernel[tap].to(flat_hidden.dtype).view(1, num_groups, group_size)
            + flat_delta[:, tap, :, None]
        )
        valid = positions.ge(tap).view(-1, 1, 1)
        output = output + shifted * coefficient * valid

    return output.reshape(original_shape)


class GroupedDynamicCausalConv(nn.Module):
    """Input/output local convolutions driven by one hidden-state projection.

    ``prepare`` convolves the sublayer input and returns the output-side dynamic
    kernel; ``finish`` convolves the sublayer output with that kernel. With the
    default initialization (``base_kernel[0] = 1``, everything else zero) both
    calls are the identity, so a freshly built model is DFlash-equivalent until
    training moves the kernels.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        block_size: int,
        kernel_size: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size}).")
        if kernel_size > block_size:
            raise ValueError(f"kernel_size ({kernel_size}) cannot exceed block_size ({block_size}).")
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * self.num_groups, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize both convolutions as identity transforms."""
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve sublayer inputs and return its output-side dynamic kernel."""
        kernels = self.kernel_projection(hidden_states).view(
            *hidden_states.shape[:-1],
            2,
            self.kernel_size,
            self.num_groups,
        )
        prepared = grouped_dynamic_conv(
            hidden_states,
            kernels[..., 0, :, :],
            self.base_kernel[0],
            block_size=self.block_size,
            group_size=self.group_size,
        )
        return prepared, kernels[..., 1, :, :]

    def finish(self, hidden_states: torch.Tensor, delta_kernel: torch.Tensor) -> torch.Tensor:
        """Convolve sublayer outputs with the saved output-side kernel."""
        return grouped_dynamic_conv(
            hidden_states,
            delta_kernel,
            self.base_kernel[1],
            block_size=self.block_size,
            group_size=self.group_size,
        )
