"""The drafter-algorithm seam: what a method must provide to the shared stack.

The training loop, dataset/capture pipeline, evaluator and exporter are shared
across algorithms; everything method-specific (model architecture, training
objective, checkpoint fields, decoder) is reached through this interface. The
first implementation is :class:`vlm_dflash.algorithms.dflash.DFlashAlgorithm`;
new methods register themselves in :mod:`vlm_dflash.algorithms.registry`.

Contract (kept deliberately small; see ``docs/adding_an_algorithm.md``):

- ``build_draft_config`` / ``build_trainer_module`` produce the training-side
  objects from the target contract and the run config.
- ``load_decoder`` returns the inference-side module for eval/export. It must
  expose ``spec_generate(target=..., input_ids=..., ...)`` with the same
  signature as the DFlash draft model (the evaluator only relies on that
  surface).
- ``checkpoint_metadata`` returns the algorithm-specific fields stored in every
  checkpoint; ``eval``/``export`` refuse checkpoints whose recorded algorithm
  does not match the resolved one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmMetadata:
    """Machine-readable support statement (rendered into docs, not just prose).

    ``maturity`` values should be one of ``stable``/``validated``/
    ``experimental``/``architecture-only``/``unsupported``; ``targets`` maps
    target families (``qwen25_vl``, ``qwen3_vl``) to those same states plus
    ``reduced_validation`` (full pipeline recorded on that family at a
    reduced protocol -- smaller slice/prompt budget; see
    ``RELEASE_VALIDATION.md``). Maturity describes how trustworthy the
    algorithm is, ``targets`` how far the pipeline is validated per family,
    and ``capabilities`` what the implementation supports at all -- keep the
    three axes separate and derived from recorded runs.
    """

    name: str
    description: str
    maturity: dict[str, str]
    references: dict[str, str]
    capabilities: dict[str, bool]
    targets: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "maturity": dict(self.maturity),
            "references": dict(self.references),
            "capabilities": dict(self.capabilities),
            "targets": dict(self.targets),
        }


class DrafterAlgorithm(ABC):
    """Interface every parallel/diffusion drafter implementation must satisfy."""

    name: str
    metadata: AlgorithmMetadata

    @abstractmethod
    def validate_config(self, cfg: dict) -> None:
        """Reject configs the algorithm cannot train with (fail loud, early)."""

    @abstractmethod
    def build_draft_config(self, contract: dict, cfg: dict):
        """Build the draft model config from the target contract + run config."""

    @abstractmethod
    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight,
        lm_head_weight=None,
    ):
        """Training module: ``module(**feature_batch) -> metrics`` with ``.loss``."""

    @abstractmethod
    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        """Return ``(decoder, checkpoint_meta)`` for eval/export."""

    @abstractmethod
    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        """Algorithm-specific checkpoint fields (re-read by eval/export)."""

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        """Rebuild the inference-side drafter from a serialized draft config.

        Used by export loading; ``draft_config_dict`` is the ``draft_config``
        field of an algorithm checkpoint / export (a ``Qwen3Config.to_dict()``).
        """
        raise NotImplementedError(
            f"algorithm {self.name!r} does not implement build_inference_module()."
        )

    def configure_verify_backend(self, reference: bool) -> None:
        """Toggle verify-time numerics for the shared speculative inference path.

        Every algorithm registered here reuses the shared block-parallel
        ``spec_generate`` machinery, whose reference behavior narrows the
        verify-time SDPA backends (a cuDNN-plan / throughput choice) unless
        parity mode is requested. The default implementation forwards to that
        shared switch so an evaluation's ``verify_sdpa_backend`` setting can
        never silently become a no-op for a plugin; algorithms with an
        independent inference path may override this.
        """
        from ..dflash import draft_model

        draft_model.set_verify_sdpa_backend_switch(bool(reference))

    def plan_knobs(self, cfg: dict, contract: dict | None = None) -> dict:
        """Effective algorithm knobs recorded in the training-plan digest.

        The shared payload already covers the drafter-common sections; an
        algorithm whose config changes the training trajectory beyond those
        (e.g. DFlare/DFly fusion context, K/V layout, hidden correction)
        overrides this to return its *resolved* knobs with defaults filled in.

        ``contract`` is the frozen run contract; normalize layout-dependent
        values against it (``context_layer_ids: null`` means "every captured
        layer", so the effective list is only known with the capture layout).
        Raising when the contract is required but missing is intentional: a
        digest computed without it could silently disagree with the gate.
        """
        return {}
