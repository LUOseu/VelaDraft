"""CPU coverage of official dummy scopes and strict real eight-query dispatch."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import official_spatial_graph_adapter_v3 as graph


def make_dispatcher():
    from vllm.config import CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    comp = NS(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
              is_attention_compiled_piecewise=lambda: True,
              max_cudagraph_capture_size=16, cudagraph_capture_sizes=[8, 16],
              compile_sizes=None)
    cfg = NS(compilation_config=comp, speculative_config=NS(num_speculative_tokens=7),
             lora_config=None, scheduler_config=NS(max_num_seqs=1),
             parallel_config=NS(data_parallel_size=1))
    dispatcher = CudagraphDispatcher(cfg)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE)
    return cfg, dispatcher


class ScopedGuardRegression(unittest.TestCase):
    def test_dummy_query_bounds_and_padding(self):
        for n in (1, 2, 4, 8):
            for padded in (n, 8):
                graph.validate_scoped_query_batch(n, padded, "NONE", dummy=True,
                    graphs_requested=False, keys_initialized=True)
        for n, padded in ((0, 0), (9, 9), (2, 1), (1, 16), (8, 16)):
            with self.assertRaises(ValueError):
                graph.validate_scoped_query_batch(n, padded, "NONE", dummy=True,
                    graphs_requested=False, keys_initialized=True)

    def test_real_propose_exactly_eight_and_piecewise(self):
        graph.validate_scoped_query_batch(8, 8, "PIECEWISE", dummy=False,
            graphs_requested=True, keys_initialized=True)
        for n, padded, mode in ((1, 1, "PIECEWISE"), (8, 16, "PIECEWISE"), (8, 8, "NONE")):
            with self.assertRaises(ValueError):
                graph.validate_scoped_query_batch(n, padded, mode, dummy=False,
                    graphs_requested=False, keys_initialized=True)

    def test_actual_official_dispatcher_startup_and_real_batches(self):
        cfg, dispatcher = make_dispatcher()
        for n in (1, 2, 4, 8):
            for graphs_requested in (False, True):
                from vllm.config import CUDAGraphMode
                mode, desc = dispatcher.dispatch(n,
                    valid_modes=None if graphs_requested else {CUDAGraphMode.NONE})
                graph.validate_scoped_query_batch(n, desc.num_tokens, mode.name, dummy=True,
                    graphs_requested=graphs_requested, keys_initialized=True)
        mode, desc = dispatcher.dispatch(8)
        graph.validate_scoped_query_batch(8, desc.num_tokens, mode.name, dummy=False,
            graphs_requested=True, keys_initialized=True)

    def make_proposer(self):
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        cfg, dispatcher = make_dispatcher()
        class Proposer:
            def dummy_run(self, n, fail=False, nested_real=False):
                result = self._determine_batch_execution_and_padding(n, use_cudagraphs=False)
                if nested_real:
                    return self.propose(n)
                if fail:
                    raise RuntimeError("dummy failed")
                return result
            def propose(self, n=8):
                return self._determine_batch_execution_and_padding(n)
            _determine_batch_execution_and_padding = SpecDecodeBaseProposer._determine_batch_execution_and_padding
        graph.install_scoped_query_guards(Proposer, SpecDecodeBaseProposer._determine_batch_execution_and_padding)
        proposer = Proposer()
        proposer.vllm_config = cfg
        proposer.cudagraph_dispatcher = dispatcher
        proposer.draft_model_config = NS(hf_config=NS(dflash_config={"official_spatial_adapter_v1": True}))
        return proposer

    def test_scope_is_restored_after_dummy_success_and_failure(self):
        proposer = self.make_proposer()
        self.assertFalse(graph.in_dummy_scope())
        for n in (1, 2, 4, 8):
            proposer.dummy_run(n)
            self.assertFalse(graph.in_dummy_scope())
        with self.assertRaisesRegex(RuntimeError, "dummy failed"):
            proposer.dummy_run(1, fail=True)
        self.assertFalse(graph.in_dummy_scope())
        with self.assertRaises(ValueError):
            proposer.propose(1)
        self.assertFalse(graph.in_dummy_scope())
        proposer.propose(8)

    def test_nested_real_propose_cannot_inherit_dummy_permission(self):
        proposer = self.make_proposer()
        with self.assertRaises(ValueError):
            proposer.dummy_run(1, nested_real=True)
        self.assertFalse(graph.in_dummy_scope())

    def test_v3_installer_preserves_v2_metadata_wrapper_and_math(self):
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
        forward, call = DFlashQwen3Model.forward, DFlashQwen3Model.__call__
        bindings = graph.install()
        self.assertIn("graph_v2_adapter", bindings)
        self.assertIs(DFlashQwen3Model.forward, forward)
        self.assertIs(DFlashQwen3Model.__call__, call)
        self.assertTrue(hasattr(DFlashQwen3Model.__init__, "__wrapped__"))

    def test_prior_runtime_files_are_preserved(self):
        root = Path(__file__).parent
        for name, expected in {
            "official_spatial_graph_adapter.py": "748538d2b1e91e8c9bd69a902c7b78f7c57cec333bfb977a0b1694749be74f87",
            "official_spatial_graph_adapter_v2.py": "3c66daaa63db7731de144eda1d215ab2d1623a7a4ee836d4109df583382698c8",
            "official_spatial_adapter.py": "17c9c160ba86770238a43f0ba898d840976225cc38a374f82136f0cceb0213ff",
        }.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
