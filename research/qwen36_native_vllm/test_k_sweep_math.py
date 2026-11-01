"""CPU checks of short-block boundaries, position continuation and sequential correction."""
import importlib
import os
import unittest
from types import SimpleNamespace as NS

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import contract
import spatial_adapter as spatial
import dflash2_adapter as df2
import dfly_adapter as fly


class ShortBlockMath(unittest.TestCase):
    def tearDown(self):
        os.environ.pop('Q36_SWEEP_K', None)
        importlib.reload(contract)

    def set_k(self, k):
        os.environ['Q36_SWEEP_K'] = str(k)
        importlib.reload(contract)

    def test_query_axes_continue_from_last_accepted_position(self):
        axes = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        for k in range(1, 8):
            self.set_k(k)
            actual = spatial.spatial_query_positions(axes, torch.tensor([0, 4]), torch.tensor([2]), k + 1)
            expected = axes[:, 1:2] + torch.arange(1, k + 2)[None, :]
            torch.testing.assert_close(actual, expected)
            spatial.validate_scoped_query_batch(k + 1, k + 1, 'PIECEWISE', dummy=False, keys_initialized=True)
            with self.assertRaises(ValueError):
                spatial.validate_scoped_query_batch(k + 1, k + 2, 'PIECEWISE', dummy=False, keys_initialized=True)

    def test_convolution_never_reads_previous_block(self):
        for k in range(1, 8):
            self.set_k(k); b = k + 1
            x = torch.arange(2 * b * 4, dtype=torch.float32).reshape(2 * b, 4)
            delta = torch.zeros(2 * b, 2, 2)
            base = torch.tensor([[2., 2., 2., 2.], [3., 3., 3., 3.]])
            actual = df2.grouped_dynamic_conv(x, delta, base, block_size=contract.BLOCK_SIZE, group_size=2)
            expected = torch.stack([2 * x[i] + (3 * x[i-1] if i % b else 0) for i in range(2 * b)])
            torch.testing.assert_close(actual, expected)

    def test_correction_uses_actual_previous_selection_for_every_k(self):
        class Model:
            def __init__(self):
                self.model = NS(hidden_correction=lambda h, e: h + e)
                self.seen = []
            def embed_input_ids(self, ids):
                self.seen.append(int(ids.item()))
                return torch.nn.functional.one_hot(ids, 3).float()
            def compute_logits(self, h):
                return h
        for k in range(1, 8):
            self.set_k(k); model = Model()
            h = torch.tensor([[0., 2., 0.], [0., 0., 3.], [4., 0., 0.]]).repeat(3, 1)[:k]
            expected = []; prev = 0
            for row in h:
                prev = int((row + torch.nn.functional.one_hot(torch.tensor(prev), 3)).argmax())
                expected.append(prev)
            actual = fly.sequential_greedy(model, h, torch.tensor([0]))
            self.assertEqual(actual.tolist(), expected)
            self.assertEqual(model.seen, [0] + expected[:-1])

    def test_launch_requires_matching_k_and_graph_capture_width(self):
        from vllm.config import CompilationMode, CUDAGraphMode
        for k in range(1, 8):
            self.set_k(k)
            cfg = NS(parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1),
                     scheduler_config=NS(max_num_seqs=1),
                     model_config=NS(hf_config=NS(model_type='qwen3_5_moe'), enforce_eager=False),
                     speculative_config=NS(num_speculative_tokens=k, draft_model_config=NS(uses_mrope=True), enforce_eager=False),
                     compilation_config=NS(mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE, cudagraph_capture_sizes=[k+1]))
            contract.validate_launch(cfg)
            cfg.speculative_config.num_speculative_tokens = k + 1
            with self.assertRaises(ValueError): contract.validate_launch(cfg)
            cfg.speculative_config.num_speculative_tokens = k
            cfg.compilation_config.cudagraph_capture_sizes = [k+2]
            with self.assertRaises(ValueError): contract.validate_launch(cfg)


if __name__ == '__main__':
    unittest.main()
