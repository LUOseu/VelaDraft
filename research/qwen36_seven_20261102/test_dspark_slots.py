"""Exercise the official Markov sampler with the checkpoint's anchor-slot layout."""
import unittest
from types import SimpleNamespace
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
try:
    from research.qwen36_seven_20261102.dspark_bridge import anchor_slots_for_native_sampler
except ImportError:
    anchor_slots_for_native_sampler = None


class SlotTests(unittest.TestCase):
    def sample(self, k, use_markov):
        self.assertIsNotNone(anchor_slots_for_native_sampler, 'Anchor-slot adapter is missing')
        hidden = torch.zeros(k+1, 16, dtype=torch.bfloat16)
        for i in range(k+1):
            hidden[i, i+1] = 10
        def bias(previous):
            output = torch.zeros(1, 16, dtype=torch.bfloat16)
            if use_markov:
                output[0, (int(previous[0])+2) % 16] = 20
            return output
        model = SimpleNamespace(compute_draft_logits=lambda h:h,
            markov_embed=lambda t:t, markov_bias=bias)
        spec = SimpleNamespace(_draft_topk=None, num_speculative_steps=k,
            sample_indices=torch.arange(1,k+1), sample_pos=torch.arange(1,k+1),
            sample_idx_mapping=torch.zeros(k,dtype=torch.long), model=model,
            use_confidence_head=False, input_buffers=SimpleNamespace(input_ids=torch.tensor([0]+[7]*k)),
            _anchor_idx=torch.tensor([0]), draft_tokens=torch.zeros(1,k,dtype=torch.long),
            _sample_logits=lambda logits,*args:logits.argmax(-1))
        DSparkSpeculator._sample_sequential(spec, 1, anchor_slots_for_native_sampler(hidden))
        return spec.draft_tokens[0].tolist()

    def test_anchor_row_is_first_prediction(self):
        for k in (1,2,3,5,7):
            with self.subTest(k=k):
                self.assertEqual(self.sample(k,False),list(range(1,k+1)))

    def test_native_markov_walk_uses_previous_sample(self):
        self.assertEqual(self.sample(3,True),[2,4,6])


if __name__=='__main__':unittest.main()
