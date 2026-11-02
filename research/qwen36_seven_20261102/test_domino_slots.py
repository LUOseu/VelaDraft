import unittest
from types import SimpleNamespace
import torch
try:
    from research.qwen36_seven_20261102.domino_bridge import sample_anchor_prefix
except ImportError:
    sample_anchor_prefix=None


class DominoSlotTests(unittest.TestCase):
    def test_real_reference_walk_gets_anchor_prefix_not_mask_rows(self):
        self.assertIsNotNone(sample_anchor_prefix,'Domino layout bridge missing')
        model=SimpleNamespace(
            compute_logits=lambda h:h,
            prefix_gru=SimpleNamespace(weight_ih_l0=torch.zeros(9,16),weight_hh_l0=torch.zeros(9,3)),
            embed_input_ids=lambda ids:torch.zeros(len(ids),16),
            embed_proj=lambda x:torch.zeros(x.shape[0],16))
        for k in (1,2,3,5,7):
            hidden=torch.zeros(k+1,16)
            for i in range(k+1):hidden[i,i+1]=10
            actual=sample_anchor_prefix(model,hidden,torch.tensor([0]),k)
            self.assertEqual(actual.tolist(),list(range(1,k+1)))


if __name__=='__main__':unittest.main()
