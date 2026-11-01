"""Keep unequal T/H/W axes and rejected suffixes correct across requests."""
import unittest
import torch
from research.qwen36_v032.spatial_adapter import query_axes


class PositionTests(unittest.TestCase):
    def test_rejected_suffix_preserves_each_axis(self):
        axes=torch.tensor([[5,6,7,8],[10,11,12,13],[20,21,22,23]])
        actual=query_axes(axes,torch.tensor([0,4]),torch.tensor([2]),3)
        self.assertTrue(torch.equal(actual,torch.tensor([[7,8,9],[12,13,14],[22,23,24]])))

    def test_request_boundaries_and_zero_rejection(self):
        axes=torch.tensor([[2,3,10,11,12],[5,6,20,21,22],[8,9,30,31,32]])
        actual=query_axes(axes,torch.tensor([0,2,5]),torch.tensor([0,1]),2)
        self.assertTrue(torch.equal(actual,torch.tensor([[4,5,12,13],[7,8,22,23],[10,11,32,33]])))


if __name__=='__main__':unittest.main()
