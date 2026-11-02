import unittest
from research.qwen36_grouped_e2e_20261102.dispatch import eligible,make_dispatcher


class DispatchTests(unittest.TestCase):
    def test_ar_and_wide_inputs_use_existing_path(self):
        for q in (0,1,2,9,64,4096):self.assertFalse(eligible(q,8,256,None))
        for q in range(3,9):self.assertTrue(eligible(q,8,256,None))

    def test_unsupported_geometry_mapping_and_quantization_fall_back(self):
        self.assertFalse(eligible(8,4,256,None))
        self.assertFalse(eligible(8,8,128,None))
        self.assertFalse(eligible(8,8,256,object()))
        self.assertFalse(eligible(8,8,256,None,use_int4_w4a16=True))
        self.assertFalse(eligible(8,8,256,None,block_shape=[128,128]))

    def test_forwards_same_ids_and_mask_semantics(self):
        ids=object();cfg={'BLOCK_SIZE_M':16};seen=[]
        def native(*a,**kw):seen.append(('native',a,kw));return 'n'
        def aligned(*a,**kw):seen.append(('aligned',a,kw));return 'g'
        f=make_dispatcher(native,aligned)
        self.assertEqual(f(ids,cfg,1,8,256,None,ignore_invalid_experts=True),'n')
        self.assertEqual(f(ids,cfg,8,8,256,None,ignore_invalid_experts=True),'g')
        self.assertIs(seen[0][1][0],ids);self.assertIs(seen[1][1][0],ids)
        self.assertEqual(seen[1][1],(ids,16,256,None));self.assertTrue(seen[1][2]['ignore_invalid_experts'])


if __name__=='__main__':unittest.main()
