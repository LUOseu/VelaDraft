import unittest
from launch_override import LaunchOverride

class LaunchTests(unittest.TestCase):
    def test_only_changes_warp_count_and_preserves_output(self):
        calls=[]
        class Kernel:
            def __getitem__(self,grid):
                def run(*a,**kw):calls.append((grid,a,kw));return 'original-result'
                return run
        args={'num_warps':4,'num_stages':3,'q':'tensor','IS_SPEC_DECODING':True}
        out=LaunchOverride(Kernel(),1)[(1,4,16)](**args)
        self.assertEqual(out,'original-result');self.assertEqual(args['num_warps'],4)
        self.assertEqual(calls,[((1,4,16),(),dict(args,num_warps=1))])
    def test_invalid_setting_rejected(self):
        with self.assertRaises(ValueError):LaunchOverride(None,3)

if __name__=='__main__':unittest.main()
