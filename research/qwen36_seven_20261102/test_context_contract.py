import unittest
import torch
try:
    from research.qwen36_seven_20261102.context_bridge import build_contexts
except ImportError:
    build_contexts=None


class ContextTests(unittest.TestCase):
    def test_layer_fusion_and_dfly_base_are_both_retained(self):
        self.assertIsNotNone(build_contexts,'Raw-context bridge missing')
        raw=torch.cat([torch.full((2,2048),x) for x in (1.,3.,8.)],dim=-1)
        fusion=torch.zeros(3,3)
        norm=lambda x:x
        for method,expected in (('dflare',4.),('dfly',6.)):
            contexts=build_contexts(method,raw,fusion,norm,lambda x:2*x[:,:2048])
            self.assertEqual(len(contexts),3)
            for context in contexts:
                torch.testing.assert_close(context,torch.full((2,2048),expected))

    @unittest.skipUnless(torch.cuda.is_available(),'GPU replay check runs separately before model timing')
    def test_graph_replay_reads_new_raw_context_for_each_proposal(self):
        self.assertIsNotNone(build_contexts)
        for method in ('dflare','dfly'):
            raw=torch.ones(4,6144,device='cuda',dtype=torch.bfloat16)
            fusion=torch.zeros(3,3,device='cuda',dtype=torch.bfloat16)
            def compute():
                return torch.stack(build_contexts(method,raw,fusion,lambda x:x,lambda x:2*x[:,:2048]))
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):compute()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=compute()
            snapshots=[]
            for value in (2.,5.):
                raw.fill_(value);graph.replay();torch.cuda.synchronize()
                expected=compute();torch.cuda.synchronize()
                torch.testing.assert_close(output,expected,rtol=0,atol=0)
                snapshots.append(output.clone())
            self.assertFalse(torch.equal(*snapshots),'Captured graph used stale raw context')


if __name__=='__main__':unittest.main()
