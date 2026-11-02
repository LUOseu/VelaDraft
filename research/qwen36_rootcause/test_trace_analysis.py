import unittest
from analyze_trace import analyze

class TraceTests(unittest.TestCase):
    def test_maps_async_gpu_kernel_by_launch_correlation_not_cpu_time(self):
        events=[{'cat':'user_annotation','name':'q36::target_forward','ph':'X','ts':10,'dur':10,'tid':1},
                {'cat':'cuda_runtime','name':'cudaGraphLaunch','ts':12,'dur':1,'tid':1,'args':{'correlation':7}},
                {'cat':'kernel','name':'fused_moe_kernel','ts':1000,'dur':4,'args':{'correlation':7}},
                {'cat':'kernel','name':'other','ts':1002,'dur':20,'args':{'correlation':8}}]
        spans=[{'component':'target_forward','phase':'decode','scheduled_tokens':1,'stream_elapsed_ms':.01}]
        out=analyze(events,spans,'target_forward','decode',1)
        self.assertEqual(out['matched_kernel_count'],1);self.assertEqual(out['kernel_activity_us_per_step']['moe_expert_gemm'],4)
    def test_unmatched_cpu_span_counts_fail(self):
        with self.assertRaises(ValueError):analyze([], [{'component':'target_forward'}], 'target_forward','decode',1)

if __name__=='__main__':unittest.main()
