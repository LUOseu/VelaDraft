import unittest
from summary import summarize_stage

def record(rank, phase, q, ms):
    return {'rank':rank,'spans':[{'component':'target_forward','step':1,'phase':phase,'scheduled_tokens':q,'stream_elapsed_ms':ms,'host_ms':ms/2}]}

class SummaryTests(unittest.TestCase):
    def test_execute_sampling_excludes_draft_and_nested_forward(self):
        ranks=[]
        for rank in (0,1):
            r=record(rank,'verify',4,1)
            for label,ms in [('target_execute_inclusive',4+rank),('target_sampling_rejection',2+rank),('proposal_total_inclusive',100)]:
                r['spans'].append(dict(r['spans'][0],component=label,stream_elapsed_ms=ms,host_ms=ms))
            ranks.append(r)
        rows=summarize_stage({'arm':'sd','k':3,'rows':[{'id':'x','turn':0,'ranks':ranks}]})
        total=next(x for x in rows if x['component']=='verification_execute_plus_sampling')
        self.assertEqual(total['mean_stream_ms'],8)

    def test_uses_slowest_rank_not_sum(self):
        d={'arm':'sd','k':3,'rows':[{'id':'x','turn':0,'ranks':[record(0,'verify',4,2),record(1,'verify',4,3)]}]}
        r=summarize_stage(d)[0]
        self.assertEqual(r['mean_stream_ms'],3);self.assertEqual(r['calls'],1)

    def test_excludes_prefill_and_short_final_verify_from_main_curve(self):
        d={'arm':'sd','k':3,'rows':[{'id':'x','turn':0,'ranks':[record(0,'prefill',99,100),record(1,'prefill',99,110)]}]}
        self.assertEqual(summarize_stage(d),[])

    def test_missing_rank_is_an_error(self):
        d={'arm':'ar','k':1,'rows':[{'id':'x','turn':0,'ranks':[record(0,'decode',1,2)]}]}
        with self.assertRaises(ValueError):summarize_stage(d)

if __name__=='__main__':unittest.main()
