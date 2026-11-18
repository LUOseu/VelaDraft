import unittest
from .compare_quality import paired_delta,validate_pair

class PairedQualityTest(unittest.TestCase):
 def test_clusters_are_resampled_as_whole_groups(self):
  r=paired_delta([0,0,1],[1,0,1],['page1','page1','page2'],samples=2000,seed=42)
  self.assertAlmostEqual(r['delta_percentage_points'],100/3)
  self.assertEqual(r['clusters'],2)
  self.assertEqual(r['questions'],3)
 def test_identical_scores_produce_zero_interval(self):
  r=paired_delta([1,0],[1,0],['a','b'])
  self.assertEqual(r['paired_cluster_percentile_CI95'],[0.,0.])
 def test_pair_requires_same_prompt_and_real_gold(self):
  a={'id':'1','group_id':'p','prompt_token_ids':[1,2],'image_sha256':'x','gold_answers':['gold'],'status':'complete'}
  validate_pair(a,dict(a))
  for field,value in (('prompt_token_ids',[2]),('gold_answers',['other']),('image_sha256','wrong'),('group_id','other'),('status','failed')):
   b=dict(a);b[field]=value
   with self.assertRaises(ValueError):validate_pair(a,b)
if __name__=='__main__':unittest.main()
