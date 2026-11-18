import importlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class ReferenceScoringTests(unittest.TestCase):
    def scorer(self):
        try:return importlib.import_module('quality_eval_v1.scoring')
        except ModuleNotFoundError:self.fail('Pinned reference scoring module is missing')

    def test_textvqa_exact_leave_one_out_credit(self):
        scorer=self.scorer()
        for count,expected in [(1,.3),(2,.6),(3,.9),(4,1.)]:
            gold=['yes']*count+['no'+str(i) for i in range(10-count)]
            self.assertAlmostEqual(scorer.score('textvqa','yes',gold),expected)

    def test_textvqa_reference_decimal_contractions_and_articles(self):
        normalize=self.scorer().normalize_textvqa
        self.assertEqual(normalize('3.14'),'3.14')
        self.assertEqual(normalize("don't"),"don't")
        self.assertEqual(normalize('The two'),'2')

    def test_docvqa_strict_half_threshold_and_case_space_rules(self):
        score=self.scorer().score
        self.assertEqual(score('docvqa','ac',['ab']),0)
        self.assertEqual(score('docvqa','HELLO',['hello']),1)
        self.assertAlmostEqual(score('docvqa','a  b',['a b']),.75)

    def test_chartqa_reference_numeric_percent_and_zero_rules(self):
        score=self.scorer().score
        self.assertEqual(score('chartqa','0.5',['50%']),1)
        self.assertEqual(score('chartqa','105',['100']),1)
        self.assertEqual(score('chartqa','106',['100']),0)
        self.assertEqual(score('chartqa','0.0',['0']),0)
        self.assertEqual(score('chartqa','1000',['1,000']),0)
        self.assertEqual(score('chartqa','one  two',['one two']),0)

    def test_full_output_is_not_silently_answer_extracted(self):
        score=self.scorer().score
        self.assertEqual(score('chartqa','Answer: blue',['blue']),0)
        self.assertEqual(score('chartqa','blue\nexplanation',['blue']),0)


if __name__=='__main__':unittest.main()
