"""Pinned upstream TextVQA and Pix2Struct reference metric semantics."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT=Path(__file__).resolve().parent
SOURCES=json.loads((ROOT/'vendor/SOURCES.json').read_text())
for name,entry in SOURCES.items():
    if hashlib.sha256((ROOT/'vendor'/name).read_bytes()).hexdigest()!=entry['sha256']:
        raise RuntimeError('Pinned upstream metric source/license changed: '+name)

from .vendor.m4c_evaluators import EvalAIAnswerProcessor,TextVQAAccuracyEvaluator


def levenshtein(left,right):
    if len(left)<len(right):left,right=right,left
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def load_reference_function(name):
    source=ROOT/'vendor/pix2struct_metrics.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'Optional':Optional,'editdistance':SimpleNamespace(eval=levenshtein)}
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,str(source),'exec'),scope)
    return scope[name]


_RELAXED=load_reference_function('relaxed_correctness')
_ANLS=load_reference_function('anls_metric')
_TEXT=TextVQAAccuracyEvaluator()
normalize_textvqa=EvalAIAnswerProcessor()


def score(task,prediction,answers):
    if not isinstance(prediction,str) or not isinstance(answers,list) or not answers or any(not isinstance(a,str) for a in answers):
        raise ValueError('A decoded prediction and real string-list gold answers are required')
    if task=='textvqa':
        if len(answers)!=10:raise ValueError('TextVQA reference requires all ten human answers')
        return _TEXT.eval_pred_list([{'pred_answer':prediction,'gt_answers':answers}])
    if task=='docvqa':
        if any(not a for a in answers):raise ValueError('DocVQA gold must be nonempty')
        return max(_ANLS(a.lower(),prediction.lower()) for a in answers)
    if task=='chartqa':return float(max(_RELAXED(a,prediction) for a in answers))
    raise ValueError('Unknown quality task: '+task)
