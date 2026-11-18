"""Frozen real-gold workload; predictions never enter the answer source."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
INSTRUCTION='\nAnswer the question using a single word or phrase.'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load_task(name):
    pins=json.loads((ROOT/'DATA_MANIFEST.json').read_text())['tasks'][name]
    for path,sha in pins['files'].items():
        if digest(path)!=sha:raise ValueError('Frozen gold/prompt metadata changed: '+path)
    prompts=[json.loads(s) for s in Path(pins['prompts']).read_text().splitlines()]
    targets=[json.loads(s) for s in Path(pins['targets']).read_text().splitlines()]
    if len(prompts)!=pins['count'] or len(targets)!=pins['count']:
        raise ValueError('Gold workload scope changed')
    ids=[x['id'] for x in prompts]
    if len(set(ids))!=len(ids) or ids!=[x['id'] for x in targets]:raise ValueError('Gold IDs duplicated or reordered')
    records=[];directory=Path(pins['root'])/'images'
    for prompt,target in zip(prompts,targets):
        conversations=prompt['conversations']
        if (len(conversations)!=2 or conversations[0]['from']!='human'
                or conversations[1]['from']!='gpt' or conversations[1]['value']!=''):
            raise ValueError('Quality prompt contains an unexpected answer/history')
        question=conversations[0]['value']
        if not question.endswith(INSTRUCTION):raise ValueError('Frozen short-answer instruction changed')
        original=question[:-len(INSTRUCTION)]
        question_sha=hashlib.sha256(json.dumps(original,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if question_sha!=target['question_sha256']:raise ValueError('Gold was paired with another question')
        if prompt['images']!=[target['image']] or prompt['group_id']!=target['group_id']:
            raise ValueError('Gold image/group identity differs')
        image=(directory/target['image']).resolve()
        if directory.resolve() not in image.parents or digest(image)!=target['image_sha256']:
            raise ValueError('Quality image bytes or root changed')
        answers=target['answers']
        if not isinstance(answers,list) or not answers or any(not isinstance(x,str) for x in answers):
            raise ValueError('Missing real string-list gold answers')
        if name=='textvqa' and len(answers)!=10:raise ValueError('All ten TextVQA answers are required')
        records.append({'id':prompt['id'],'question':question,'image':str(image),
                        'image_sha256':target['image_sha256'],'group_id':target['group_id'],'answers':answers})
    return pins,records
