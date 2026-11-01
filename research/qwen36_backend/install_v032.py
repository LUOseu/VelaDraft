"""Install a resolved public runtime into an isolated environment with a disk floor."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

root=Path('/root/autodl-tmp/qwen36_backend_20261101')
log=root/'v032_resolve_clean.log'
pins=re.findall(r'^ \+ ([A-Za-z0-9_.-]+==\S+)\s*$',log.read_text(),re.M)
assert 'vllm==0.32.0' in pins and len(pins)>100
lock=root/'v032_resolved_requirements.txt';lock.write_text('\n'.join(pins)+'\n')
assert shutil.disk_usage(root).free>20*(1<<30),'Less than 20 GiB free before isolated install'
env=dict(os.environ,TMPDIR=str(root/'install_tmp'),UV_HTTP_TIMEOUT='60',UV_HTTP_RETRIES='2')
for name in ('UV_INDEX','UV_INDEX_URL','UV_EXTRA_INDEX_URL','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL'):env.pop(name,None)
Path(env['TMPDIR']).mkdir(exist_ok=True)
cmd=['/root/autodl-tmp/qwen36_tp1_20261101/runtime/bin/uv','pip','install','--python',str(root/'runtime_v032/bin/python'),
     '--index-url','https://pypi.org/simple','--no-cache','--link-mode','hardlink','--no-deps','-r',str(lock)]
with (root/'v032_install.log').open('x') as stream:
    child=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    status='running'
    while child.poll() is None:
        free=shutil.disk_usage(root).free
        (root/'v032_install_STATUS.json').write_text(json.dumps({'status':status,'pid':child.pid,'free_gib':free/(1<<30),'pins':len(pins)}))
        if free<4*(1<<30):
            os.killpg(child.pid,signal.SIGTERM);status='stopped_disk_floor';child.wait(timeout=30);break
        time.sleep(5)
    if status=='running':status='complete' if child.returncode==0 else 'failed'
    (root/'v032_install_STATUS.json').write_text(json.dumps({'status':status,'returncode':child.returncode,'free_gib':shutil.disk_usage(root).free/(1<<30),'pins':len(pins)}))
raise SystemExit(0 if status=='complete' else 1)
