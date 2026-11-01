"""Use a faster mirror while requiring hashes obtained from official PyPI."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

root=Path('/root/autodl-tmp/qwen36_backend_20261101')
pins=(root/'v032_resolved_requirements.txt').read_text().splitlines()
def hashes(pin):
    name,version=pin.split('==')
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as response:
        metadata=json.load(response)
    values=sorted({file['digests']['sha256'] for file in metadata['urls']})
    if not values:raise ValueError('Official release has no distribution hashes: '+pin)
    return pin+' '+ ' '.join('--hash=sha256:'+value for value in values)
with ThreadPoolExecutor(max_workers=12) as pool:lines=list(pool.map(hashes,pins))
lock=root/'v032_pypi_hashes.txt';lock.write_text('\n'.join(lines)+'\n')
print('OFFICIAL_PYPI_HASHES_PINNED',len(lines),flush=True)
assert shutil.disk_usage(root).free>20*(1<<30)
env=dict(os.environ,TMPDIR=str(root/'install_tmp_mirror'),UV_HTTP_TIMEOUT='60',UV_HTTP_RETRIES='2',UV_CONCURRENT_DOWNLOADS='8')
for key in ('UV_INDEX','UV_INDEX_URL','UV_EXTRA_INDEX_URL','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL'):env.pop(key,None)
Path(env['TMPDIR']).mkdir(exist_ok=True)
cmd=['/root/autodl-tmp/qwen36_tp1_20261101/runtime/bin/uv','pip','install','--python',str(root/'runtime_v032/bin/python'),
     '--index-url','https://pypi.tuna.tsinghua.edu.cn/simple','--no-cache','--link-mode','hardlink','--no-deps','--require-hashes','-r',str(lock)]
with (root/'v032_install_mirror.log').open('x') as stream:
    child=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    status='running'
    while child.poll() is None:
        free=shutil.disk_usage(root).free
        (root/'v032_install_mirror_STATUS.json').write_text(json.dumps({'status':status,'pid':child.pid,'free_gib':free/(1<<30),'official_hashes':len(lines)}))
        if free<4*(1<<30):
            os.killpg(child.pid,signal.SIGTERM);status='stopped_disk_floor';child.wait(timeout=30);break
        time.sleep(5)
    if status=='running':status='complete' if child.returncode==0 else 'failed'
    (root/'v032_install_mirror_STATUS.json').write_text(json.dumps({'status':status,'returncode':child.returncode,'free_gib':shutil.disk_usage(root).free/(1<<30),'official_hashes':len(lines)}))
raise SystemExit(0 if status=='complete' else 1)
