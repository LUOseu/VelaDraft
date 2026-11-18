"""Refresh the preliminary K table from real result files; never infer missing speeds."""
import json
from pathlib import Path
import re
import time

TASK = Path('/home/lyh/qwen36_k_sweep_20261101')
PATTERN = re.compile(r'(dflash|dflash2|dfly)_k([1357])_r1(?:_a\d+)?\.json')

def refresh():
    found = {}
    for path in sorted(TASK.glob('*_r1*.json'), key=lambda p: p.stat().st_mtime):
        match = PATTERN.fullmatch(path.name)
        if not match: continue
        try: value = json.loads(path.read_text())
        except ValueError: continue
        found[match[1], int(match[2])] = (path, value)
    rows = []
    for method in ('dflash', 'dflash2', 'dfly'):
        for k in (1, 3, 5, 7):
            row = dict(method=method, k=k, status='pending', diagnostic_only=True)
            entry = found.get((method, k))
            if entry:
                path, value = entry
                row.update(status=value.get('status'), result_file=str(path))
                if value.get('status') == 'complete' and value.get('resources_verified'):
                    row.update(speedup=value['speedup_sum_ar_over_sd'], ar_seconds=value['ar_seconds'], sd_seconds=value['sd_seconds'], ar_output_tokens=value['ar_output_tokens'], sd_output_tokens=value['sd_output_tokens'], exact_pairs=value['exact_output_pairs'], timed_pairs=value['timed_pairs'], CI95=value['session_cluster_CI95'])
                elif value.get('error'): row['error'] = value['error']
            # A preserved failed attempt may already have a newer attempt running.
            attempts = sorted(TASK.glob(f'{method}_k{k}_r1*.evidence'), key=lambda p: p.stat().st_mtime)
            if attempts and not attempts[-1].with_suffix('.json').exists():
                row.update(status='running', evidence=str(attempts[-1]))
            rows.append(row)
    table = ['# Qwen3.6 A6000 K sweep — preliminary', '', '12 sessions / 16 turns, one paired repeat per K. Native AR; TP2. These are diagnostic test-subset results, not the full paper table.', '', '| Method | K | Status | E2E speedup | AR seconds | SD seconds |', '|---|---:|---|---:|---:|---:|']
    for row in rows:
        values = [f'{row[x]:.3f}' if x in row else '—' for x in ('speedup', 'ar_seconds', 'sd_seconds')]
        table.append(f"| {row['method']} | {row['k']} | {row['status']} | " + ' | '.join(values) + ' |')
    table += ['', 'Failed attempts and raw evidence are preserved. Sequence equality and output lengths are reported separately in RESULTS.json; no quality-equivalence claim.', 'A small-sample point estimate above 1 does not establish a general or statistically robust speedup.']
    for name, content in [('RESULTS.json', json.dumps({'updated_unix': time.time(), 'rows': rows}, indent=2)), ('RESULTS.md', '\n'.join(table)+'\n')]:
        tmp = TASK/(name+'.tmp'); tmp.write_text(content);tmp.replace(TASK/name)
    return sum(r['status']=='complete' for r in rows)

if __name__ == '__main__':
    while True:
        count=refresh()
        if count==12: break
        time.sleep(60)
