"""Install an explicitly pinned SD adapter in both client and spawned workers."""
import os
if os.environ.get('VELADRAFT_NATIVE_PLAN'):
    try:
        import json
        from pathlib import Path
        import runpy
        import sys
        from vlm_dflash.eval.native import digest,verify_plan,validate_environment
        path=Path(os.environ['VELADRAFT_NATIVE_PLAN'])
        if digest(path)!=os.environ['VELADRAFT_NATIVE_PLAN_SHA256']:raise ValueError('Native plan changed')
        plan=json.loads(path.read_text());verify_plan(plan)
        adapter=plan.get('adapter')
        if os.environ['VELADRAFT_NATIVE_ARM']=='sd' and adapter:
            original=list(sys.path);os.environ.update(adapter.get('environment',{}))
            namespace=runpy.run_path(adapter['bootstrap'],run_name='__veladraft_native_adapter__')
            if adapter.get('call'):namespace[adapter['call']]()
            sys.path[:]=original+[p for p in sys.path if p not in original]
        validate_environment(os.environ)
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(86)
