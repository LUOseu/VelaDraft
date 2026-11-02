"""Opt-in per-worker diagnostic spans; never imported by the E2E timer runs."""
from functools import wraps
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import time


class Recorder:
    def __init__(self, cuda):
        self.cuda = cuda
        self.active = False
        self.current = {}
        self.records = []
        self.originals = []
        self.step = 0
        self.scope = lambda name: nullcontext()

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def measured(*args, **kwargs):
            if not self.active: return original(*args, **kwargs)
            begin, end = self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
            metadata = dict(self.current)
            first = args[0] if args else next((kwargs[x] for x in ('context_states','target_hidden_states','input_ids','inputs_embeds') if kwargs.get(x) is not None), None)
            if hasattr(first, 'shape'):
                metadata['first_tensor_shape'] = list(first.shape)
            with self.scope('q36::'+label):
                begin.record(); start = time.perf_counter_ns()
                try: return original(*args, **kwargs)
                finally:
                    host = (time.perf_counter_ns() - start) / 1e6
                    end.record()
                    self.records.append((label, metadata, host, begin, end))
        self.originals.append((obj, name, original))
        setattr(obj, name, measured)

    def finish(self):
        self.active = False
        self.cuda.synchronize()
        return [dict(component=label, host_ms=host, stream_elapsed_ms=a.elapsed_time(b), **meta)
                for label, meta, host, a, b in self.records]


class ProfileWorker:
    def q36_profile_install(self):
        import torch
        if hasattr(self, '_q36_recorder'): raise RuntimeError('Hooks already installed')
        runner = self.model_runner
        rec = Recorder(torch.cuda)
        self._q36_recorder = rec
        rec.wrap(runner, 'execute_model', 'target_execute_inclusive')
        original_execute = runner.execute_model
        @wraps(original_execute)
        def tagged(scheduler_output, *args, **kwargs):
            spec = getattr(scheduler_output, 'scheduled_spec_decode_tokens', {})
            proposals = sum(len(x) for x in spec.values())
            q = int(scheduler_output.total_num_scheduled_tokens)
            rec.step += 1
            rec.current = dict(step=rec.step, scheduled_tokens=q, scheduled_draft_tokens=proposals,
                               phase='verify' if proposals else ('decode' if q == 1 else 'prefill'))
            return original_execute(scheduler_output, *args, **kwargs)
        runner.execute_model = tagged
        rec.wrap(runner, '_model_forward', 'target_forward')
        rec.wrap(runner, '_sample', 'target_sampling_rejection')
        rec.wrap(runner, 'sample_tokens', 'sampling_and_proposal_inclusive')
        draft = getattr(runner, 'drafter', None)
        if draft is not None:
            for name, label in [('propose', 'proposal_total_inclusive'),
                                ('set_inputs_first_pass', 'proposal_positions_inputs'),
                                ('build_per_group_and_layer_attn_metadata', 'proposal_attention_metadata'),
                                ('build_model_inputs_first_pass', 'proposal_build_inputs_inclusive'),
                                ('_sample_draft_tokens', 'proposal_sampling_inclusive')]:
                rec.wrap(draft, name, label)
            for name, label in [('combine_hidden_states', 'proposal_feature_fusion'),
                                ('precompute_and_store_context_kv', 'proposal_context_kv'),
                                ('forward', 'proposal_backbone'),
                                ('compute_logits', 'proposal_lm_head')]:
                rec.wrap(draft.model, name, label)
        runner._diagnostic_hooks_installed = True
        return {'components': [x[1] for x in rec.originals], 'rank': getattr(self, 'rank', None)}

    def q36_profile_begin(self,label='request'):
        rec = self._q36_recorder
        rec.cuda.synchronize(); rec.records.clear(); rec.current = {}; rec.step = 0; rec.active = True
        trace=os.environ.get('Q36_TRACE_DIR')
        if trace:
            import torch
            self._q36_trace_label=re.sub(r'[^A-Za-z0-9_-]','_',label)
            self._q36_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
            self._q36_profiler.start()
            rec.scope=torch.profiler.record_function
        return True

    def q36_profile_end(self):
        result={'rank': getattr(self, 'rank', None), 'spans': self._q36_recorder.finish(),
                'semantics': 'host spans and CUDA stream elapsed time; inclusive spans overlap nested children; not kernel-only time and not headline E2E'}
        prof=getattr(self,'_q36_profiler',None)
        if prof is not None:
            prof.stop();self._q36_recorder.scope=lambda name:nullcontext()
            root=Path(os.environ['Q36_TRACE_DIR']);root.mkdir(parents=True,exist_ok=True)
            path=root/f'{self._q36_trace_label}_rank{result["rank"]}.json'
            prof.export_chrome_trace(str(path));result['trace_file']=str(path);self._q36_profiler=None
        return result
