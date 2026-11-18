"""Real public auto-loader selection and production export/receipt boundaries."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from . import contracts as c


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='selection_',dir=c.HERE/'evidence') as directory:
        parent=Path(directory);export=parent/'hf';export.mkdir()
        real=c.HERE/'hf_step75000_v1'
        shutil.copy2(real/'config.json',export/'config.json')
        # Read-only test hardlinks reuse the actual51-tensor artifact without copying or changing it.
        os.link(real/'model.safetensors',export/'model.safetensors')
        shutil.copy2(real/'export_receipt.json',export/'export_receipt.json')
        # Synthetic fixture only; never used as production CPU evidence.
        receipt={'status':'complete','all51_values_bitwise_equal':True,'cuda_initialized':False,
                 'checkpoint_sha256':c.CHECKPOINT_SHA,'tensor_count':51}
        receipt['schema']=getattr(c,'RECEIPT_SCHEMA','dflash2-native-cpu-v1')
        receipt['export_dir']=str(export)
        receipt['export_artifacts']={str(export/name):c.sha(export/name) for name in ('config.json','model.safetensors','export_receipt.json')}
        receipt['own_source_sha256']=c.own_sources()
        from .public_config import materialize_public_config
        receipt['actual_public_config'],_=materialize_public_config(export)
        yield parent,export,receipt


def auto_selection(export):
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.config import LoadConfig
    loader=get_model_loader(LoadConfig())
    return loader._prepare_weights(str(export),subfolder=None,revision=None,fall_back_to_pt=True,allow_patterns_overrides=None)


class WeightSelectionTests(unittest.TestCase):
    def require_v3(self):
        self.assertTrue(hasattr(c,'validate_export_layout'),'Missing production exact-directory binding')
        self.assertTrue(hasattr(c,'validate_draft_loader_selection'),'Missing real public loader binding')
        self.assertTrue(hasattr(c,'RECEIPT_SCHEMA'),'Missing explicitly pinned V3 receipt contract')

    def test_real_public_auto_loader_selects_only_bound_actual51_file(self):
        self.require_v3()
        from .public_config import materialize_public_config
        with fixture() as (_,export,receipt):
            folder,files,safe=auto_selection(export)
            self.assertEqual(Path(folder),export);self.assertEqual(files,[str(export/'model.safetensors')]);self.assertTrue(safe)
            _,vc=materialize_public_config(export)
            proof=c.validate_draft_loader_selection(vc,receipt)
            self.assertEqual(proof['selected_files'],files);self.assertTrue(proof['use_safetensors'])
            self.assertFalse(torch.cuda.is_initialized())

    def test_real_auto_loader_index_can_select_extra51_file_but_guard_rejects_it(self):
        self.require_v3()
        with fixture() as (_,export,receipt):
            alternate=export/'unbound51.safetensors';os.link(export/'model.safetensors',alternate)
            index={'metadata':{},'weight_map':{key:alternate.name for key in c.SCHEMA}}
            (export/'model.safetensors.index.json').write_text(json.dumps(index))
            _,files,safe=auto_selection(export)
            self.assertEqual(files,[str(alternate)]);self.assertTrue(safe)
            with self.assertRaises(ValueError):c.validate_export_layout(export,receipt)

    def test_extra_weight_index_bin_and_other_entries_are_rejected(self):
        self.require_v3()
        with fixture() as (_,export,receipt):
            for name in ('extra.safetensors','model.safetensors.index.json','pytorch_model.bin','weights.pt','unreviewed.txt'):
                extra=export/name;extra.write_bytes(b'unbound')
                with self.subTest(name=name),self.assertRaises(ValueError):c.validate_export_layout(export,receipt)
                extra.unlink()

    def test_export_and_all_three_entries_must_be_regular_non_symlink_paths(self):
        self.require_v3()
        with fixture() as (parent,export,receipt):
            for name in ('config.json','model.safetensors','export_receipt.json'):
                original=export/name;saved=parent/name;original.rename(saved)
                original.symlink_to(saved)
                with self.subTest(name=name),self.assertRaises(ValueError):c.validate_export_layout(export,receipt)
                original.unlink();saved.rename(original)
            alias=parent/'alias';alias.symlink_to(export,target_is_directory=True)
            with self.assertRaises(ValueError):c.validate_export_layout(alias,receipt)
            (export/'model.safetensors').unlink();(export/'model.safetensors').mkdir()
            with self.assertRaises(ValueError):c.validate_export_layout(export,receipt)

    def test_artifact_keys_and_all_parents_must_close_over_export(self):
        self.require_v3()
        with fixture() as (parent,export,receipt):
            wrong=copy.deepcopy(receipt);original=str(export/'model.safetensors')
            wrong['export_artifacts'][str(parent/'other.safetensors')]=wrong['export_artifacts'].pop(original)
            with self.assertRaises(ValueError):c.validate_export_layout(export,wrong)
            wrong=copy.deepcopy(receipt);wrong['export_dir']=str(parent)
            with self.assertRaises(ValueError):c.validate_export_layout(export,wrong)

    def test_actual_spec_model_and_draft_ModelConfig_paths_are_both_bound(self):
        self.require_v3()
        from .public_config import materialize_public_config
        with fixture() as (_,export,receipt):
            _,vc=materialize_public_config(export)
            c.validate_launch_config(vc,receipt)
            for role in ('spec','ModelConfig'):
                wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
                if role=='spec':wrong.speculative_config.model='/unbound/spec-model'
                else:
                    wrong.speculative_config.draft_model_config=copy.copy(vc.speculative_config.draft_model_config)
                    wrong.speculative_config.draft_model_config.model='/unbound/draft-ModelConfig'
                with self.subTest(role=role),self.assertRaises(ValueError):c.validate_launch_config(wrong,receipt)
            wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
            wrong.speculative_config.draft_model_config=copy.copy(vc.speculative_config.draft_model_config)
            wrong.speculative_config.draft_model_config.hf_config=copy.deepcopy(vc.speculative_config.draft_model_config.hf_config)
            wrong.speculative_config.draft_model_config.hf_config.hidden_act='relu'
            with self.assertRaises(ValueError):c.validate_launch_config(wrong,receipt)

    def test_dummy_custom_nonauto_and_quantized_loader_configs_fail_before_selection(self):
        self.require_v3()
        from .public_config import materialize_public_config
        from vllm.config import LoadConfig
        from vllm.model_executor.model_loader import get_model_loader
        self.assertIn('Dummy',type(get_model_loader(LoadConfig(load_format='dummy'))).__name__)
        with fixture() as (_,export,receipt):
            _,vc=materialize_public_config(export)
            bad_loaders=(LoadConfig(load_format='dummy'),LoadConfig(load_format='safetensors'),
                LoadConfig(model_loader_extra_config={'unreviewed_custom_loader':'elsewhere'}))
            for config in bad_loaders:
                wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
                wrong.speculative_config.draft_load_config=config
                with self.subTest(config=str(config)),self.assertRaises(ValueError):c.validate_draft_loader_selection(wrong,receipt)
            wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
            wrong.speculative_config.draft_model_config=copy.copy(vc.speculative_config.draft_model_config)
            wrong.speculative_config.draft_model_config.quantization='bitsandbytes'
            with self.assertRaises(ValueError):c.validate_draft_loader_selection(wrong,receipt)
            wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
            wrong.speculative_config.draft_load_config=LoadConfig(safetensors_load_strategy='eager')
            with self.assertRaises(ValueError):c.validate_draft_loader_selection(wrong,receipt)

    def test_production_receipt_requires_external_expected_sha_and_rejects_tampering(self):
        self.require_v3()
        with fixture() as (parent,_,receipt):
            path=parent/'receipt.json';path.write_text(json.dumps(receipt))
            expected=c.sha(path)
            env={'DFLASH2_NATIVE_CPU_RECEIPT':str(path),'DFLASH2_NATIVE_CPU_RECEIPT_SHA256':''}
            with mock.patch.dict(os.environ,env):
                with self.assertRaises(ValueError):c.runtime_receipt()
                os.environ['DFLASH2_NATIVE_CPU_RECEIPT_SHA256']='0'*64
                with self.assertRaises(ValueError):c.runtime_receipt()
                os.environ['DFLASH2_NATIVE_CPU_RECEIPT_SHA256']=expected
                self.assertEqual(c.runtime_receipt()['export_dir'],receipt['export_dir'])
                receipt['own_source_sha256']={'forged.py':'0'*64};path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):c.runtime_receipt()

    def test_fresh_production_bootstrap_aborts_without_explicit_receipt_sha(self):
        self.require_v3()
        with fixture() as (parent,_,receipt):
            path=parent/'receipt.json';path.write_text(json.dumps(receipt))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DFLASH2_NATIVE_ENABLE='1',
                DFLASH2_NATIVE_CPU_RECEIPT=str(path),DFLASH2_NATIVE_CPU_RECEIPT_SHA256='',
                PYTHONPATH=str(c.HERE/'bootstrap')+os.pathsep+str(c.HERE.parent))
            result=subprocess.run([sys.executable,'-c','print("UNREVIEWED_BOOTSTRAP_REACHED_MAIN")'],
                env=env,cwd=c.HERE.parent,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0,'Production bootstrap accepted an unpinned receipt')
            self.assertNotIn('UNREVIEWED_BOOTSTRAP_REACHED_MAIN',result.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
