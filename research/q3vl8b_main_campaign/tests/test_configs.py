import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('prepare_configs', Path(__file__).resolve().parents[1] / 'prepare_configs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def base(method):
    return {'algorithm': {'name': method}, 'dataset': {'seed': 42},
            'target': {'device': 'cuda:1', 'dtype': 'bfloat16'},
            'capture': {'seq_length': 4096}, 'drafter': {'block_size': 8, 'num_anchors': 512},
            'train': {'seed': 42, 'num_epochs': 3, 'micro_batch_size': 2,
                      'grad_accumulation_steps': 4, 'online_feature_cache_gb': 0}}


def test_eight_paired_arms_preserve_recipe_and_vary_only_execution_switches():
    configs = module.derive({'dflash2': base('dflash2'), 'dfly': base('dfly')})
    assert len(configs) == 8
    assert configs['dfly_s17_ref']['train']['seed'] == 17
    assert configs['dfly_s17_opt']['train']['seed'] == 17
    assert configs['dfly_s29_opt']['drafter']['num_anchors'] == 512
    assert configs['dflash2_s17_ref']['dataset']['seed'] == 42
    assert configs['dflash2_s17_opt']['target']['device'] == 'cuda:0'
    assert configs['dflash2_s17_ref']['drafter']['dflash2_teacher_projection_reuse'] is False
    assert configs['dflash2_s17_opt']['drafter']['dflash2_teacher_projection_reuse'] is True


def test_reduced_or_wrong_recipe_fails_closed():
    c = base('dflash2')
    c['drafter']['num_anchors'] = 256
    try:
        module.derive({'dflash2': c, 'dfly': base('dfly')})
    except ValueError:
        pass
    else:
        raise AssertionError('reduced 256-anchor config accepted')
