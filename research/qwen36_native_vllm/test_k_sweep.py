"""Budget sweep guards: trained B8 weights stay fixed; runtime query width is K+1."""
import importlib
import os
import unittest
from unittest.mock import patch

import contract


class BudgetContract(unittest.TestCase):
    def tearDown(self):
        os.environ.pop('Q36_SWEEP_K', None)
        importlib.reload(contract)

    def test_default_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('Q36_SWEEP_K', None)
            importlib.reload(contract)
            self.assertEqual((contract.SPECULATIVE_TOKENS, contract.BLOCK_SIZE), (7, 8))

    def test_runtime_budget_does_not_change_training_contract(self):
        for k in range(1, 8):
            with self.subTest(k=k), patch.dict(os.environ, {'Q36_SWEEP_K': str(k)}):
                importlib.reload(contract)
                self.assertEqual(contract.SPECULATIVE_TOKENS, k)
                self.assertEqual(contract.BLOCK_SIZE, k + 1)
                cfg = dict(contract.DRAFT_FIELDS, block_size=8,
                           dflash_config=dict(contract.DFLASH_FIELDS),
                           rope_parameters={'rope_theta': 1e7, 'partial_rotary_factor': .25})
                contract.validate_draft_config(cfg, require_marker=False)
                cfg['block_size'] = 6
                with self.assertRaises(ValueError):
                    contract.validate_draft_config(cfg, require_marker=False)

    def test_invalid_budget_rejected(self):
        for value in ('0', '8', '-1', 'abc'):
            with self.subTest(value=value), patch.dict(os.environ, {'Q36_SWEEP_K': value}):
                with self.assertRaises(ValueError):
                    importlib.reload(contract)


if __name__ == '__main__':
    unittest.main()
