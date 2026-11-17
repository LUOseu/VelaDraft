"""CPU checks for the anonymous release's local-checkpoint contract."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from research.qwen36_a100_seven_eval.prepare import fetch_checkpoint


class LocalCheckpointTests(unittest.TestCase):
    def test_matching_local_bytes_are_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            entry = {"size": checkpoint.stat().st_size,
                     "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            self.assertEqual(fetch_checkpoint("dflash", entry, [checkpoint], root / "unused", root / "hashes.json"), checkpoint)
            self.assertFalse((root / "unused").exists())

    def test_missing_or_changed_weights_fail_without_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"wrong")
            entry = {"size": 5, "sha256": "0" * 64}
            with self.assertRaisesRegex(FileNotFoundError, "Q36_LOCAL_CHECKPOINTS"):
                fetch_checkpoint("dflash", entry, [checkpoint], root / "unused", root / "hashes.json")
            self.assertFalse((root / "unused").exists())


if __name__ == "__main__":
    unittest.main()
