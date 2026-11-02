import unittest
import msgspec
from research.qwen36_seven_20261102.method_contract import validate_conv_blocks


class ContractTests(unittest.TestCase):
    def test_real_msgpack_roundtrip_preserves_accepted_values(self):
        original=[(4,4)]*3
        wire=msgspec.msgpack.decode(msgspec.msgpack.encode(original))
        self.assertNotEqual(wire,original)
        validate_conv_blocks(original,3)
        validate_conv_blocks(wire,3)

    def test_wrong_runtime_width_or_layer_count_still_fails(self):
        for bad in ([[8,8]]*3, [[4,4]]*2, [[4,4],[4,8],[4,4]], [4,4,4]):
            with self.assertRaises(AssertionError):validate_conv_blocks(bad,3)


if __name__=='__main__':unittest.main()
