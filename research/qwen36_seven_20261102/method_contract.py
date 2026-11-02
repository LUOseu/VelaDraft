"""Strict value checks that accept msgpack's tuple-to-list wire representation."""
def validate_conv_blocks(blocks, k):
    assert isinstance(blocks, (list, tuple)) and len(blocks) == 3
    assert all(isinstance(row, (list, tuple)) and len(row) == 2 for row in blocks)
    assert [list(row) for row in blocks] == [[k+1, k+1]]*3, blocks

