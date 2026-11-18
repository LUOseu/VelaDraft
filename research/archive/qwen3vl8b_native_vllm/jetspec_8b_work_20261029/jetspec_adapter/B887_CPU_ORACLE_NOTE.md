# Frozen B887 CPU fixture index erratum

This note concerns a CPU test stand-in, discovered while mounting the **local shared-block JetSpec reference**, not the upstream parallel-tree runtime. No frozen production/test/receipt file was changed.

`bounded_lookahead/test_bounded_adapter.py:402` fills the mocked expansion output with `arange(B-1)`, and line 441 checks `arange(S)`. Those are rows `0..S-1`. This limits the older CPU receipt's **sample-index value coverage**: the fixture did not establish that the anchor row was excluded.

The installed official `v1/spec_decode/utils.py:550–562` requires `query_off > 0` and writes `query_out`; for one request the real proposal indices are `1..S`. Frozen B887 calls the official expansion through `original_set` at `official_spatial_bounded_adapter.py:264` and returns its result unchanged at line 273. The production path therefore already uses the official convention. This is a test-fixture erratum, not a production numerical change.

The new JetSpec boundary test fills `torch.arange(1,B)`, asserts the exact `1..S` indices and explicitly excludes zero for S1/S3/S7 and rejection counts 0..7. It remains a CPU stand-in at the kernel boundary: no Triton/GPU kernel was run. Existing observed F2 GPU token IDs and capture records remain actual evidence.

Exact SHA256 values, source lines and the new test name are recorded in [B887_CPU_ORACLE_NOTE.json](B887_CPU_ORACLE_NOTE.json). In particular, the frozen adapter remains `b887520946c0e72b0ce59df9c10bcaf909e6a24e8dd521cadced95bdcc9c84da` and the installed official expansion source is `129fbe9c32b6567d513463442a5c11ae3365bd38accbbd5547f3af5dbd48624d`.
