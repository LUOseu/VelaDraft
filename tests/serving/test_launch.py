from __future__ import annotations

from vlm_dflash.serving.launch import build_server_command, load_config


def test_ar_and_ngram_configs_build_commands():
    ar = load_config("configs/platform_extensions/serving/ar.json")
    ngram = load_config("configs/platform_extensions/serving/ngram.json")
    ar_cmd = build_server_command(ar, port=8100, gpu_index=0)
    ngram_cmd = build_server_command(ngram, port=8101, gpu_index=0)
    assert "--speculative-config" not in ar_cmd
    assert "--speculative-config" in ngram_cmd
    spec = ngram_cmd[ngram_cmd.index("--speculative-config") + 1]
    assert '"method":"ngram"' in spec
    assert "--no-enable-prefix-caching" in ar_cmd
    assert ar_cmd[ar_cmd.index("--port") + 1] == "8100"
    assert ngram_cmd[ngram_cmd.index("--port") + 1] == "8101"
