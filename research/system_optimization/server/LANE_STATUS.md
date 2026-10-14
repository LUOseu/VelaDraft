# RSI system-optimization lane — status (2026-10-14 ~13:50, Asia/Shanghai)

Owner: Codex session executing `VelaDraft_System_RSI_Goal_20261014.md`
(branch `research/online-systems` @ e15256d). Repo copy: `/home/lyh/VelaDraft-systems`.

Queued job: screen `syslane` -> `watch_gpu.sh` launches `run_s1.sh` on the FIRST
truly idle RTX A6000 (0 compute procs, >40 GiB free, util<=5%, confirmed 60 s
apart). It never touches a busy device; GPU3 (other user) and any running
screens are respected. Horizon: 20 h from 13:40.

Chain: S1a smoke unified-boundary re-measurement -> S1b group sweep ->
S2 256-slice cache A/B (off/on x3, interleaved) -> S3 canonical 2K cache A/B.
Budget: <=16 GPU-h (this lane share of the route 32); spent so far: 0.0.

Logs: `/home/lyh/vlm-systems/logs/` (`waiter.log`, `s1a/s1b/s2/s3_progress.log`,
per-leg `.log` + `.time` files). Harvest: `python3 /home/lyh/vlm-systems/analyze_ab.py`.

Stop: `touch /home/lyh/vlm-systems/STOP_WAITER` (blocks future launches) or
`screen -S syslane -X quit`. Neither touches already-started runs.

Disk: scripts prune their own `ckpt_last.pt`; S3 prunes `ckpt_init.pt` after
hashing. `/home` is 100% full — if free space drops below 12 GiB, S3 skips
itself rather than risk the volume.
