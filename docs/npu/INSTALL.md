# Reproduce the 910B environment

The hardware bring-up uses aarch64 Linux, two Ascend 910B3 devices with 64 GiB
each, driver 26.0.rc1, CANN toolkit **and 910b ops 9.0.0**, Python 3.10,
PyTorch 2.10.0 and `torch-npu==2.10.0`. The original CANN 8.0.RC2 image could not
satisfy this project's PyTorch requirements.

The exact wheel constraints ship in
`vlm_dflash/resources/constraints-npu.txt`. Other hardware/version combinations
need their own validation. Check the
[upstream compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
before selecting a different combination.

## Isolated installation

Use a dedicated environment and installation prefix. The CANN installer requires
755 permissions on the installation directory and its parents: use a suitable
prefix such as `/opt/veladraft-ascend` rather than changing `/root` permissions.
The toolkit installer also writes installation records under `/etc/Ascend`;
it backs up earlier records. The commands below do not install a host driver.

```bash
python3.10 -m venv /opt/veladraft-env
source /opt/veladraft-env/bin/activate
python -m pip install --upgrade pip 'setuptools>=68' wheel

mkdir -p /tmp/veladraft-cann
cd /tmp/veladraft-cann
curl -fLO 'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run'
curl -fLO 'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-aarch64.run'
bash Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install --install-path=/opt/veladraft-ascend
bash Ascend-cann-910b-ops_9.0.0_linux-aarch64.run --install --install-path=/opt/veladraft-ascend
source /opt/veladraft-ascend/cann/set_env.sh

cd /path/to/VelaDraft
pip install -c vlm_dflash/resources/constraints-npu.txt \
  torch torchvision torch-npu einops scipy decorator attrs psutil
pip install -c vlm_dflash/resources/constraints-npu.txt -e '.[dev]'
```

The CANN environment must be sourced in every new SSH shell. If `npu-smi`
cannot find `libc_sec.so` in a container whose driver is mounted at the standard
path, add the actual driver libraries (not toolkit stub libraries):

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
npu-smi info
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())'
vlm-dflash doctor --device npu:0 --json
```

Device indices inside PyTorch can differ from physical `npu-smi` indices.
Validate each visible logical device before assigning jobs.

## First checks

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H0,H1,H2 --work-dir runs/npu/basic
python scripts/npu/run_npu_smoke.py --device npu:1 --phases H0,H1 --work-dir runs/npu/second-device
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H3 --h3-steps 200 --work-dir runs/npu/continuity
```

These are synthetic-feature checks. Follow
[HARDWARE_BRINGUP.md](HARDWARE_BRINGUP.md) for real target training and
[VALIDATION.md](VALIDATION.md) for evidence boundaries.
