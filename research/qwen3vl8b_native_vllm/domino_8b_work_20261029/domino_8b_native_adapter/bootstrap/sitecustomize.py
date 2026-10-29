"""One explicit Domino bootstrap. Initial logic receipts cannot launch hardware."""
import os,sys
from pathlib import Path
if 'FAIR_DOMINO_ADAPTER' in os.environ:
    try:
        if os.environ['FAIR_DOMINO_ADAPTER']!='1':raise RuntimeError('Domino flag must be1')
        if any(name in sys.modules for name in ('torch','vllm')):raise RuntimeError('Bootstrap must precede Torch/vLLM')
        if any(name.startswith('VLLM_OFFICIAL_') for name in os.environ):raise RuntimeError('Stacked official adapters are forbidden')
        if 'FAIR_DOMINO_CPU_PROOF' in os.environ:raise RuntimeError('CPU proof flags are forbidden in production bootstrap')
        if os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK')!='1' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':raise RuntimeError('NVML1 and PCI_BUS_ID required before Python')
        root=Path(__file__).resolve().parents[1]
        if 'contracts' in sys.modules and Path(sys.modules['contracts'].__file__).resolve()!=root/'contracts.py':raise RuntimeError('Foreign contracts module')
        sys.path.insert(0,str(root))
        import contracts
        contracts.check_full_header(os.environ.get('DOMINO_FULL_PREFLIGHT',str(root/'FULL_PREFLIGHT.json')),os.environ.get('DOMINO_APPROVED_PREFLIGHT_SHA256'),os.environ.get('DOMINO_RUNTIME_APPROVAL'),os.environ.get('DOMINO_APPROVED_RUNTIME_SHA256'))
        from install import install
        install()
    except Exception as error:
        sys.stderr.write('DOMINO_BOOTSTRAP_REJECTED '+str(error)+'\n');sys.stderr.flush();os._exit(86)
