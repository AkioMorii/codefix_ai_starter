import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    print('=== nvidia-smi ===')
    proc = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
    print(proc.stdout if proc.returncode == 0 else proc.stderr)
    info = {
        'data_utc': datetime.now(timezone.utc).isoformat(),
        'torch': torch.__version__,
        'cuda_runtime_pytorch': torch.version.cuda,
        'cuda_disponivel': torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        print('ERRO: PyTorch não detectou GPU CUDA. Confira o driver e o PyTorch instalado.')
        raise SystemExit(1)
    info['gpu'] = torch.cuda.get_device_name(0)
    info['capability'] = torch.cuda.get_device_capability(0)
    info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    # Operação REAL executada na GPU:
    x = torch.tensor([2.0, 3.0], device='cuda:0')
    info['teste_tensor_gpu'] = (x * 2).tolist()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    Path('resultados').mkdir(exist_ok=True)
    Path('resultados/gpu.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
