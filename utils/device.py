import os, torch, platform, warnings
warnings.filterwarnings('default')
CPU_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(CPU_THREADS)

def select_torch_device():
    forced = os.environ.get('DIAFED_DEVICE', 'auto').strip().lower()
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if forced not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('DIAFED_DEVICE must be auto, cpu, mps, or cuda')
    if forced == 'mps' and (not mps):
        raise RuntimeError('MPS requested, but torch.backends.mps.is_available() is false.')
    if forced == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested, but not available.')
    if forced != 'auto':
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if mps:
        return torch.device('mps')
    return torch.device('cpu')

def synchronize_device(target):
    target = torch.device(target)
    if target.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()
    elif target.type == 'cuda':
        torch.cuda.synchronize(target)

def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
import random
import numpy as np
