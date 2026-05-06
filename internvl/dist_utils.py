import os
import socket
import subprocess
from datetime import timedelta

import deepspeed
import torch
import torch.multiprocessing as mp
from torch import distributed as dist

timeout = timedelta(minutes=60)


def _find_free_port():
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port):
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def _get_accelerator_and_count():
    """Detect available accelerator (cuda/npu/cpu) and device count."""
    try:
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            return 'cuda', torch.cuda.device_count()
    except Exception:
        pass
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return 'npu', torch.npu.device_count()
    except Exception:
        pass
    return 'cpu', 0


def _resolve_backend(preferred_backend: str | None) -> str:
    """Resolve backend based on accelerator availability.

    - Prefer user-specified backend if compatible; otherwise, fall back.
    - CUDA -> nccl, Ascend NPU -> hccl, CPU -> gloo.
    """
    accel, count = _get_accelerator_and_count()
    # Auto selection
    if preferred_backend in (None, 'auto'):
        if accel == 'cuda' and count > 0:
            return 'nccl'
        if accel == 'npu' and count > 0:
            return 'hccl'
        return 'gloo'

    # User specified but incompatible -> pick a sensible fallback
    if preferred_backend == 'nccl' and not (accel == 'cuda' and count > 0):
        if accel == 'npu' and count > 0:
            print('[dist] CUDA not available, switching backend nccl -> hccl for Ascend NPU.')
            return 'hccl'
        print('[dist] CUDA not available, switching backend nccl -> gloo.')
        return 'gloo'
    if preferred_backend == 'hccl' and not (accel == 'npu' and count > 0):
        if accel == 'cuda' and count > 0:
            print('[dist] Ascend NPU not available, switching backend hccl -> nccl.')
            return 'nccl'
        print('[dist] Ascend NPU not available, switching backend hccl -> gloo.')
        return 'gloo'
    return preferred_backend or 'gloo'


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    backend = _resolve_backend(backend)
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # Prefer LOCAL_RANK if provided, else use RANK
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    accel, count = _get_accelerator_and_count()
    if accel == 'cuda' and count > 0:
        torch.cuda.set_device(rank % count)
    elif accel == 'npu' and count > 0:
        # Ascend NPU device selection
        torch.npu.set_device(rank % count)
    else:
        # CPU fallback, no device setting required
        pass
    # Initialize distributed using DeepSpeed for consistency
    deepspeed.init_distributed(dist_backend=backend)


def _init_dist_mpi(backend, **kwargs):
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    accel, count = _get_accelerator_and_count()
    if accel == 'cuda' and count > 0:
        torch.cuda.set_device(local_rank)
    elif accel == 'npu' and count > 0:
        torch.npu.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    accel, count = _get_accelerator_and_count()
    num_devices = max(1, count)
    if accel == 'cuda' and count > 0:
        torch.cuda.set_device(proc_id % num_devices)
    elif accel == 'npu' and count > 0:
        torch.npu.set_device(proc_id % num_devices)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_devices)
    os.environ['RANK'] = str(proc_id)
    # dist.init_process_group(backend=backend, timeout=timeout)
    deepspeed.init_distributed(dist_backend=backend)
