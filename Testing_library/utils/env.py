import os
import subprocess
from torch import distributed as dist
from torch import nn
import uuid
from loguru import logger
import random
import torch
import cv2
import numpy as np 
__all__ = ["get_num_devices","synchronize","configure_nccl",'configure_module','get_world_size','get_rank','get_local_rank','is_main_process','is_parallel','configure_omp','reduce_mean']

_LOCAL_PROCESS_GROUP=None

def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if _LOCAL_PROCESS_GROUP is None:
        return get_rank()

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group, i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0



def configure_nccl():
    """Configure multi-machine environment variables of NCCL."""
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def configure_omp(num_threads=1):
    """
    If OMP_NUM_THREADS is not configured and world_size is greater than 1,
    Configure OMP_NUM_THREADS environment variables of NCCL to `num_thread`.

    Args:
        num_threads (int): value of `OMP_NUM_THREADS` to set.
    """
    # We set OMP_NUM_THREADS=1 by default, which achieves the best speed on our machines
    # feel free to change it for better performance.
    if "OMP_NUM_THREADS" not in os.environ and get_world_size() > 1:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        if is_main_process():
            logger.info(
                "\n***************************************************************\n"
                "We set `OMP_NUM_THREADS` for each process to {} to speed up.\n"
                "please further tune the variable for optimal performance.\n"
                "***************************************************************".format(
                    os.environ["OMP_NUM_THREADS"]
                )
            )

def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)



def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    """
    # system setting
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass

    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might rasie exceptions.
        pass

