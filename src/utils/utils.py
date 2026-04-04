import os
import builtins as __builtin__
from typing import Any, Dict, Optional, Iterable
import numpy as np
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from collections import defaultdict, deque


def is_dist_avail_and_initialized() -> bool:
    """Checks if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Returns the number of processes in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Returns the rank of the current process in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Checks if the current process is the main one (rank 0)."""
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(cfg: Dict[str, Any]):
    """Initializes the distributed mode."""
    # Check if CPU mode is forced
    cpu_mode = cfg.get("cpu", False)
    
    if cpu_mode:
        print("CPU mode enabled - disabling distributed training")
        cfg["distributed"] = False
        cfg["rank"] = 0
        cfg["world_size"] = 1
        cfg["gpu"] = 0
        return
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg["rank"] = int(os.environ["RANK"])
        cfg["world_size"] = int(os.environ["WORLD_SIZE"])
        cfg["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg["rank"] = int(os.environ["SLURM_PROCID"])
        cfg["gpu"] = cfg["rank"] % torch.cuda.device_count()
        # SLURM should provide WORLD_SIZE, but provide fallback
        cfg["world_size"] = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NPROCS", "1")))
    else:
        print("Not using distributed mode")
        cfg["distributed"] = False
        cfg["rank"] = 0
        cfg["world_size"] = 1
        cfg["gpu"] = 0
        return

    cfg["distributed"] = True
    torch.cuda.set_device(cfg["gpu"])
    cfg["dist_backend"] = "nccl"
    print(f"| distributed init (rank {cfg['rank']}): env://", flush=True)
    dist.init_process_group(
        backend=cfg["dist_backend"],
        init_method="env://",
        world_size=cfg["world_size"],
        rank=cfg["rank"],
    )
    dist.barrier()
    setup_for_distributed(cfg["rank"] == 0)


def get_grad_norm_(parameters: Optional[Iterable[torch.nn.Parameter]],
                   norm_type: float = 2.0) -> torch.Tensor:
    if parameters is None:
        return torch.tensor(0.0)

    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return torch.tensor(0.0)

    device = grads[0].device
    norm_type = float(norm_type)

    grad_norms = [torch.norm(g.detach(), norm_type).to(device) for g in grads]
    total_norm = torch.norm(torch.stack(grad_norms), norm_type)

    return total_norm


class NativeScalerWithGradNormCount:
    """
    A gradient scaler for mixed-precision training that also tracks gradient norm.
    Ported from BEiT.
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        # self._scaler = torch.cuda.amp.GradScaler()
        self._scaler = GradScaler('cuda')

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: Optimizer,
        clip_grad: Optional[float] = None,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> float:
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                grad_norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            grad_norm = get_grad_norm_(parameters)
        if grad_norm is not None:
            return grad_norm.item()
        return 0.0

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the scaler's state dictionary."""
        self._scaler.load_state_dict(state_dict)


def patchify_img(img: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Rearranges an image into a sequence of patches.

    Args:
        img (torch.Tensor): A tensor of images with shape (B, C, H, W).
        patch_size (int, optional): The size of each square patch. Defaults to 16.

    Returns:
        torch.Tensor: A tensor of patches with shape (B, L, patch_size**2 * C).
    """
    p = patch_size
    B, C, H, W = img.shape
    x = img.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p * C)
    return x


def cosine_scheduler(
    base_value: float, final_value: float, epochs: int, niter_per_ep: int,
    warmup_epochs: int = 0, start_warmup_value: float = 1e-6
) -> np.ndarray:
    """Generates a cosine learning rate schedule with a linear warmup."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(warmup_iters, epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * \
               (1 + np.cos(np.pi * (iters - warmup_iters) / (len(iters))))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
