import torch 
from typing import Optional

@torch.jit.script_if_tracing
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """Broadcast util, from torch_scatter"""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

@torch.jit.script_if_tracing
def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter sum function, from torch_scatter module (https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py)"""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

@torch.jit.script_if_tracing
def scatter_mean(src: torch.Tensor,
                 index: torch.Tensor,
                 dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter mean function, from torch_scatter module (https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py)"""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out