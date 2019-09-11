"""pointwise correlation function and module."""

from pathlib import Path
from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from torch.utils import cpp_extension


# JIT compilation
this_dir = Path(__file__).resolve().parent
_ext = cpp_extension.load(
    name="ext",
    sources=[
        Path(this_dir, srcfile)
        for srcfile in ["pointwise_correlation.cpp", "pointwise_correlation_cuda.cu"]
    ],
    # extra_include_paths=[str(this_dir.parent)],
    extra_cuda_cflags=["-arch=sm_60"],
)


class PointwiseCorrelationFunction(Function):
    """pointwise local correlations.
    see https://arxiv.org/abs/1710.03958"""

    @staticmethod
    def forward(ctx, FM0: Tensor, FM1: Tensor, d_max: int, stride: int) -> Tensor:
        """pointwise correlations between FM0 and FM1.

        Args:
            FM0: (|B|, C, H, W) feature map at time t.
            FM1: (|B|, C, H, W) feature mat at time t+tau.
            d_max: maximum displacement.
            stride: stride between displacements.

        Returns:
            out: (|B|, H, W, (2d+1), (2d+1)) pointwise correlations.
        """
        ctx.save_for_backward(FM0, FM1)
        ctx.d_max = d_max
        ctx.stride = stride

        out = _ext.pointwise_correlation_forward(FM0, FM1, d_max, stride)

        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """given derivatives wrt out, compute derivatives wrt FM0 and FM1.

        Args:
            grad_out: (|B|, H, W, (2d+1), (2d+1)) gradients wrt pointwise
                correlations.

        Returns:
            grad_FM0: (|B|, C, H, W) gradients wrt FM0.
            grad_FM1: (|B|, C, H, W) gradients wrt FM1.
        """
        grad_out = grad_out.contiguous()
        FM0, FM1 = ctx.saved_tensors
        grad_FM0, grad_FM1 = _ext.pointwise_correlation_backward(
            grad_out, FM0, FM1, ctx.d_max, ctx.stride
        )
        return grad_FM0, grad_FM1, None, None


class PointwiseCorrelation(Module):
    """pointwise local correlations.
    see https://arxiv.org/abs/1710.03958

    Args:
        d_max: maximum displacement.
        stride: displacement stride.
    """

    def __init__(self, d_max: int, stride: int) -> None:
        super().__init__()
        self.d_max = d_max
        self.stride = stride

    def forward(self, FM0: Tensor, FM1: Tensor) -> Tensor:
        """given feature maps from two different time steps, compute pointwise
        correlations.

        Args:
            FM0: (|B|, C, H, W) feature map at time t.
            FM1: (|B|, C, H, W) feature map at time t+tau.

        Returns:
            out: (|B|, H, W, (2d+1), (2d+1)) pointwise correlations.
        """
        return PointwiseCorrelationFunction.apply(FM0, FM1, self.d_max, self.stride)
