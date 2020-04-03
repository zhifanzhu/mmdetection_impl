from pathlib import Path
from torch.utils import cpp_extension

import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
from torch.autograd.function import once_differentiable


# JIT compilation
cxx_args = ['-std=c++11']
nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]
this_dir = Path(__file__).resolve().parent
_ext = cpp_extension.load(
    name="ext",
    sources=[
        Path(this_dir, srcfile)
        for srcfile in ["naive_assemble2_cuda.cc", "naive_assemble2_cuda_kernel.cu"]
    ],
    extra_cflags=cxx_args,
    extra_cuda_cflags=nvcc_args,
)


class NaiveAssemble2Function(Function):

    @staticmethod
    def forward(ctx,
                Aff,
                input2,
                pad_size,
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                corr_multiply):
        ctx.save_for_backward(Aff, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(Aff):
            rbot2 = input2.new()
            update = input2.new()

            _ext.forward(update, input2, rbot2, Aff,
                         pad_size, kernel_size, max_displacement,
                         stride1, stride2, corr_multiply)

        return update

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_update):
        grad_update = grad_update.contiguous()
        Aff, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input2):
            rbot2 = input2.new()

            grad_input2 = input2.new()
            grad_aff = input2.new()

            _ext.backward(
                grad_update, input2, rbot2, Aff,
                grad_aff, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement,
                ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_aff, grad_input2, None, None, None, None, None, None


class NaiveAssemble2(Module):
    def __init__(self,
                 pad_size=0,
                 kernel_size=0,
                 max_displacement=0,
                 stride1=1,
                 stride2=2,
                 corr_multiply=1):
        super(NaiveAssemble2, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, aff, input2):

        result = NaiveAssemble2Function.apply(
            aff, input2, self.pad_size, self.kernel_size, self.max_displacement,
            self.stride1, self.stride2, self.corr_multiply)

        return result
