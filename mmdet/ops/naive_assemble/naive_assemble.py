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
        for srcfile in ["naive_assemble_cuda.cc", "naive_assemble_cuda_kernel.cu"]
    ],
    extra_cflags=cxx_args,
    extra_cuda_cflags=nvcc_args,
    verbose=True,
)

class NaiveAssembleFunction(Function):

    @staticmethod
    def forward(ctx,
                cur_prev_aff,
                feat,
                k):
        # ctx.save_for_backward(cur_prev_aff, feat)
        ctx.k = k

        with torch.cuda.device_of(cur_prev_aff):
            output = feat.new()
            masked_cpa = feat.new()

            _ext.forward(cur_prev_aff, feat, output, k, masked_cpa)

        ctx.save_for_backward(cur_prev_aff, feat, masked_cpa)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        cur_prev_aff, feat, masked_cpa = ctx.saved_tensors

        with torch.cuda.device_of(cur_prev_aff):
            grad_cur_prev_aff = cur_prev_aff.new()
            grad_feat = feat.new()

            _ext.backward(
                cur_prev_aff, feat,
                grad_output, grad_cur_prev_aff, grad_feat,
                ctx.k, masked_cpa)

        return grad_cur_prev_aff, grad_feat, None


class NaiveAssemble(Module):
    def __init__(self,
                 k):
        super(NaiveAssemble, self).__init__()
        self.k = k

    def forward(self, cur_prev_aff, feat):

        result = NaiveAssembleFunction.apply(
            cur_prev_aff, feat, self.k)

        return result
