#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int fast_assemble_forward_cuda_kernel(at::Tensor& output,
        at::Tensor& cur_prev_aff,
        int aff_c,

        at::Tensor& input2,
        int ib,
        int ic,
        int ih,
        int iw,

        int k,
        at::Tensor& masked_cpa,
        cudaStream_t stream);


int fast_assemble_backward_cuda_kernel(
        at::Tensor& gradOutput,
        int gob,
        int goc,
        int goh,
        int gow,

        at::Tensor& cur_prev_aff,
        int aff_c,
        at::Tensor& feat,
        at::Tensor& gradAff,
        at::Tensor& gradFeat,
        int k,
        at::Tensor& masked_cpa,
        cudaStream_t stream);
