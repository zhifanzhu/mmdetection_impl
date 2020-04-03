#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int naive_assemble2_forward_cuda_kernel(at::Tensor& Aff,
    int ab,
    int ac,
    int ah,
    int aw,
    int asb,
    int asc,
    int ash,
    int asw,

    at::Tensor& update,
    int uc,
    int uh,
    int uw,
    int usb,
    int usc,
    int ush,
    int usw,

    at::Tensor& input2,
    int gc,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    /* at::Tensor& rInput1, */
    at::Tensor& rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);


int naive_assemble2_backward_cuda_kernel(   
    at::Tensor& Aff,
    int gab,
    int gac,
    int gah,
    int gaw,
    int gasb,
    int gasc,
    int gash,
    int gasw,

    at::Tensor& gradAff,
    int uc,
    int uh,
    int uw,
    int usb,
    int usc,
    int ush,
    int usw,

    at::Tensor& input2,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    at::Tensor& gradUpdate, 
    int gusb,
    int gusc,
    int gush,
    int gusw,

    at::Tensor& gradInput2,
    int ggc,
    int ggsb,
    int ggsc,
    int ggsh,
    int ggsw,

    at::Tensor& rGradUpdate,
    at::Tensor& rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);
