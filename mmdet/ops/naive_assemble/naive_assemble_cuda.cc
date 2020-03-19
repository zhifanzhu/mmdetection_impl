#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <iostream>

#include "naive_assemble_cuda_kernel.cuh"

int naive_assemble_forward_cuda(
        at::Tensor& cur_prev_aff,  // [B, D^2, H, W]
        at::Tensor& feat,          // [B, C, H, W]
        at::Tensor& output,
        int k,
        at::Tensor& masked_cpa)
{
  int batchSize = feat.size(0);
  int nInputChannels = feat.size(1);
  int inputHeight = feat.size(2);
  int inputWidth = feat.size(3);

  output.resize_({batchSize, nInputChannels, inputHeight, inputWidth});
  masked_cpa.resize_({batchSize, cur_prev_aff.size(1), inputHeight, inputWidth});

  int success = naive_assemble_forward_cuda_kernel(
    output,
    cur_prev_aff,
    cur_prev_aff.size(1),
    feat,
    batchSize,
    nInputChannels,
    inputHeight,
    inputWidth,
    k,     
    masked_cpa,
	at::cuda::getCurrentCUDAStream()
  );

  //check for errors
  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;

}

int naive_assemble_backward_cuda(
        at::Tensor& cur_prev_aff, 
        at::Tensor& feat, 
        at::Tensor& gradOutput, 
        at::Tensor& gradAff,
        at::Tensor& gradFeat,
        int k,
        at::Tensor& masked_cpa)
{
  int batchSize = feat.size(0);
  int nInputChannels = feat.size(1);
  int inputHeight = feat.size(2);
  int inputWidth = feat.size(3);
  int D_sqr = cur_prev_aff.size(1);
  gradAff.resize_({batchSize, D_sqr, inputHeight, inputWidth});
  gradFeat.resize_({batchSize, nInputChannels, inputHeight, inputWidth});

  gradAff.fill_(0);
  gradFeat.fill_(0);

  int success = naive_assemble_backward_cuda_kernel(gradOutput,
                                                gradOutput.size(0),
                                                gradOutput.size(1),
                                                gradOutput.size(2),
                                                gradOutput.size(3),
                                                cur_prev_aff,
                                                cur_prev_aff.size(1),
                                                feat,  
                                                gradAff,
                                                gradFeat,
                                                k,
                                                masked_cpa,
												at::cuda::getCurrentCUDAStream()
                                                //at::globalContext().getCurrentCUDAStream()
                                               );

  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &naive_assemble_forward_cuda, "NaiveAssemble forward (CUDA)");
  m.def("backward", &naive_assemble_backward_cuda, "NaiveAssemble backward (CUDA)");
}

