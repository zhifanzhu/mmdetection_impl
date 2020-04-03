#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <iostream>

#include "naive_assemble2_cuda_kernel.cuh"

int naive_assemble2_forward_cuda(
        at::Tensor& update, at::Tensor& input2, /*at::Tensor& rInput1,*/ at::Tensor& rInput2, at::Tensor& Aff,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2,
        int corr_type_multiply)
{

  int batchSize = input2.size(0);

  int nInputChannels = input2.size(1);
  int inputHeight = input2.size(2);
  int inputWidth = input2.size(3);

  int paddedInputHeight = inputHeight + 2 * pad_size;
  int paddedInputWidth = inputWidth + 2 * pad_size;

  update.resize_({batchSize,nInputChannels, inputHeight, inputWidth});
  rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  /* Aff.resize_({batchSize, nAffChannels, affHeight, affWidth}); */

  update.fill_(0);
  rInput2.fill_(0);
  /* Aff.fill_(0); */

  int success = naive_assemble2_forward_cuda_kernel(
    Aff,
    Aff.size(0), 
    Aff.size(1),
    Aff.size(2),
    Aff.size(3),
    Aff.stride(0),
    Aff.stride(1),
    Aff.stride(2),
    Aff.stride(3),
    update,
    input2.size(1),
    input2.size(2),
    input2.size(3),
    input2.stride(0),
    input2.stride(1),
    input2.stride(2),
    input2.stride(3),
    input2,
    input2.size(1),
    input2.stride(0),
    input2.stride(1),
    input2.stride(2),
    input2.stride(3),
    /* rInput1, */
    rInput2,
    pad_size,     
    kernel_size,
    max_displacement,
    stride1,
    stride2,
    corr_type_multiply,
	at::cuda::getCurrentCUDAStream()
    //at::globalContext().getCurrentCUDAStream()
  );

  //check for errors
  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;

}

int naive_assemble2_backward_cuda(
        at::Tensor& gradUpdate, at::Tensor& input2, /*at::Tensor& rInput1,*/ at::Tensor& rInput2, at::Tensor& Aff, 
        at::Tensor& gradAff, at::Tensor& gradInput2,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2,
        int corr_type_multiply)
{

  int batchSize = input2.size(0);
  int nInputChannels = input2.size(1);
  int paddedInputHeight = input2.size(2)+ 2 * pad_size;
  int paddedInputWidth = input2.size(3)+ 2 * pad_size;

  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;

  int nAffChannels = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1);

  int affHeight = ceil(static_cast<float>(paddedInputHeight - 2 * border_radius) / static_cast<float>(stride1));
  int affWidth = ceil(static_cast<float>(paddedInputWidth - 2 * border_radius) / static_cast<float>(stride1));

  int height = input2.size(2);
  int width = input2.size(3);

  /* rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels}); */
  rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  gradAff.resize_({batchSize, nAffChannels, affHeight, affWidth});
  gradInput2.resize_({batchSize, nInputChannels, height, width});

  /* rInput1.fill_(0); */
  rInput2.fill_(0);
  gradAff.fill_(0);
  gradInput2.fill_(0);

  int success = naive_assemble2_backward_cuda_kernel(Aff,
                                                Aff.size(0),
                                                Aff.size(1),
                                                Aff.size(2),
                                                Aff.size(3),
                                                Aff.stride(0),
                                                Aff.stride(1),
                                                Aff.stride(2),
                                                Aff.stride(3),
                                                gradAff,
                                                gradUpdate.size(1),
                                                gradUpdate.size(2),
                                                gradUpdate.size(3),
                                                gradUpdate.stride(0),
                                                gradUpdate.stride(1),
                                                gradUpdate.stride(2),
                                                gradUpdate.stride(3),
                                                input2,  
                                                input2.stride(0),
                                                input2.stride(1),
                                                input2.stride(2),
                                                input2.stride(3),
                                                gradUpdate,
                                                gradUpdate.stride(0),
                                                gradUpdate.stride(1),
                                                gradUpdate.stride(2),
                                                gradUpdate.stride(3),
                                                gradInput2,
                                                gradInput2.size(1),
                                                gradInput2.stride(0),
                                                gradInput2.stride(1),
                                                gradInput2.stride(2),
                                                gradInput2.stride(3),
                                                /* rInput1, */
                                                rInput2,
                                                pad_size,
                                                kernel_size,
                                                max_displacement,
                                                stride1, 
                                                stride2,
                                                corr_type_multiply,
												at::cuda::getCurrentCUDAStream()
                                                //at::globalContext().getCurrentCUDAStream()
                                               );

  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &naive_assemble2_forward_cuda, "naive_assemble2 forward (CUDA)");
  m.def("backward", &naive_assemble2_backward_cuda, "naive_assemble2 backward (CUDA)");
}

