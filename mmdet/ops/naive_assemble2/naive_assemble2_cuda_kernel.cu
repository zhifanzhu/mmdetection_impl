#include <stdio.h>

#include "naive_assemble2_cuda_kernel.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32
#define FULL_MASK 0xffffffff

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

using at::Half;

template<typename scalar_t>
__forceinline__ __device__ scalar_t warpReduceSum(scalar_t val) {
        for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_down_sync(FULL_MASK, val, offset);
        return val;
}

template<typename scalar_t>
__forceinline__ __device__ scalar_t blockReduceSum(scalar_t val) {

        static __shared__ scalar_t shared[32];
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        val = warpReduceSum(val);

        if (lane == 0)
                shared[wid] = val;

        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

        if (wid == 0)
                val = warpReduceSum(val);

        return val;
}

template <typename scalar_t>
__global__ void channels_first(const scalar_t* __restrict__ input, scalar_t* rinput, int channels, int height, int width, int pad_size)
{

    // n (batch size), c (num of channels), y (height), x (width)
    int n = blockIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.z;

    int ch_off = threadIdx.x;
    scalar_t value;

    int dimcyx = channels * height * width;
    int dimyx = height * width;

    int p_dimx = (width + 2 * pad_size);
    int p_dimy = (height + 2 * pad_size);
    int p_dimyxc = channels * p_dimy * p_dimx;
    int p_dimxc = p_dimx * channels;

    for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
      value = input[n * dimcyx + c * dimyx + y * width + x];
      rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
    }
}

// cuda kernel for assemble2 forward
template <typename scalar_t>
__global__ void naive_assemble2_forward(int item, scalar_t* update, int nInputChannels, int inputHeight, int inputWidth, 
        const scalar_t* __restrict__ Aff, int nAffChannels, int affHeight, int affWidth, 
        const scalar_t* __restrict__ rInput2, 
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2)
{
    // n (batch size), c (num of channels), y (height), x (width)
    int n = item; 
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;
    int tch_off = threadIdx.x;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int xmin = (x - kernel_rad - max_displacement) / stride1;
    int ymin = (y - kernel_rad - max_displacement) / stride1;

    int xmax = (x + kernel_rad - max_displacement) / stride1;
    int ymax = (y + kernel_rad - max_displacement) / stride1;

    if (xmax < 0 || ymax < 0 || xmin >= affWidth || ymin >= affHeight) {
        // assumes `update` is pre-allocated and zero filled
      return;
    }

    if (xmin > xmax || ymin > ymax) {
        // assumes `update` is pre-allocated and zero filled
        return;
    }

    xmin = max(0,xmin);
    xmax = min(affWidth-1,xmax);

    ymin = max(0,ymin);
    ymax = min(affHeight-1,ymax);

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nAffChannels * affHeight * affWidth;
    int tdimyx = affHeight * affWidth;
    int tdimx = affWidth;

    int odimcyx = nInputChannels * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    __shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nAffChannels; tc += THREADS_PER_BLOCK) {

      int i2 = (tc % displacement_size - displacement_rad) * stride2;
      int j2 = (tc / displacement_size - displacement_rad) * stride2;

      int indx2 = n * pdimyxc + (y + j2)* pdimxc + (x + i2) * pdimc + c;
      
      scalar_t val2 = rInput2[indx2];

      for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
          int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
          prod_sum[tch_off] += Aff[tindx] * val2;
        }
      }
    }
    __syncthreads();

    if(tch_off == 0) {
      scalar_t reduce_sum = 0;
      for(int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
          reduce_sum += prod_sum[idx];
      }
      const int indx1 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
      update[indx1] = reduce_sum;
    }

}

// cuda kernel for assemble2 backward for affinity
template<typename scalar_t>
__global__ void naive_assemble2_backward_aff(scalar_t* __restrict__ gradAff, const int nAffChannels,
        const int affHeight, const int affWidth, const scalar_t* __restrict__ gradUpdate,
        const int nInputChannels, const int inputHeight, const int inputWidth,
        const scalar_t* __restrict__ rInput2, const int pad_size, const int kernel_size,
        const int max_displacement, const int stride1, const int stride2) 
{

        int32_t pInputWidth = inputWidth + 2 * pad_size;
        int32_t pInputHeight = inputHeight + 2 * pad_size;

        int32_t kernel_rad = (kernel_size - 1) / 2;

        int32_t displacement_rad = max_displacement / stride2;

        int32_t displacement_size = 2 * displacement_rad + 1;

        int32_t n = blockIdx.x;
        int32_t y1 = blockIdx.y * stride1 + max_displacement;
        int32_t x1 = blockIdx.z * stride1 + max_displacement;
        int32_t c = threadIdx.x;

        int32_t pdimyxc = pInputHeight * pInputWidth * nInputChannels;

        int32_t pdimxc = pInputWidth * nInputChannels;

        int32_t pdimc = nInputChannels;

        int32_t tdimcyx = nAffChannels * affHeight * affWidth;
        int32_t tdimyx = affHeight * affWidth;
        int32_t tdimx = affWidth;

        // element-wise product along channel axis
        for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
                for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
                        int x2 = x1 + ti * stride2;
                        int y2 = y1 + tj * stride2;

                        float acc0 = 0.0f;

                        for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                                        // THREADS_PER_BLOCK
                                        #pragma unroll
                                        for (int ch = c; ch < pdimc; ch += blockDim.x) {

                                                int indx1 = n * pdimyxc + (y1 + j) * pdimxc
                                                                + (x1 + i) * pdimc + ch;
                                                int indx2 = n * pdimyxc + (y2 + j) * pdimxc
                                                                + (x2 + i) * pdimc + ch;
                                                acc0 += static_cast<float>(gradUpdate[indx1] * rInput2[indx2]);
                                        }
                                }
                        }

                        if (blockDim.x == warpSize) {
                            __syncwarp();
                            acc0 = warpReduceSum(acc0);
                        } else {
                            __syncthreads();
                            acc0 = blockReduceSum(acc0);
                        }

                        if (threadIdx.x == 0) {

                                int tc = (tj + displacement_rad) * displacement_size
                                                + (ti + displacement_rad);
                                const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx
                                                + blockIdx.z;
                                gradAff[tindx] = static_cast<scalar_t>(acc0);
                        }
            }
        }
}


template <typename scalar_t>
__global__ void naive_assemble2_backward_input2(int item, scalar_t*  gradInput2, int nInputChannels, int inputHeight, int inputWidth,
        const scalar_t* __restrict__ Aff, int nAffChannels, int affHeight, int affWidth,
        const scalar_t* __restrict__ gradUpdate,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2)
{
    // n (batch size), c (num of channels), y (height), x (width)

    int n = item;
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;

    int tch_off = threadIdx.x;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nAffChannels * affHeight * affWidth;
    int tdimyx = affHeight * affWidth;
    int tdimx = affWidth;

    int odimcyx = nInputChannels * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    __shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nAffChannels; tc += THREADS_PER_BLOCK) {
      int i2 = (tc % displacement_size - displacement_rad) * stride2;
      int j2 = (tc / displacement_size - displacement_rad) * stride2;

      int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
      int ymin = (y - kernel_rad - max_displacement - j2) / stride1;

      int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
      int ymax = (y + kernel_rad - max_displacement - j2) / stride1;

      if (xmax < 0 || ymax < 0 || xmin >= affWidth || ymin >= affHeight) {
          // assumes gradInput2 is pre-allocated and zero filled
        continue;
      }

      if (xmin > xmax || ymin > ymax) {
          // assumes gradInput2 is pre-allocated and zero filled
          continue;
      }

      xmin = max(0,xmin);
      xmax = min(affWidth-1,xmax);

      ymin = max(0,ymin);
      ymax = min(affHeight-1,ymax);
      
      int indx1 = n * pdimyxc + (y - j2)* pdimxc + (x - i2) * pdimc + c;
      scalar_t val1 = gradUpdate[indx1];

      for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
          int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
          prod_sum[tch_off] += Aff[tindx] * val1;
        }
      }
    }

    __syncthreads();

    if(tch_off == 0) {
      scalar_t reduce_sum = 0;
      for(int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
          reduce_sum += prod_sum[idx];
      }
      const int indx2 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
      gradInput2[indx2] = reduce_sum;
    }

}


// naive assemble2 forward cuda kernel
int naive_assemble2_forward_cuda_kernel(
                                    at::Tensor& Aff,
                                    int gab,
                                    int gac,
                                    int gah,
                                    int gaw,
                                    int gasb,
                                    int gasc,
                                    int gash,
                                    int gasw,

                                    at::Tensor& update,
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

                                    /* at::Tensor& rInput1, */
                                    at::Tensor& rInput2,
                                    int pad_size,
                                    int kernel_size,
                                    int max_displacement,
                                    int stride1,
                                    int stride2,
                                    int corr_type_multiply,
                                    cudaStream_t stream)
{

    int batchSize = gab;
    int num = batchSize;

    int nInputChannels = uc;
    int inputWidth = uw;
    int inputHeight = uh;

    int nAffChannels = gac;
    int affWidth = gaw;
    int affHeight = gah;

    dim3 blocks_grid(batchSize, inputHeight, inputWidth);
    dim3 threads_block(THREADS_PER_BLOCK);


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "lltm_forward_cuda", ([&] {

        channels_first<scalar_t><<<blocks_grid, threads_block, 0, stream>>>(
            input2.data<scalar_t>(),
            rInput2.data<scalar_t>(),
            nInputChannels,
            inputHeight,
            inputWidth,
            pad_size
        );
    }));

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(inputHeight, inputWidth, nInputChannels);

    for (int n = 0; n < num; ++n) {

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "lltm_forward_cuda", ([&] {

          naive_assemble2_forward<scalar_t><<<totalBlocksCorr, threadsPerBlock, 0, stream>>> (
              n, update.data<scalar_t>(), nInputChannels, inputHeight, inputWidth,
              Aff.data<scalar_t>(), nAffChannels, affHeight, affWidth,
              rInput2.data<scalar_t>(),
              pad_size,
              kernel_size,
              max_displacement,
              stride1,
              stride2);
      }));
    }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in naive_assemble2_backward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}

// naive assemble2 backward cuda kernel
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
                                    /* at::Tensor& input1, */
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

                                    /* at::Tensor& rInput1, */
                                    at::Tensor& rInput2,
                                    int pad_size,
                                    int kernel_size,
                                    int max_displacement,
                                    int stride1,
                                    int stride2,
                                    int corr_type_multiply,
                                    cudaStream_t stream)
{

    int batchSize = gab;
    int num = batchSize;

    int nInputChannels = uc;
    int inputWidth = uw;
    int inputHeight = uh;

    int nAffChannels = gac;
    int affWidth = gaw;
    int affHeight = gah;

    dim3 blocks_grid(batchSize, inputHeight, inputWidth);
    dim3 threads_block(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "lltm_forward_cuda", ([&] {

        channels_first<scalar_t><<<blocks_grid, threads_block, 0, stream>>>(
            input2.data<scalar_t>(),
            rInput2.data<scalar_t>(),
            nInputChannels,
            inputHeight,
            inputWidth,
            pad_size
        );
    }));


    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(batchSize, affHeight, affWidth);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "naive_assemble2_forward", ([&] {

    naive_assemble2_backward_aff<scalar_t><<<totalBlocksCorr, threadsPerBlock, 0, stream>>> 
                        (gradAff.data<scalar_t>(), nAffChannels, affHeight, affWidth,
                         gradUpdate.data<scalar_t>(), nInputChannels, inputHeight, inputWidth,
                         rInput2.data<scalar_t>(),
                         pad_size,
                         kernel_size,
                         max_displacement,
                         stride1,
                         stride2);

    }));


    totalBlocksCorr = dim3(inputHeight, inputWidth, nInputChannels);

    for(int n = 0; n < batchSize; n++) {

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradUpdate.type(), "lltm_forward_cuda", ([&] {

        naive_assemble2_backward_input2<scalar_t><<<totalBlocksCorr, threadsPerBlock, 0, stream>>>(
            n, gradInput2.data<scalar_t>(), nInputChannels, inputHeight, inputWidth,
            Aff.data<scalar_t>(), nAffChannels, affHeight, affWidth,
            gradUpdate.data<scalar_t>(),
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);

        }));
    }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in naive_assemble2_backward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}
