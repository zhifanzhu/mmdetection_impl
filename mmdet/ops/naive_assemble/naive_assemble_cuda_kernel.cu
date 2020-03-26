#include <stdio.h>

#include "naive_assemble_cuda_kernel.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32
#define FULL_MASK 0xffffffff

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

inline int GET_BLOCKS(const int N) 
{ return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS; }

template<typename scalar_t>
__global__ void naive_assemble_forward(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ cur_prev_aff,
        const int P, // P = D^2 = (2*k+1)^2
        const scalar_t* __restrict__ feat,
        const int B,
        const int C,
        const int H,
        const int W,
        const int k,
        scalar_t* __restrict__ masked_cpa) 
{
    int n = B * H * W;
    CUDA_KERNEL_LOOP(index, n) {
        int HW = H*W;
        int b = index / HW;
        int y = (index / W) % H;
        int x = index % W;
        float bound = 1e-7;

        int D = 2 * k + 1;

        for (int c = 0; c < C; ++c) {

            // Init a mass counter for normalization
            float mass = 0.0; 
            for (int i = -k; i <= k; ++i) {
                for (int j = -k; j <= k; ++j) {
                    int prev_y = y + i;
                    int prev_x = x + j;
                    if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W) {
                        int flat_idx = b * P * HW + ((i+k) * D + (j+k)) * HW + y * W + x;
                        float coef = cur_prev_aff[flat_idx];
                        if (coef > 0)
                            mass += coef;
                    }
                }
            }

            float val = 0.0;
            if (mass > -bound && mass < bound) {
                // Avoid div by 0
                int flat_idx = b * P * HW + (k * D + k) * HW + y * W + x;
                int feat_flat_idx = b * C * HW + c * HW + y * W + x;
                val = feat[feat_flat_idx];
                if (c == 0)
                    masked_cpa[flat_idx] += 1.0;
            } else {
                for (int i = -k; i <= k; ++i) {
                    for (int j = -k; j <= k; ++j) {
                        int prev_y = y + i;
                        int prev_x = x + j;
                        if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W) {
                            int flat_idx = b * P * HW + ((i+k) * D + (j+k)) * HW + y * W + x;
                            float a = cur_prev_aff[flat_idx];
                            if (a > 0) {
                                a = a / mass;
                                int feat_flat_idx = b * C * HW + c * HW + prev_y * W + prev_x;
                                float fc = feat[feat_flat_idx];
                                val += a * fc;
                                if (c == 0)
                                    masked_cpa[flat_idx] += a;
                            }
                        }
                    }
                }
            }
            int out_idx = b * C * HW + c * HW + y * W + x;
            output[out_idx ] = val;
        }
        __syncthreads();
    }
}


template <typename scalar_t>
__global__ void naive_assemble_backward_Feat(
        const scalar_t* __restrict__ gradOutput,
        const int B,
        const int C,
        const int H,
        const int W,
        const int P,
        scalar_t* gradFeat,
        const int k,
        const scalar_t* __restrict__ masked_cpa)
{
    int n = B * H * W;
    CUDA_KERNEL_LOOP(index, n) {
        int HW = H*W;
        int b = index / HW;
        int y = (index / W) % H;
        int x = index % W;

        int D = 2 * k + 1;

        for (int c = 0; c < C; ++c ) {
            float grad_cum = 0.0;
            int gradFeat_ind = b * C * HW + c * HW + y * W + x;
            for (int i = -k; i <= k; ++i) {
                for (int j = -k; j <= k; ++j) {
                    int m = y - i;
                    int n = x - j;
                    if (m >= 0 && m < H && n >= 0 && n < W ) {
                        int gradOutput_ind = b * C * HW + c * HW + m * W + n;
                        int flat_ind = b * P * HW + ((i+k) * D + (j+k)) * HW + m * W + n;
                        float mask = masked_cpa[flat_ind];
                        float g_o = gradOutput[gradOutput_ind];
                        grad_cum += mask * g_o;
                    }
                }
            }
            gradFeat[gradFeat_ind] = grad_cum;
        }
    }
}

int naive_assemble_forward_cuda_kernel(at::Tensor& output,
                                       at::Tensor& cur_prev_aff,
                                       int aff_c,

                                       at::Tensor& feat,
                                       int ib,
                                       int ic,
                                       int ih,
                                       int iw,

                                       int k,
                                       at::Tensor& masked_cpa,
                                       cudaStream_t stream) 
{
   int batchSize = ib;

   int nAffChannels = aff_c;
   int nInputChannels = ic;
   int inputHeight = ih;
   int inputWidth = iw;

   dim3 threadsPerBlock(THREADS_PER_BLOCK);
   dim3 totalBlocksCorr(GET_BLOCKS(batchSize * inputHeight * inputWidth));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.type(), "naive_assemble_forward", ([&] {

   naive_assemble_forward<scalar_t><<<totalBlocksCorr, threadsPerBlock, 0, stream>>> 
                        (output.data<scalar_t>(), 
                         cur_prev_aff.data<scalar_t>(), nAffChannels,
                         feat.data<scalar_t>(), 
                         batchSize, nInputChannels, inputHeight, inputWidth,
                         k, masked_cpa.data<scalar_t>());
  }));

  cudaError_t err = cudaGetLastError();

  // check for errors
  if (err != cudaSuccess) {
    printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}


int naive_assemble_backward_cuda_kernel(
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
                                    cudaStream_t stream)
{
    int batchSize = gob;

    int nAffChannels = aff_c;

    int nOutputChannels = goc;
    int outputWidth = gow;
    int outputHeight = goh;

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(GET_BLOCKS(batchSize * outputHeight * outputWidth));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.type(), "naive_assemble_backward_Feat", ([&] {
        naive_assemble_backward_Feat<scalar_t><<<totalBlocksCorr, threadsPerBlock, 0, stream>>> (
            gradOutput.data<scalar_t>(), batchSize, nOutputChannels, outputHeight, outputWidth,
            nAffChannels, gradFeat.data<scalar_t>(),
            k, masked_cpa.data<scalar_t>());
    }));

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in naive_assemble_backward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}