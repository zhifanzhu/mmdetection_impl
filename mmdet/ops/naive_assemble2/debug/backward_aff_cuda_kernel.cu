#include <iostream>
#include "debug_utils.h"
using namespace std;

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
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


int main() {
    using namespace debug;
    int batchSize = 1;
    int nAffChannels = 9, affHeight = 2, affWidth = 2,
        nInputChannels = 1, inputHeight = 2, inputWidth = 2,
        pad_size = 1, kernel_size = 1, max_displacement = 1,
        stride1 = 1, stride2 = 1;
    typedef double scalar_t;
    scalar_t *gradAff, **rgradUpdate, *rInput2;
    gradAff = Zeros<scalar_t>({batchSize, nAffChannels, affHeight, affWidth});
    rgradUpdate = Ones<scalar_t>({batchSize, inputHeight, inputWidth, nInputChannels});
    rInput2 = Ones<scalar_t>({batchSize, inputHeight, inputWidth, nInputChannels});

    naive_assemble2_backward_input2<<<totalBlocksCorr, threadsPerBlock, 0, stream>>>
        (gradAff, nAffChannels, affHeight, affWidth,
         rgradUpdate, nInputChannels, inputHeight, inputWidth,
         rInput2,
         pad_size,
         kernel_size,
         max_displacement,
         stride1,
         stride2);
    cudaFree(gradAff);
    cudaFree(rgradUpdate);
    cudaFree(rInput2);
    checkCudaErrors(cudaGetLastError());
    std::cout << "Execution success" << std::endl;
    return 0;
}

/* void attgridgen_gpu(const at::Tensor attx, const at::Tensor atty, */
/*     at::Tensor map_xi, at::Tensor map_yi, */
/*     at::Tensor index_x, at::Tensor index_y, */
/*     const int batch_size, const int att_size, const int out_size, */ 
/*     const float threshold, const int iters) */
/* { */
/*     int num_kernels = batch_size; */
/*     AT_DISPATCH_FLOATING_TYPES( */
/*         attx.type(), "att_grid_generator_gpu", ([&] { */
/*             scalar_t *attx_ = attx.data<scalar_t>(); */
/*             scalar_t *atty_ = atty.data<scalar_t>(); */
/*             scalar_t *map_xi_ = map_xi.data<scalar_t>(); */
/*             scalar_t *map_yi_ = map_yi.data<scalar_t>(); */
/*             scalar_t *index_x_ = index_x.data<scalar_t>(); */
/*             scalar_t *index_y_ = index_y.data<scalar_t>(); */

/*             att_grid_generator_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>( */
/*                 num_kernels, attx_, atty_, map_xi_, map_yi_, index_x_, index_y_, */ 
/*                 batch_size, att_size, out_size, threshold, iters); */

/*         }) */
/*     ); */

/*     cudaError_t err = cudaGetLastError(); */
/*     if (err != cudaSuccess) { */
/*         printf("error in att_grid_generator: %s\n", cudaGetErrorString(err)); */
/*     } */

/* } */

