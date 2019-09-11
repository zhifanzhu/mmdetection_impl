#include <ATen/ATen.h>
#include "common/cuda_common.cuh"


// TODO - replace placeInputPtr and placeOutputPtr with single function


/*
    given input (|B|, C, H, W), return pointer to input[b, 0, i, j]
    by mapping from 4-dimensional index to contiguous memory offset.
*/
template<typename scalar_t>
__device__ __forceinline__ scalar_t* placeInputPtr(
    scalar_t* const FM,  // (|B|, C, H, W) input feature map.
    const int b, const int i, const int j,  // n-dimensional index.
    const int iC, const int iH, const int iW  // input shape.
)
{
    int offset(  // "unroll" dimensions
          b * iW * iH * iC
//      + 0 * iW * iH
        + i * iW
        + j
    );

    return FM + offset;
}


/*
    given output (|B|, H, W, (2d+1), (2d+1)), return pointer to
    out[b, i, j, ci, cj]
    by mapping from 5-dimensional index to contiguous memory offset.
*/
template<typename scalar_t>
__device__ __forceinline__ scalar_t* placeOutputPtr(
    scalar_t* const out,  // (|B|, H, W, (2d+1), (2d+1)) correlation maps.
    const int b, const int i, const int j,  // n-dimensional index.
    const int ci, const int cj,  // correlation map index.
    const int iH, const int iW,  // input width and height.
    const int cHW  // correlation map width and height (2d+1).
)
{
    int offset(  // "unroll" dimensions.
          b * cHW * cHW * iW * iH
        + i * cHW * cHW * iW
        + j * cHW * cHW
        + ci * cHW
        + cj
    );

    return out + offset;
}


/*
    pointwise correlations CUDA kernel
    each thread is responsible for out[b, i, j, :, :] for some (b, i, j)
    for each (i, j) location in FM0, compute correlations in a local
    neighborhood around (i, j) with FM1.
*/
template <typename scalar_t>
__global__ void pointwiseCorrelationKernelForward(
    const scalar_t* const __restrict__ FM0,  // (|B|, C, H, W) feature map at time t
    const scalar_t* const __restrict__ FM1,  // (|B|, C, H, W) feature map at time t+tau
    scalar_t* const __restrict__ out,  // (|B|, H, W, (2d+1), (2d+1)) assume zero-initialized
    const int iB,  // input batches
    const int iC,  // input feature map channels
    const int iH,  // input feature map height
    const int iW,  // input feature map width
    const int dMax,  // maximum displacement
    const int stride  // correlation stride
)
{
    for (  // grid-stride loop
        int ind = blockIdx.x * blockDim.x + threadIdx.x;
        ind < (iB * iH * iW);
        ind += blockDim.x * gridDim.x
    )
    {
        // this thread is responsible for values in out[b, i, j, :, :]
        // determine values of b, i, and j by
        // mapping from offset in contiguous memory to n-dimensional index.
        const int b(ind / iW / iH);
        const int i((ind / iW) % iH);
        const int j(ind % iW);

        // get pointer to FM0[b, 0, i, j]
        const scalar_t* const FM0Cntr(placeInputPtr(FM0, b, i, j, iC, iH, iW));

        // iterate over all local displacements
        for (int di = max(0, i-dMax); di < min(i+dMax, iH); di += stride) {
            for (int dj = max(0, j-dMax); dj < min(j+dMax, iW); dj += stride) {

                // get pointer to FM1[b, 0, di, dj]
                const scalar_t* const FM1Disp(placeInputPtr(FM1, b, di, dj, iC, iH, iW));
                // get pointer to out[b, i, j, ci, cj]
                scalar_t* const outDisp(placeOutputPtr(
                    out, b, i, j, di-i+dMax, dj-j+dMax, iH, iW, (2*dMax+1)
                ));

                // inner product (worth it to replace with cuBLAS implmentation?)
                // <FM0[b, :, i, j], FM1[b, :, di, dj]>
                const int cStride(iH * iW);
                for (int c = 0; c < iC; c++) {  // iterate through channels
                    *outDisp += FM0Cntr[c*cStride] * FM1Disp[c*cStride];
                }
            }
        }
    }
}


/*
    given loss derivatives wrt output, compute loss derivatives wrt
    FM0 and FM1.
    each thread is responsible for derivatives associated with
    out[b, i, j, :, :] for some (b, i, j).
*/
template<typename scalar_t>
__global__ void pointwiseCorrelationKernelBackward(
    const scalar_t* const __restrict__ gradOut,  // (|B|, H, W, (2d+1), (2d+1)) correlation maps.
    const scalar_t* const __restrict__ FM0,  // (|B|, C, H, W) feature map at time t.
    const scalar_t* const __restrict__ FM1,  // (|B|, C, H, W) feature map at time t+tau.
    scalar_t* const __restrict__ gradFM0,  // (|B|, C, H, W) assume zero-initialized.
    scalar_t* const __restrict__ gradFM1,  // (|B|, C, H, W) assume zero-initialized.
    const int iB,  // input batches
    const int iC,  // input feature map channels
    const int iH,  // input feature map height
    const int iW,  // input feature map width
    const int dMax,  // maximum displacement
    const int stride  // stride between displacements
)
{
    for (  // grid-stride loop
        int ind = blockIdx.x * blockDim.x + threadIdx.x;
        ind < (iB * iH * iW);
        ind += blockDim.x * gridDim.x
    )
    {
        // this thread is responsible for computing derivatives from
        // output[b, i, j, :, :]
        // determine which b, i, j this thread is responsible for by
        // mapping from offset in contiguous memory to n-dimensional index.
        const int b(ind / iW / iH);
        const int i((ind / iW) % iH);
        const int j(ind % iW);

        // get pointers to FM0[b, 0, i, j] and gradFM0[b, 0, i, j]
        const scalar_t* const FM0Cntr(placeInputPtr(FM0, b, i, j, iC, iH, iW));
        scalar_t* const gradFM0Cntr(placeInputPtr(gradFM0, b, i, j, iC, iH, iW));

        // iterate over all local displacements
        for (int di = max(0, i-dMax); di < min(i+dMax, iH); di += stride) {
            for (int dj = max(0, j-dMax); dj < min(j+dMax, iW); dj += stride) {

                // get pointers to FM1[b, 0, di, dj] and gradFM1[b, 0, di, dj]
                const scalar_t* const FM1Disp(placeInputPtr(FM1, b, di, dj, iC, iH, iW));
                scalar_t* const gradFM1Disp(placeInputPtr(gradFM1, b, di, dj, iC, iH, iW));
                // get pointer to gradOut[b, i, j, ci, cj]
                const scalar_t* const gradOutDisp(placeOutputPtr(
                    gradOut, b, i, j, di-i+dMax, dj-j+dMax, iH, iW, (2*dMax+1)
                ));

                // dL/dIn = dL/dOut * dOut/dIn
                const int cStride(iH * iW);
                for (int c = 0; c < iC; c++) {  // iterate through channels
                    gradFM0Cntr[c*cStride] += *gradOutDisp * FM1Disp[c*cStride];
                    atomicAdd(gradFM1Disp + c*cStride, *gradOutDisp * FM0Cntr[c*cStride]);
                }
            }
        }
    }
}


/* initialize output and launch kernel. */
at::Tensor pointwiseCorrelationCudaForward(
    const at::Tensor& FM0,  // (|B|, C, H, W) feature map at time t.
    const at::Tensor& FM1,  // (|B|, C, H, W) feature map at time t+tau.
    const int dMax,  // maximum displacement.
    const int stride  // stride between displacements.
)
{
    const int iB(FM0.size(0));  // input+output batch size
    const int iC(FM0.size(1));  // input channels
    const int iH(FM0.size(2));  // input+output height
    const int iW(FM0.size(3));  // input+output width
    const int cHW(2 * dMax + 1);  // output correlation map height and width

    // CUDA kernel will assume zero-initialization
    at::Tensor out = at::zeros({iB, iH, iW, cHW, cHW}, FM0.options());

    const dim3 numBlocks(ceilDivide(iB * iH * iW, THREADS_PER_BLOCK));

    AT_DISPATCH_FLOATING_TYPES(
        out.type(), "pointwiseCorrelationsKernelForward", ([&] {
            pointwiseCorrelationKernelForward<scalar_t>
            <<<numBlocks, THREADS_PER_BLOCK>>>(
                FM0.data<scalar_t>(),
                FM1.data<scalar_t>(),
                out.data<scalar_t>(),
                iB, iC, iH, iW,
                dMax, stride
            );
        })
    );

    return out;
}


/* initialize output and launch kernel. */
std::tuple<at::Tensor, at::Tensor> pointwiseCorrelationCudaBackward(
    const at::Tensor& gradOut,  // (|B|, H, W, (2d+1), (2d+1)) correlation maps.
    const at::Tensor& FM0,  // (|B|, C, H, W) feature map at time t.
    const at::Tensor& FM1,  // (|B|, C, H, W) feature map at time t+tau.
    const int dMax,  // maximum displacement.
    const int stride  // stride between displacements.
)
{
    const int iB(FM0.size(0));  // input+output batch size
    const int iC(FM0.size(1));  // input channels
    const int iH(FM0.size(2));  // input+output height
    const int iW(FM0.size(3));  // input+output width

    // CUDA kernel will assume zero-initialization
    at::Tensor gradFM0 = at::zeros_like(FM0);
    at::Tensor gradFM1 = at::zeros_like(FM1);

    const dim3 numBlocks(ceilDivide(iB * iH * iW, THREADS_PER_BLOCK));

    AT_DISPATCH_FLOATING_TYPES(
        gradFM0.type(), "pointwiseCorrelationsKernelBackward", ([&] {
            pointwiseCorrelationKernelBackward<scalar_t>
            <<<numBlocks, THREADS_PER_BLOCK>>>(
                gradOut.data<scalar_t>(),
                FM0.data<scalar_t>(),
                FM1.data<scalar_t>(),
                gradFM0.data<scalar_t>(),
                gradFM1.data<scalar_t>(),
                iB, iC, iH, iW,
                dMax, stride
            );
        })
    );

    return std::make_tuple(gradFM0, gradFM1);
}