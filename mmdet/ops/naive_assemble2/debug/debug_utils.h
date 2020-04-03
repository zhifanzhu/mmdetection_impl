#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <cstring>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

namespace debug {
    // Example Usage:
    //
    // scalar_t *attx;
    // attx = Ones<scalar_t>({4, 3, 1, 64});
    // example_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(attx);
    // cudaFree(attx);  // <- Need this
    // checkCudaErrors(cudaGetLastError());
    //
    // Note:
    // 1. One must call cudaFree() at the end of program. Otherwise it would
    // not be able to break inside kernel code.
    //
    // 2. When you see some error like"too many resources requeted, error code 0x7",
    // consider reduce CUDA_NUM_THREADS from 1024 to a smaller one (e.g. 256).
    //

    template <typename scalar_t>
    inline scalar_t* Ones(std::vector<int> shape) {
        scalar_t *d_ptr;
        size_t num_elem = 1;
        for (auto &s : shape)
            num_elem *= s;
        size_t tensor_size = sizeof(scalar_t) * num_elem;

        scalar_t *h_ptr = new scalar_t[tensor_size];
        double initValue = 1.0;
        std::fill_n(h_ptr, num_elem, initValue);
        checkCudaErrors(cudaMalloc(&d_ptr, tensor_size));
        checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, tensor_size, cudaMemcpyHostToDevice));
        delete []h_ptr;
        return d_ptr;
    }

    template <typename scalar_t>
    inline scalar_t* Zeros(std::vector<int> shape) {
        scalar_t *d_ptr;
        size_t num_elem = 1;
        for (auto &s : shape)
            num_elem *= s;
        size_t tensor_size = sizeof(scalar_t) * num_elem;

        scalar_t *h_ptr = new scalar_t[tensor_size];
        std::memset(h_ptr, 0, num_elem);
        checkCudaErrors(cudaMalloc(&d_ptr, tensor_size));
        checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, tensor_size, cudaMemcpyHostToDevice));
        delete []h_ptr;
        return d_ptr;
    }

} // namespace debug
#endif
