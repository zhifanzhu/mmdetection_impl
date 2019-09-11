#include <torch/torch.h>
#include "common/cpp_common.hpp"

/* CUDA forward declaration */
at::Tensor pointwiseCorrelationCudaForward(
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
);

/* CUDA forward declaration */
std::tuple<at::Tensor, at::Tensor> pointwiseCorrelationCudaBackward(
    const at::Tensor& gradOutput,
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
);


/* check input and forward to CUDA function */
at::Tensor pointwiseCorrelationForward(
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
)
{
    CHECK_INPUT(FM0);
    CHECK_INPUT(FM1);
    return pointwiseCorrelationCudaForward(FM0, FM1, dMax, stride);
}

/* check input and forward to CUDA function */
std::tuple<at::Tensor, at::Tensor> pointwiseCorrelationBackward(
    const at::Tensor& gradOut,
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
)
{
    CHECK_INPUT(gradOut);
    CHECK_INPUT(FM0);
    CHECK_INPUT(FM1);
    return pointwiseCorrelationCudaBackward(gradOut, FM0, FM1, dMax, stride);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "pointwise_correlation_forward",
        &pointwiseCorrelationForward,
        "pointwise correlation forward pass"
    );
    m.def(
        "pointwise_correlation_backward",
        &pointwiseCorrelationBackward,
        "pointwise correlation backward pass"
    );
}