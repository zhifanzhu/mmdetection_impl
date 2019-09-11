#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), "CPU op not implemented")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

