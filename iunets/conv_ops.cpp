#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

/*
This creates a Python-wrapper for the adjoint of the convolution with respect to the weight, which
handles the memory favorably compared to the Python implementation of Pytorch. Code adapted from
https://discuss.pytorch.org/t/cuda-error-with-cudnn-convolution-backward-weight-function/41214
by Hans Pinckaers.
*/

at::Tensor convolution_backward_weight(
    const at::Tensor& input,
    c10::ArrayRef<int64_t> weight_size,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

    return at::cudnn_convolution_backward_weight(
        weight_size,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
}