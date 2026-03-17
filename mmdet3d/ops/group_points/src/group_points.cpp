#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void group_points_kernel_launcher(int b, int c, int n, int npoints, int nsample,
                                  const float *points, const int *idx,
                                  float *out, cudaStream_t stream);

void group_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                       int nsample, const float *grad_out,
                                       const int *idx, float *grad_points,
                                       cudaStream_t stream);

int group_points_grad_wrapper(int b, int c, int n, int npoints, int nsample,
                              at::Tensor grad_out_tensor, at::Tensor idx_tensor,
                              at::Tensor grad_points_tensor) {
  const float *grad_points = grad_points_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  const float *grad_out = grad_out_tensor.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  group_points_grad_kernel_launcher(b, c, n, npoints, nsample, grad_out, idx,
                                    const_cast<float*>(grad_points), stream);
  return 1;
}

int group_points_wrapper(int b, int c, int n, int npoints, int nsample,
                         at::Tensor points_tensor, at::Tensor idx_tensor,
                         at::Tensor out_tensor) {
  const float *points = points_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  float *out = out_tensor.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  group_points_kernel_launcher(b, c, n, npoints, nsample, points, idx, out,
                               stream);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("group_points_wrapper", &group_points_wrapper, "group_points_wrapper");
  m.def("group_points_grad_wrapper", &group_points_grad_wrapper, "group_points_grad_wrapper");
}
