#include <torch/extension.h>
#include <vector>
#include <string>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor init_cuda(
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid);

std::vector<torch::Tensor> render_forward_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid,
    std::string phase_name);

torch::Tensor init(
    torch::Tensor points,
    torch::Tensor tindex,
    std::vector<int> grid) {
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return init_cuda(points, tindex, grid);
}

std::vector<torch::Tensor> render_forward(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    std::vector<int> grid,
    std::string phase_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return render_forward_cuda(sigma, origin, points, tindex, grid, phase_name);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "DVR init (CUDA)");
  m.def("render_forward", &render_forward, "DVR render forward (CUDA)");
}
