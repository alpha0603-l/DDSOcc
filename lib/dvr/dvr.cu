#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <cfloat>
#include <cmath>
#include <algorithm>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#define MAX_D 1446
#define MAX_STEP 1000

enum PhaseName { TEST = 0, TRAIN = 1 };

template <typename scalar_t>
__global__ void init_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> occupancy) {

  const int n = (int)blockIdx.y; // batch
  const int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; // ray

  const int M = (int)points.size(1);
  const int T = (int)occupancy.size(1);

  if (c >= M) return;

  const int t = tindex[n][c];
  if (t < 0) return;

  const int ts = (T == 1) ? 0 : t;

  const int vzsize = (int)occupancy.size(2);
  const int vysize = (int)occupancy.size(3);
  const int vxsize = (int)occupancy.size(4);

  // 注意：这里做了 cast，确保精度转换
  const int vx = (int)points[n][c][0];
  const int vy = (int)points[n][c][1];
  const int vz = (int)points[n][c][2];

  if (0 <= vx && vx < vxsize &&
      0 <= vy && vy < vysize &&
      0 <= vz && vz < vzsize) {
    occupancy[n][ts][vz][vy][vx] = static_cast<scalar_t>(1);
  }
}

template <typename scalar_t>
__global__ void render_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coord_index,
    PhaseName train_phase) {

  const int n = (int)blockIdx.y;
  const int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

  const int M = (int)points.size(1);
  const int T = (int)sigma.size(1);

  if (c >= M) return;

  const int t = tindex[n][c];
  if (t < 0) return;

  const int ts = (T == 1) ? 0 : t;

  const int vzsize = (int)sigma.size(2);
  const int vysize = (int)sigma.size(3);
  const int vxsize = (int)sigma.size(4);

  const double xo = static_cast<double>(origin[n][t][0]);
  const double yo = static_cast<double>(origin[n][t][1]);
  const double zo = static_cast<double>(origin[n][t][2]);

  const double xe = static_cast<double>(points[n][c][0]);
  const double ye = static_cast<double>(points[n][c][1]);
  const double ze = static_cast<double>(points[n][c][2]);

  int vx = (int)xo;
  int vy = (int)yo;
  int vz = (int)zo;

  const double rx = xe - xo;
  const double ry = ye - yo;
  const double rz = ze - zo;

  double gt_d = ::sqrt(rx * rx + ry * ry + rz * rz);

  if (gt_d <= 1e-12) {
    pred_dist[n][c] = static_cast<scalar_t>(0);
    gt_dist[n][c] = static_cast<scalar_t>(0);
    coord_index[n][c][0] = static_cast<scalar_t>(vx);
    coord_index[n][c][1] = static_cast<scalar_t>(vy);
    coord_index[n][c][2] = static_cast<scalar_t>(vz);
    return;
  }

  const double dx = rx / gt_d;
  const double dy = ry / gt_d;
  const double dz = rz / gt_d;

  const int stepX = (dx >= 0) ? 1 : -1;
  const int stepY = (dy >= 0) ? 1 : -1;
  const int stepZ = (dz >= 0) ? 1 : -1;

  const double next_x = vx + (stepX < 0 ? 0.0 : 1.0);
  const double next_y = vy + (stepY < 0 ? 0.0 : 1.0);
  const double next_z = vz + (stepZ < 0 ? 0.0 : 1.0);

  double tMaxX = (dx != 0.0) ? (next_x - xo) / dx : DBL_MAX;
  double tMaxY = (dy != 0.0) ? (next_y - yo) / dy : DBL_MAX;
  double tMaxZ = (dz != 0.0) ? (next_z - zo) / dz : DBL_MAX;

  const double tDeltaX = (dx != 0.0) ? (double)stepX / dx : DBL_MAX;
  const double tDeltaY = (dy != 0.0) ? (double)stepY / dy : DBL_MAX;
  const double tDeltaZ = (dz != 0.0) ? (double)stepZ / dz : DBL_MAX;

  int3 path[MAX_D];
  double csd[MAX_D];
  double d[MAX_D];

  int step = 0;
  int count = 0;
  double last_d = 0.0;
  bool was_inside = false;

  while (true) {
    const bool inside =
        (0 <= vx && vx < vxsize) &&
        (0 <= vy && vy < vysize) &&
        (0 <= vz && vz < vzsize);

    if (inside) {
      was_inside = true;
      if (count < MAX_D) path[count] = make_int3(vx, vy, vz);
    } else if (was_inside) {
      break;
    }

    double _d;
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        _d = tMaxX; vx += stepX; tMaxX += tDeltaX;
      } else {
        _d = tMaxZ; vz += stepZ; tMaxZ += tDeltaZ;
      }
    } else {
      if (tMaxY < tMaxZ) {
        _d = tMaxY; vy += stepY; tMaxY += tDeltaY;
      } else {
        _d = tMaxZ; vz += stepZ; tMaxZ += tDeltaZ;
      }
    }

    if (inside && count < MAX_D) {
      const int3 &v = path[count];
      // 这里的 scalar_t 如果是 half，static_cast 会自动处理
      const double _sigma = static_cast<double>(sigma[n][ts][v.z][v.y][v.x]);
      const double _delta = ::fmax(0.0, _d - last_d);
      const double sd = _sigma * _delta;

      if (count == 0) {
        csd[count] = sd;
      } else {
        csd[count] = csd[count - 1] + sd;
      }
      d[count] = _d;
      count++;
    }

    last_d = _d;
    step++;
    if (step > MAX_STEP || count >= MAX_D) break;
  }

  if (count <= 0) return;

  double exp_d = d[count - 1];
  int x = path[count - 1].x;
  int y = path[count - 1].y;
  int z = path[count - 1].z;

  for (int i = 0; i < count; i++) {
    const int3 &v = path[i];
    const double occ = static_cast<double>(sigma[n][ts][v.z][v.y][v.x]);
    if (occ > 0.5) {
      exp_d = d[i];
      x = v.x; y = v.y; z = v.z;
      break;
    }
  }

  const double max_d = d[count - 1];
  if (train_phase == TRAIN) {
    gt_d = ::fmin(gt_d, max_d);
  }

  pred_dist[n][c] = static_cast<scalar_t>(exp_d);
  gt_dist[n][c] = static_cast<scalar_t>(gt_d);

  coord_index[n][c][0] = static_cast<scalar_t>(x);
  coord_index[n][c][1] = static_cast<scalar_t>(y);
  coord_index[n][c][2] = static_cast<scalar_t>(z);
}

std::vector<torch::Tensor> render_forward_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> /*grid*/,
    std::string phase_name) {

  const auto N = points.size(0);
  const auto M = points.size(1);

  at::cuda::CUDAGuard device_guard(sigma.device());

  auto pred_dist = torch::full({N, M}, -1, sigma.options());
  auto gt_dist   = torch::full({N, M}, -1, sigma.options());
  auto coord_index = torch::zeros({N, M, 3}, sigma.options());

  PhaseName train_phase;
  if (phase_name == "test") train_phase = TEST;
  else if (phase_name == "train") train_phase = TRAIN;
  else TORCH_CHECK(false, "UNKNOWN PHASE NAME: ", phase_name);

  const int threads = 1024;
  const dim3 blocks((M + threads - 1) / threads, N);

  // 确保索引为 int32
  if (tindex.scalar_type() != at::kInt) {
    tindex = tindex.to(at::kInt);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigma.scalar_type(), "render_forward_cuda", ([&] {
    render_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        sigma.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        tindex.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        pred_dist.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        gt_dist.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        coord_index.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        train_phase);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pred_dist, gt_dist, coord_index};
}

torch::Tensor init_cuda(
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid) {

  const auto N = points.size(0);
  const auto M = points.size(1);

  const int T = grid[0];
  const int H = grid[1];
  const int L = grid[2];
  const int W = grid[3];

  at::cuda::CUDAGuard device_guard(points.device());

  auto occupancy = torch::zeros({N, T, H, L, W}, points.options());

  const int threads = 1024;
  const dim3 blocks((M + threads - 1) / threads, N);

  if (tindex.scalar_type() != at::kInt) {
    tindex = tindex.to(at::kInt);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "init_cuda", ([&] {
    init_cuda_kernel<scalar_t><<<blocks, threads>>>(
        points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        tindex.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        occupancy.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return occupancy;
}
