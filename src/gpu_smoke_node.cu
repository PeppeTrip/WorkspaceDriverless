#include <rclcpp/rclcpp.hpp>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <vector>
#include <numeric>
#include <cmath>

struct SquareOp
{
  __host__ __device__
  float operator()(float x) const { return x * x; }
};

struct NearPredicate
{
  const float* points;
  float split2;

  __host__ __device__
  bool operator()(int idx) const
  {
    const float x = points[idx * 4 + 0];
    const float y = points[idx * 4 + 1];
    return (x * x + y * y) < split2;
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("gpu_smoke_node");
  auto logger = node->get_logger();

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count == 0)
  {
    RCLCPP_ERROR(logger, "CUDA check failed. status=%d (%s), device_count=%d",
                 static_cast<int>(st), cudaGetErrorString(st), device_count);
    rclcpp::shutdown();
    return 1;
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  RCLCPP_INFO(logger, "CUDA device[0]: %s | capability %d.%d", prop.name, prop.major, prop.minor);

  // Thrust transform + reduce smoke test
  thrust::host_vector<float> h(1024, 2.0f);
  thrust::device_vector<float> d = h;
  thrust::transform(d.begin(), d.end(), d.begin(), SquareOp());
  float sum = thrust::reduce(d.begin(), d.end(), 0.0f, thrust::plus<float>());

  const float expected = 1024.0f * 4.0f;
  if (std::fabs(sum - expected) > 1e-3f)
  {
    RCLCPP_ERROR(logger, "Thrust transform/reduce mismatch. got=%f expected=%f", sum, expected);
    rclcpp::shutdown();
    return 2;
  }

  // Thrust copy_if smoke test for near/far partition on [x,y,z,i]
  const int n = 8;
  std::vector<float> interleaved = {
      1.0f, 1.0f, 0.0f, 0.0f,  // near
      4.0f, 0.0f, 0.0f, 1.0f,  // far
      2.0f, 0.0f, 0.0f, 2.0f,  // near
      0.0f, 5.0f, 0.0f, 3.0f,  // far
      0.5f, 0.5f, 0.0f, 4.0f,  // near
      3.1f, 0.0f, 0.0f, 5.0f,  // far
      0.0f, 2.9f, 0.0f, 6.0f,  // near
      0.0f, 3.2f, 0.0f, 7.0f   // far
  };

  thrust::device_vector<float> d_points = interleaved;
  thrust::device_vector<int> d_idx(n);
  thrust::counting_iterator<int> first(0);
  thrust::copy(first, first + n, d_idx.begin());

  thrust::device_vector<int> near_idx(n);
  const float split_m = 3.0f;
  auto near_end = thrust::copy_if(d_idx.begin(), d_idx.end(), near_idx.begin(),
                                  NearPredicate{thrust::raw_pointer_cast(d_points.data()), split_m * split_m});
  int near_count = static_cast<int>(near_end - near_idx.begin());
  int far_count = n - near_count;

  if (near_count != 4 || far_count != 4)
  {
    RCLCPP_ERROR(logger, "Thrust copy_if partition mismatch. near=%d far=%d (expected 4/4)", near_count, far_count);
    rclcpp::shutdown();
    return 3;
  }

  RCLCPP_INFO(logger, "GPU smoke test passed: thrust transform/reduce and copy_if partition are working.");
  rclcpp::shutdown();
  return 0;
}
