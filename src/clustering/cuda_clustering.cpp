#include "cuda_clustering/clustering/cuda_clustering.hpp"
#include "cuda_clustering/clustering/cluster_filtering/dimension_filter.hpp"

#include <cmath>
#include <iostream>

CudaClustering::CudaClustering(clustering_parameters& param)
{
  // Populate near parameters from YAML/ROS params
  this->ecp_near.minClusterSize   = param.clustering.minClusterSize;   // Minimum cluster size to filter out noise
  this->ecp_near.maxClusterSize   = param.clustering.maxClusterSize;   // Maximum size for large objects
  this->ecp_near.voxelX           = param.clustering.voxelX;           // Down-sampling resolution in X (meters)
  this->ecp_near.voxelY           = param.clustering.voxelY;           // Down-sampling resolution in Y (meters)
  this->ecp_near.voxelZ           = param.clustering.voxelZ;           // Down-sampling resolution in Z (meters)
  this->ecp_near.countThreshold   = param.clustering.countThreshold;   // Minimum points per voxel

  // Far parameters: start identical (you can tune later)
  this->ecp_far = this->ecp_near;

  filter = new DimensionFilter(param.filtering);
  cudaStreamCreate(&stream);
}

void CudaClustering::getInfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
      cudaGetDeviceProperties(&prop, i);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"----device id: %d info----\n", i);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  GPU : %s \n", prop.name);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  Capability: %d.%d\n", prop.major, prop.minor);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  Const memory: %luKB\n", prop.totalConstMem  >> 10);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  warp size: %d\n", prop.warpSize);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  threads in a block: %d\n", prop.maxThreadsPerBlock);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  block dim: (%d,%d,%d)\n",
                  prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"  grid dim: (%d,%d,%d)\n",
                  prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"),"\n");
}

void CudaClustering::reallocateMemory(unsigned int sizeEC)
{
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "REALLOC");

  // indexEC is used as:
  // indexEC[0] = number of clusters
  // indexEC[i] = size of cluster i (for i>=1)
  // The original code allocates *4*sizeEC, keep it as-is.
  cudaFree(indexEC);
  cudaStreamSynchronize(stream);

  cudaMallocManaged(&indexEC, sizeof(unsigned int) * 4 * (sizeEC), cudaMemAttachHost);
  cudaStreamSynchronize(stream);

  cudaStreamAttachMemAsync(stream, indexEC);
  cudaStreamSynchronize(stream);

  // Also ensure split buffers exist and are large enough for worst case (sizeEC points)
  cudaFree(nearInput);
  cudaFree(farInput);
  nearInput = nullptr;
  farInput  = nullptr;

  cudaMallocManaged(&nearInput, sizeof(float) * 4 * (sizeEC), cudaMemAttachHost);
  cudaMallocManaged(&farInput,  sizeof(float) * 4 * (sizeEC), cudaMemAttachHost);
  cudaStreamSynchronize(stream);

  cudaStreamAttachMemAsync(stream, nearInput);
  cudaStreamAttachMemAsync(stream, farInput);
  cudaStreamSynchronize(stream);
}

void CudaClustering::clustersToMarkers(
    float* cloud_in,
    unsigned int nCount,
    float* outputEC,
    unsigned int* index,
    std::shared_ptr<visualization_msgs::msg::Marker> cones
)
{
  (void)cloud_in; // not used here (we operate on outputEC + index like original code)
  (void)nCount;

  cones->points.clear();

  // Same logic from your original extractClusters() loop
  for (size_t i = 1; i <= index[0]; i++)
  {
    unsigned int outoff = 0;
    for (size_t w = 1; w < i; w++)
    {
      if (i > 1)
        outoff += index[w];
    }

    std::optional<geometry_msgs::msg::Point> pnt_opt =
        filter->analiseCluster(&outputEC[outoff * 4], index[i]);

    if (pnt_opt.has_value())
      cones->points.push_back(pnt_opt.value());
  }
}

void CudaClustering::extractClusters(
    float* input,
    unsigned int inputSize,
    float* outputEC,
    std::shared_ptr<visualization_msgs::msg::Marker> cones)
{
  std::cout << "\n------------ CUDA Clustering ---------------- " << std::endl;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  if (memoryAllocated < inputSize)
  {
    reallocateMemory(inputSize);
    memoryAllocated = inputSize;
  }

  // Keep your original copy direction/behavior unchanged
  cudaMemcpyAsync(outputEC, input, sizeof(float) * 4 * inputSize, cudaMemcpyHostToDevice, stream);

  cudaMemsetAsync(indexEC, 0, sizeof(unsigned int) * 4 * (inputSize), stream);
  cudaStreamSynchronize(stream);

  cudaExtractCluster cudaec(stream);
  cudaec.set(this->ecp_near); // <- use near params as default
  cudaStreamSynchronize(stream);

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
      std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA Memory Time: %f ms.", time_span.count());

  cudaec.extract(input, inputSize, outputEC, indexEC);
  cudaStreamSynchronize(stream);

  // Convert clustered points -> filtered markers
  clustersToMarkers(input, inputSize, outputEC, indexEC, cones);

  std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t3 - t2);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA extract by Time: %f ms.", time_span.count());
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t3 - t1);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA Total Time: %f ms.", time_span.count());
}

void CudaClustering::extractClustersNearFar(
    float* input,
    unsigned int inputSize,
    float* outputEC,
    std::shared_ptr<visualization_msgs::msg::Marker> conesNear,
    std::shared_ptr<visualization_msgs::msg::Marker> conesFar,
    float range_split_m)
{
  std::cout << "\n------------ CUDA Clustering (Near/Far) ---------------- " << std::endl;

  conesNear->points.clear();
  conesFar->points.clear();

  if (inputSize == 0)
    return;

  if (memoryAllocated < inputSize)
  {
    reallocateMemory(inputSize);
    memoryAllocated = inputSize;
  }

  nearSize = 0;
  farSize  = 0;

  const float split2 = range_split_m * range_split_m;

  // Split host-accessible input into nearInput/farInput
  // input is managed memory in your pipeline, so this should be OK after syncs upstream.
  for (unsigned int i = 0; i < inputSize; i++)
  {
    const float x = input[i * 4 + 0];
    const float y = input[i * 4 + 1];
    const float r2 = x * x + y * y;

    float* dst = (r2 < split2)
      ? &nearInput[nearSize * 4]
      : &farInput[farSize * 4];

    dst[0] = input[i * 4 + 0];
    dst[1] = input[i * 4 + 1];
    dst[2] = input[i * 4 + 2];
    dst[3] = input[i * 4 + 3];

    if (r2 < split2) nearSize++;
    else             farSize++;
  }

  // --- NEAR PASS ---
  if (nearSize > 0)
  {
    cudaMemcpyAsync(outputEC, nearInput, sizeof(float) * 4 * nearSize, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(indexEC, 0, sizeof(unsigned int) * 4 * (nearSize), stream);
    cudaStreamSynchronize(stream);

    cudaExtractCluster cudaec(stream);
    cudaec.set(this->ecp_near);
    cudaStreamSynchronize(stream);

    cudaec.extract(nearInput, nearSize, outputEC, indexEC);
    cudaStreamSynchronize(stream);

    clustersToMarkers(nearInput, nearSize, outputEC, indexEC, conesNear);
  }

  // --- FAR PASS ---
  if (farSize > 0)
  {
    cudaMemcpyAsync(outputEC, farInput, sizeof(float) * 4 * farSize, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(indexEC, 0, sizeof(unsigned int) * 4 * (farSize), stream);
    cudaStreamSynchronize(stream);

    cudaExtractCluster cudaec(stream);
    cudaec.set(this->ecp_far);
    cudaStreamSynchronize(stream);

    cudaec.extract(farInput, farSize, outputEC, indexEC);
    cudaStreamSynchronize(stream);

    clustersToMarkers(farInput, farSize, outputEC, indexEC, conesFar);
  }
}

CudaClustering::~CudaClustering()
{
  cudaFree(indexEC);
  cudaFree(nearInput);
  cudaFree(farInput);
}
