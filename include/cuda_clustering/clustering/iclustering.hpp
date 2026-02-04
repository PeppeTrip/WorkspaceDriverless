#pragma once

#include <memory>

// namespace visualization_msgs { namespace msg { class Marker; } }

#include <visualization_msgs/msg/marker.hpp>

#include "cuda_clustering/clustering/cluster_filtering/icluster_filtering.hpp"

class IClustering
{
public:
  virtual ~IClustering() = default;

  virtual void extractClusters(
      float* input,
      unsigned int inputSize,
      float* outputEC,
      std::shared_ptr<visualization_msgs::msg::Marker> cones) = 0;

  virtual void extractClustersNearFar(
      float* input,
      unsigned int inputSize,
      float* scratch,
      std::shared_ptr<visualization_msgs::msg::Marker> conesNear,
      std::shared_ptr<visualization_msgs::msg::Marker> conesFar,
      float range_split_m) = 0;

  virtual void getInfo() = 0;

  IClusterFiltering* filter = nullptr;
};