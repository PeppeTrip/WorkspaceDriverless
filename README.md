# Clustering Node

## Overview
ROS2 node for performing GPU-based point cloud clustering and optional filtering or segmentation.

## Features
- Voxel-based downsampling
- Dimension filtering
- Plane segmentation
- Configurable via YAML parameters

## Requirements
- CUDA toolkit
- PCL
- ROS2 Humble

## Configuration
- Enables or disables Z filtering, segmentation, voxel grid.
- Min/max cluster size, voxel sizes, and distance thresholds.

## Build
cd cuda_clustering
colcon build

## Usage
Edit parameters in config/config.yaml

## Run
```bash
cd ~/ros2_ws
source ./install/setup.bash
ros2 launch clustering cuda_clustering_launch.py
```

## TODO
1. FIX: The clustering stops working while the filter on Z is activated 
2. ADD: Parameters fixed at compile time to recude "if" comparisons


## GPU/Thrust smoke test
Use this minimal node to verify that CUDA + Thrust are working before running the full perception pipeline.

Run directly:
```bash
ros2 run clustering gpu_smoke_node
```

Or with launch:
```bash
ros2 launch clustering gpu_smoke_launch.py
```

The node checks:
- CUDA device visibility (`cudaGetDeviceCount`)
- Thrust `transform` + `reduce` on GPU
- Thrust `copy_if` near/far partition behavior


## Debug node/topic for end-to-end validation
If you want to validate the full clustering pipeline (not only CUDA/Thrust availability), use the debug pipeline node.

It publishes a synthetic point cloud on `/debug/lidar_points` and listens to `/perception/newclusters` to report detected near/far cones.
It also publishes a simple debug status topic on `/debug/pipeline_status` (`std_msgs/String`) with `OK/WARN/ERROR` style messages.

Run both nodes together:
```bash
ros2 launch clustering debug_pipeline_launch.py
```

Expected behavior:
- `debug_pipeline_node` logs periodic cloud publication
- `clustering_node` processes `/debug/lidar_points`
- `debug_pipeline_node` logs marker counts from `/perception/newclusters`
- `/debug/pipeline_status` reports whether cluster output is arriving and whether near/far minimum counts are met
