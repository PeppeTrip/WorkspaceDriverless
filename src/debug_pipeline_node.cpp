#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std::chrono_literals;

class DebugPipelineNode : public rclcpp::Node
{
public:
  DebugPipelineNode()
  : Node("debug_pipeline_node")
  {
    declare_parameter<std::string>("debug_input_topic", "/debug/lidar_points");
    declare_parameter<std::string>("clusters_topic", "/perception/newclusters");
    declare_parameter<std::string>("frame_id", "hesai_lidar");
    declare_parameter<double>("publish_hz", 5.0);
    declare_parameter<int>("expected_near_min", 1);
    declare_parameter<int>("expected_far_min", 1);
    declare_parameter<double>("status_timeout_s", 2.0);

    get_parameter("debug_input_topic", debug_input_topic_);
    get_parameter("clusters_topic", clusters_topic_);
    get_parameter("frame_id", frame_id_);
    get_parameter("expected_near_min", expected_near_min_);
    get_parameter("expected_far_min", expected_far_min_);
    get_parameter("status_timeout_s", status_timeout_s_);

    double publish_hz = 5.0;
    get_parameter("publish_hz", publish_hz);
    if (publish_hz <= 0.0) {
      publish_hz = 5.0;
    }

    debug_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(debug_input_topic_, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/debug/pipeline_status", 10);
    clusters_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      clusters_topic_,
      rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data),
      std::bind(&DebugPipelineNode::clustersCallback, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / publish_hz);
    publish_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&DebugPipelineNode::publishDebugCloud, this));

    health_timer_ = create_wall_timer(
      500ms,
      std::bind(&DebugPipelineNode::publishHealthStatus, this));

    RCLCPP_INFO(get_logger(), "Debug pipeline node started.");
    RCLCPP_INFO(get_logger(), "Publishing synthetic cloud on: %s", debug_input_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Listening cluster markers on: %s", clusters_topic_.c_str());
  }

  void publishStatus(const std::string & status)
  {
    std_msgs::msg::String msg;
    msg.data = status;
    status_pub_->publish(msg);
  }

private:
  void addCluster(pcl::PointCloud<pcl::PointXYZI> & cloud,
                  float cx, float cy, float cz,
                  float dx, float dy, float dz,
                  int points)
  {
    for (int i = 0; i < points; ++i) {
      const float fx = static_cast<float>((i % 4) - 1.5f) * dx;
      const float fy = static_cast<float>(((i / 4) % 4) - 1.5f) * dy;
      const float fz = static_cast<float>((i % 3) - 1.0f) * dz;

      pcl::PointXYZI p;
      p.x = cx + fx;
      p.y = cy + fy;
      p.z = cz + fz;
      p.intensity = 1.0f;
      cloud.points.push_back(p);
    }
  }

  void publishDebugCloud()
  {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.header.frame_id = frame_id_;
    cloud.height = 1;

    // Near cone-like clusters
    addCluster(cloud, 2.0f, 1.0f, 0.25f, 0.06f, 0.06f, 0.04f, 20);
    addCluster(cloud, 3.0f, -1.0f, 0.28f, 0.06f, 0.06f, 0.04f, 20);

    // Far cone-like clusters
    addCluster(cloud, 6.5f, 1.5f, 0.30f, 0.07f, 0.07f, 0.05f, 22);
    addCluster(cloud, 7.0f, -1.4f, 0.27f, 0.07f, 0.07f, 0.05f, 22);

    // Sparse noise
    for (int i = 0; i < 12; ++i) {
      pcl::PointXYZI p;
      p.x = 10.0f + 0.2f * static_cast<float>(i);
      p.y = -3.0f + 0.1f * static_cast<float>(i % 5);
      p.z = -0.4f;
      p.intensity = 0.1f;
      cloud.points.push_back(p);
    }

    cloud.width = static_cast<std::uint32_t>(cloud.points.size());

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = frame_id_;
    msg.header.stamp = now();

    debug_pub_->publish(msg);
    ++published_msgs_;

    if (published_msgs_ % 10 == 0) {
      RCLCPP_INFO(get_logger(), "Published %u debug clouds", published_msgs_);
    }
  }

  void clustersCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    std::size_t near_count = 0;
    std::size_t far_count = 0;

    for (const auto & marker : msg->markers) {
      if (marker.ns == "cones_near") {
        near_count = marker.points.size();
      } else if (marker.ns == "cones_far") {
        far_count = marker.points.size();
      }
    }

    last_cluster_msg_time_ = now();

    RCLCPP_INFO(get_logger(), "Cluster output received -> near=%zu far=%zu total=%zu",
                near_count, far_count, near_count + far_count);

    const bool pass =
      static_cast<int>(near_count) >= expected_near_min_ &&
      static_cast<int>(far_count) >= expected_far_min_;

    if (pass) {
      publishStatus("OK: cluster output meets expected minimum near/far counts");
    } else {
      publishStatus("WARN: cluster output below expected near/far counts");
    }
  }

  void publishHealthStatus()
  {
    if (published_msgs_ == 0) {
      publishStatus("BOOT: waiting to publish first debug cloud");
      return;
    }

    const double elapsed = (now() - last_cluster_msg_time_).seconds();
    if (elapsed > status_timeout_s_) {
      publishStatus("ERROR: no cluster output received within timeout");
    }
  }

  std::string debug_input_topic_;
  std::string clusters_topic_;
  std::string frame_id_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr clusters_sub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr health_timer_;

  int expected_near_min_ = 1;
  int expected_far_min_ = 1;
  double status_timeout_s_ = 2.0;
  rclcpp::Time last_cluster_msg_time_{0, 0, RCL_ROS_TIME};

  std::uint32_t published_msgs_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DebugPipelineNode>());
  rclcpp::shutdown();
  return 0;
}
