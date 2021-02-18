// Copyright 2021, Tang Yin, Nanjing University of Science and Technology

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

// 传感器消息同步处理
void callbackHandle(const sensor_msgs::ImageConstPtr &left_image_msg,
                    const sensor_msgs::CameraInfoConstPtr &left_cam_info_msg,
                    const sensor_msgs::ImageConstPtr &right_image_msg,
                    const sensor_msgs::CameraInfoConstPtr &right_cam_info_msg,
                    const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg) {
  ROS_INFO("callbackHandle Start\n");

  ROS_INFO("callbackHandle Stop\n");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "liso_node");
  ros::NodeHandle nh;

  ROS_INFO("liso Start.\n");

  std::string left_image_sub;
  std::string left_camera_info_sub;
  std::string right_image_sub;
  std::string right_camera_info_sub;
  std::string point_cloud_sub;
  nh.param<std::string>("left_image_sub", left_image_sub,
                        "/kitti/camera_color_left/image_rect");
  nh.param<std::string>("left_camera_info_sub", left_camera_info_sub,
                        "/kitti/camera_color_left/camera_info");
  nh.param<std::string>("right_image_sub", right_image_sub,
                        "/kitti/camera_color_right/image_rect");
  nh.param<std::string>("right_camera_info_sub", right_camera_info_sub,
                        "/kitti/camera_color_right/camera_info");
  nh.param<std::string>("point_cloud_sub", point_cloud_sub,
                        "/kitti/velo/pointcloud");

  std::string left_image_with_feature_pub = "/left_image_with_feature_pub";
  std::string point_cloud_with_feature_pub = "/point_cloud_with_feature_pub";
  std::string conner_points_pub = "/conner_points_pub";
  std::string conner_points_less_pub = "/conner_points_less_pub";
  std::string surf_points_pub = "/surf_points_pub";
  std::string surf_points_less_pub = "/surf_points_less_pub";
  std::string laser_odometry_pub = "/laser_odometry_pub";
  std::string camera_points_clouds_pub = "/camera_points_clouds_pub";
  std::string laser_cloud_corner_from_map_pub =
      "/laser_cloud_corner_from_map_pub";
  std::string laser_cloud_surf_from_map_pub = "/laser_cloud_surf_from_map_pub";

  // 雷达参数
  std::string cameraType;
  nh.param<std::string>("camera_type", cameraType, "KITTI-Camera");

  // 订阅话题
  message_filters::Subscriber<sensor_msgs::Image> subLeftImage(
      nh, left_image_sub, 10);
  message_filters::Subscriber<sensor_msgs::CameraInfo> subLeftCameraInfo(
      nh, left_camera_info_sub, 10);
  message_filters::Subscriber<sensor_msgs::Image> subRightImage(
      nh, right_image_sub, 10);
  message_filters::Subscriber<sensor_msgs::CameraInfo> subRightCameraInfo(
      nh, right_camera_info_sub, 10);
  message_filters::Subscriber<sensor_msgs::PointCloud2> subPointCloud(
      nh, point_cloud_sub, 10);
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo,
                                    sensor_msgs::Image, sensor_msgs::CameraInfo,
                                    sensor_msgs::PointCloud2>
      sync(subLeftImage, subLeftCameraInfo, subRightImage, subRightCameraInfo,
           subPointCloud, 10);
  sync.registerCallback(boost::bind(&callbackHandle, _1, _2, _3, _4, _5));

  ros::spin();

  ROS_INFO("liso Stop\n");
}
