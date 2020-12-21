// Copyright 2020, Tang Yin, Nanjing University of Science and Technology

#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/init.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "liso/CeresSolveFactor.hpp"
#include "liso/liso_preprocess_thread.h"

//全局变量

//参数

using cv::Mat;

//话题
static ros::Publisher pubPointCloudWithFeature;
static ros::Publisher pubLaserOdometry;
static ros::Publisher pubLeftImageWithFeature;
static ros::Publisher pubCornerPointsSharp;
static ros::Publisher pubCornerPointsLessSharp;
static ros::Publisher pubSurfPointsFlat;
static ros::Publisher pubSurfPointsLessFlat;
static ros::Publisher pubCameraPointsCloud;

//特征提取

static cv::Mat lidar_to_base_pose;
static cv::Point2i image_size;

//激光里程计

//主线程的缓存变量
static Eigen::Quaterniond q_w_curr_buf;
static Eigen::Vector3d t_w_curr_buf;
static std::mutex main_thread_mutex;

//预处理线程
PreprocessThread *preporcess_thread;

//里程计线程的缓存变量
static cv::Mat descriptors_left_buf, descriptors_right_buf;
static std::vector<cv::Point2f> imgPoints_left_buf;
static std::vector<cv::Point2f> imgPoints_right_buf;
static std::vector<cv::DMatch> good_matches_stereo_buf;
static std::vector<cv::Point3d> good_points_3d_buf;
static pcl::PointCloud<PointType> cornerPointsSharp_buf;
static pcl::PointCloud<PointType> cornerPointsLessSharp_buf;
static pcl::PointCloud<PointType> surfPointsFlat_buf;
static pcl::PointCloud<PointType> surfPointsLessFlat_buf;
static bool is_odometry_thread_ready;
static std::mutex odometry_thread_mutex;

inline cv::Scalar get_color(float depth) {
  static Accumulator<float> depth_range(50);
  if (depth < 4 * depth_range.mean() && depth > -2 * depth_range.mean())
    depth_range.addDataValue(depth);
  float up_th = 2 * depth_range.mean(), low_th = 0.f, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// 从点云中间分割平面点和边缘点（锋利点）

// 把点转环到上一帧的坐标系上
void TransformToStart(PointType const *const pi, PointType *const po,
                      const Eigen::Quaterniond &q_last_curr,
                      const Eigen::Vector3d &t_last_curr) {
  // interpolation ratio
  double s;
  if (DISTORTION)
    s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
  else
    s = 1.0;
  // s = 1;
  Eigen::Quaterniond q_point_last =
      Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
  Eigen::Vector3d t_point_last = s * t_last_curr;
  Eigen::Vector3d point(pi->x, pi->y, pi->z);
  Eigen::Vector3d un_point = q_point_last * point + t_point_last;

  po->x = un_point.x();
  po->y = un_point.y();
  po->z = un_point.z();
  po->intensity = pi->intensity;

  po->normal_x = pi->normal_x;
  po->normal_y = pi->normal_z;
  po->normal_z = pi->normal_z;
  po->curvature = pi->curvature;
}

void CvPointTransform(const cv::Point3d &pi, cv::Point3d &po,
                      const Eigen::Quaterniond &q_point,
                      const Eigen::Vector3d &t_point) {
  Eigen::Vector3d point(pi.x, pi.y, pi.z);
  Eigen::Vector3d un_point = q_point * point + t_point;

  po.x = un_point.x();
  po.y = un_point.y();
  po.z = un_point.z();
}

// 将激光点添加到观测列表
// void addMatchPointToViews(const std::vector<cv::Point2f> &points_1,
//                           const std::vector<cv::Point2f> &points_2,
//                           const cv::Mat &descriptors_1,
//                           const std::vector<cv::Point3d> &points_3d,
//                           const Eigen::Quaterniond &q_w_curr,
//                           const Eigen::Vector3d &t_w_curr,
//                           cv::Mat &descriptorsInMap, int camera_idx,
//                           std::vector<CameraView> &camera_views,
//                           std::vector<Eigen::Vector3d> &points_3d_maps) {
//   std::cout << "addMatchPointToViews() start." << std::endl;
//   //描述子和观察表中的描述子进行匹配
//   std::vector<bool> choosen_flag;
//   choosen_flag.resize(points_1.size(), false);
//   std::vector<cv::DMatch> matches_map;
//   CameraView view1, view2;
//   view1.camera_idx = camera_idx;
//   view2.camera_idx = camera_idx + 1;
//   if (camera_idx > 0) {
//     robustMatch(descriptors_1, descriptorsInMap, matches_map);
//     //根据匹配列表为已有特征点添加观测
//     for (size_t i = 0; i < matches_map.size(); i++) {
//       if (points_1[matches_map[i].queryIdx].x >= image_size.x ||
//           points_1[matches_map[i].queryIdx].y >= image_size.y ||
//           points_1[matches_map[i].queryIdx].x <= 1 ||
//           points_1[matches_map[i].queryIdx].y <= 1)
//         continue;

//       if (points_2[matches_map[i].queryIdx].x >= image_size.x ||
//           points_2[matches_map[i].queryIdx].y >= image_size.y ||
//           points_2[matches_map[i].queryIdx].x <= 1 ||
//           points_2[matches_map[i].queryIdx].y <= 1)
//         continue;

//       view1.point_idx = matches_map[i].trainIdx;
//       view1.observation_x = points_1[matches_map[i].queryIdx].x;
//       view1.observation_y = points_1[matches_map[i].queryIdx].y;
//       camera_views.push_back(view1);

//       view2.point_idx = matches_map[i].trainIdx;
//       view2.observation_x = points_2[matches_map[i].queryIdx].x;
//       view2.observation_y = points_2[matches_map[i].queryIdx].y;
//       // camera_views.push_back(view1);

//       descriptorsInMap.row(matches_map[i].trainIdx) =
//           descriptors_1.row(matches_map[i].queryIdx);
//       choosen_flag[i] = true;
//     }
//   }
//   //根据未匹配列表添加新的特征点和观测
//   for (size_t i = 0; i < choosen_flag.size(); i++) {
//     //是否已匹配
//     if (choosen_flag[i]) continue;

//     cv::Point3d point = points_3d[i];
//     cv::Mat pt_trans = left_camera_to_base_pose *
//                        (cv::Mat_<double>(4, 1) << point.x, point.y, point.z,
//                        1);
//     double depth = pt_trans.at<double>(2, 0);

//     if (depth > stereoDistanceThresh || depth < 0.54) continue;

//     if (points_1[i].x >= image_size.x || points_1[i].y >= image_size.y ||
//         points_1[i].x <= 1 || points_1[i].y <= 1)
//       continue;

//     if (points_2[i].x >= image_size.x || points_2[i].y >= image_size.y ||
//         points_2[i].x <= 1 || points_2[i].y <= 1)
//       continue;

//     view1.point_idx = descriptorsInMap.rows;
//     view1.observation_x = points_1[i].x;
//     view1.observation_y = points_1[i].y;
//     camera_views.push_back(view1);

//     view2.point_idx = descriptorsInMap.rows;
//     view2.observation_x = points_2[i].x;
//     view2.observation_y = points_2[i].y;
//     // camera_views.push_back(view2);

//     descriptorsInMap.push_back(descriptors_1.row(i));

//     //把点转换到地图坐标系
//     CvPointTransform(point, point, q_w_curr, t_w_curr);
//     //把点添加到地图特征点集中
//     Eigen::Vector3d point_temp;
//     point_temp.x() = point.x;
//     point_temp.y() = point.y;
//     point_temp.z() = point.z;
//     points_3d_maps.push_back(point_temp);
//   }
//   std::cout << "addMatchPointToViews() end." << std::endl;
// }

void addPoints3DToCloud(
    const std::vector<cv::Point3d> &points_3d,
    const pcl::PointCloud<PointType>::Ptr &points_3d_clouds) {
  for (auto point : points_3d) {
    PointType pointSel;
    pointSel.x = point.x;
    pointSel.y = point.y;
    pointSel.z = point.z;
    points_3d_clouds->push_back(pointSel);
  }
}

// 传感器消息同步处理
void callbackHandle(const sensor_msgs::ImageConstPtr &left_image_msg,
                    const sensor_msgs::CameraInfoConstPtr &left_cam_info_msg,
                    const sensor_msgs::ImageConstPtr &right_image_msg,
                    const sensor_msgs::CameraInfoConstPtr &right_cam_info_msg,
                    const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg) {
  ROS_INFO("callbackHandle Start\n");

  preporcess_thread->set_left_image_msg_buf(*left_image_msg);
  preporcess_thread->set_left_cam_info_msg_buf(*left_cam_info_msg);
  preporcess_thread->set_right_image_msg_buf(*right_image_msg);
  preporcess_thread->set_right_cam_info_msg_buf(*right_cam_info_msg);
  preporcess_thread->set_point_cloud_msg_buf(*point_cloud_msg);
  preporcess_thread->set_is_preprocess_thread_ready(true);

  main_thread_mutex.lock();
  Eigen::Quaterniond q_w_curr_new = q_w_curr_buf;
  Eigen::Vector3d t_w_curr_new = t_w_curr_buf;
  main_thread_mutex.unlock();

  tf::Transform tf_left_camera_to_base_pose(tf::Quaternion(0.5, -0.5, 0.5, 0.5),
                                            tf::Vector3(0, -0.08, -0.27));
  tf::Transform tf_base_to_world_pose(
      tf::Quaternion(q_w_curr_new.x(), q_w_curr_new.y(), q_w_curr_new.z(),
                     q_w_curr_new.w()),
      tf::Vector3(t_w_curr_new.x(), t_w_curr_new.y(), t_w_curr_new.z()));
  std::vector<tf::StampedTransform> tf_list;
  tf_list.push_back(tf::StampedTransform(tf_left_camera_to_base_pose,
                                         ros::Time::now(), "kitti_base_link",
                                         "kitti_velo_link"));
  tf_list.push_back(tf::StampedTransform(tf_base_to_world_pose,
                                         ros::Time::now(), "kitti_world_link",
                                         "kitti_base_link"));
  // （变成注释tf树每个节点只能有一个父节点）
  static tf::TransformBroadcaster broadcaster;
  broadcaster.sendTransform(tf_list);

  ROS_INFO("callbackHandle Stop\n");
}

void odometryThread() __attribute__((noreturn));
void mappingThread() __attribute__((noreturn));

void odometryThread() {
  while (1) {
    //-- 第0步：线程休眠,时长为线程运行时常
    static auto start_time = std::chrono::high_resolution_clock::now();
    static auto end_time = std::chrono::high_resolution_clock::now();
    static std::chrono::duration<double, std::milli> elapsed_duration =
        end_time - start_time;
    std::this_thread::sleep_for(elapsed_duration);

    //-- 第1步：判断是否有新消息
    odometry_thread_mutex.lock();
    bool is_odometry_thread_updated = is_odometry_thread_ready;
    odometry_thread_mutex.unlock();
    if (!is_odometry_thread_updated) {
      continue;
    }

    ROS_INFO("odometryThread Start\n");
    start_time = std::chrono::high_resolution_clock::now();

    //-- 第2步：从预处理线程读取数据
    odometry_thread_mutex.lock();
    cv::Mat descriptors_left_curr = descriptors_left_buf;
    cv::Mat descriptors_right_curr = descriptors_right_buf;
    std::vector<cv::Point2f> imgPoints_left_curr = imgPoints_left_buf;
    std::vector<cv::Point2f> imgPoints_right_curr = imgPoints_right_buf;
    std::vector<cv::DMatch> good_matches_stereo_curr = good_matches_stereo_buf;
    std::vector<cv::Point3d> good_points_3d_curr = good_points_3d_buf;
    pcl::PointCloud<PointType> cornerPointsSharp_curr = cornerPointsSharp_buf;
    pcl::PointCloud<PointType> cornerPointsLessSharp_curr =
        cornerPointsLessSharp_buf;
    pcl::PointCloud<PointType> surfPointsFlat_curr = surfPointsFlat_buf;
    pcl::PointCloud<PointType> surfPointsLessFlat_curr = surfPointsLessFlat_buf;
    is_odometry_thread_ready = false;
    odometry_thread_mutex.unlock();

    static cv::Mat descriptors_left_last, descriptors_right_last;
    static std::vector<cv::Point2f> imgPoints_left_last, imgPoints_right_last;
    static std::vector<cv::DMatch> good_matches_stereo_last;
    static std::vector<cv::Point3d> good_points_3d_last;
    static pcl::PointCloud<PointType> cornerPointsSharp_last;
    static pcl::PointCloud<PointType> cornerPointsLessSharp_last;
    static pcl::PointCloud<PointType> surfPointsFlat_last;
    static pcl::PointCloud<PointType> surfPointsLessFlat_last;

    static bool is_odometry_thread_init = false;

    static std::vector<Eigen::Quaterniond> qlist_map_curr;
    static std::vector<Eigen::Quaterniond> qlist_last_curr;
    static std::vector<Eigen::Vector3d> tlist_map_curr;
    static std::vector<Eigen::Vector3d> tlist_last_curr;

    static Eigen::Quaterniond q_last_curr(1, 0, 0, 0);
    static Eigen::Vector3d t_last_curr(0, 0, 0);
    static Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
    static Eigen::Vector3d t_w_curr(0, 0, 0);

    static Eigen::Quaterniond q_identity(1, 0, 0, 0);
    static Eigen::Vector3d t_identity(0, 0, 0);

    static Eigen::Quaterniond q_lidar_to_pose(0.5, 0.5, -0.5, 0.5);
    static Eigen::Vector3d t_lidar_to_pose(0., -0.08, -0.27);
    static Eigen::Quaterniond q_pose_to_lidar = q_lidar_to_pose.inverse();
    static Eigen::Vector3d t_pose_to_lidar =
        t_identity - q_pose_to_lidar * t_lidar_to_pose;

    // //-- 第2.1步：与上一帧的视觉进行匹配
    // std::vector<cv::DMatch> matches_left, matches_right;
    // robustMatch(descriptors_left_curr, descriptors_left_last, matches_left);
    // robustMatch(descriptors_right_curr, descriptors_right_last,
    // matches_right);

    //-- 第3步：判断是否初始化
    if (!is_odometry_thread_init) {
      is_odometry_thread_init = true;
      qlist_map_curr.push_back(q_w_curr);
      qlist_last_curr.push_back(q_last_curr);
      tlist_map_curr.push_back(t_w_curr);
      tlist_last_curr.push_back(t_last_curr);
    } else {
      pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp_last_ptr(
          new pcl::PointCloud<PointType>(cornerPointsLessSharp_last));
      pcl::PointCloud<PointType>::Ptr surfPointsLessFlat_last_ptr(
          new pcl::PointCloud<PointType>(surfPointsLessFlat_last));
      pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(
          new pcl::KdTreeFLANN<PointType>());
      pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(
          new pcl::KdTreeFLANN<PointType>());
      kdtreeCornerLast->setInputCloud(cornerPointsLessSharp_last_ptr);
      kdtreeSurfLast->setInputCloud(surfPointsLessFlat_last_ptr);

      for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter) {
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization =
            new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);

        problem.AddParameterBlock(q_last_curr.coeffs().data(), 4,
                                  q_parameterization);
        problem.AddParameterBlock(t_last_curr.data(), 3);

        PointType pointSel;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        Eigen::Quaterniond q_temp(1, 0, 0, 0);
        Eigen::Vector3d t_temp(0, 0, 0);

        t_temp = t_pose_to_lidar + q_pose_to_lidar * t_last_curr;
        q_temp = q_pose_to_lidar * q_last_curr;
        t_temp = t_temp + q_temp * t_lidar_to_pose;
        q_temp = q_temp * q_lidar_to_pose;

        // 建立边缘特征约束
        int cornerPointsSharpNum = cornerPointsSharp_curr.points.size();
        int corner_correspondence = 0;
        for (int i = 0; i < cornerPointsSharpNum; ++i) {
          TransformToStart(&(cornerPointsSharp_curr.points[i]), &pointSel,
                           q_temp, t_temp);
          kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                           pointSearchSqDis);

          int closestPointInd = -1, minPointInd2 = -1;
          if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
            closestPointInd = pointSearchInd[0];
            int closestPointScanID = int(
                cornerPointsLessSharp_last.points[closestPointInd].intensity);

            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
            // search in the direction
            // of increasing scan line
            for (int j = closestPointInd + 1;
                 j < (int)cornerPointsLessSharp_last.points.size(); ++j) {
              // if in the same scan
              // line, continue
              if (int(cornerPointsLessSharp_last.points[j].intensity) <=
                  closestPointScanID)
                continue;

              // if not in nearby scans,
              // end the loop
              if (int(cornerPointsLessSharp_last.points[j].intensity) >
                  (closestPointScanID + NEARBY_SCAN))
                break;

              double pointSqDis =
                  (cornerPointsLessSharp_last.points[j].x - pointSel.x) *
                      (cornerPointsLessSharp_last.points[j].x - pointSel.x) +
                  (cornerPointsLessSharp_last.points[j].y - pointSel.y) *
                      (cornerPointsLessSharp_last.points[j].y - pointSel.y) +
                  (cornerPointsLessSharp_last.points[j].z - pointSel.z) *
                      (cornerPointsLessSharp_last.points[j].z - pointSel.z);

              if (pointSqDis < minPointSqDis2) {
                // find nearer point
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }

            // search in the direction
            // of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j) {
              // if in the same scan
              // line, continue
              if (int(cornerPointsLessSharp_last.points[j].intensity) >=
                  closestPointScanID)
                continue;

              // if not in nearby scans,
              // end the loop
              if (int(cornerPointsLessSharp_last.points[j].intensity) <
                  (closestPointScanID - NEARBY_SCAN))
                break;

              double pointSqDis =
                  (cornerPointsLessSharp_last.points[j].x - pointSel.x) *
                      (cornerPointsLessSharp_last.points[j].x - pointSel.x) +
                  (cornerPointsLessSharp_last.points[j].y - pointSel.y) *
                      (cornerPointsLessSharp_last.points[j].y - pointSel.y) +
                  (cornerPointsLessSharp_last.points[j].z - pointSel.z) *
                      (cornerPointsLessSharp_last.points[j].z - pointSel.z);

              if (pointSqDis < minPointSqDis2) {
                // find nearer point
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
          if (minPointInd2 >= 0)  // both
                                  // closestPointInd and
                                  // minPointInd2 is
                                  // valid
          {
            Eigen::Vector3d curr_point(cornerPointsSharp_curr.points[i].x,
                                       cornerPointsSharp_curr.points[i].y,
                                       cornerPointsSharp_curr.points[i].z);
            Eigen::Vector3d last_point_a(
                cornerPointsLessSharp_last.points[closestPointInd].x,
                cornerPointsLessSharp_last.points[closestPointInd].y,
                cornerPointsLessSharp_last.points[closestPointInd].z);
            Eigen::Vector3d last_point_b(
                cornerPointsLessSharp_last.points[minPointInd2].x,
                cornerPointsLessSharp_last.points[minPointInd2].y,
                cornerPointsLessSharp_last.points[minPointInd2].z);

            double s;
            if (DISTORTION)
              s = (cornerPointsSharp_curr.points[i].intensity -
                   int(cornerPointsSharp_curr.points[i].intensity)) /
                  SCAN_PERIOD;
            else
              s = 1.0;
            ceres::CostFunction *cost_function =
                LidarEdgeFactor2::Create(curr_point, last_point_a, last_point_b,
                                         s, t_lidar_to_pose, q_lidar_to_pose);
            problem.AddResidualBlock(cost_function, loss_function,
                                     q_last_curr.coeffs().data(),
                                     t_last_curr.data());
            corner_correspondence++;
          }
        }

        // find correspondence for plane
        // features
        int surfPointsFlatNum = cornerPointsSharp_curr.points.size();
        int plane_correspondence = 0;
        for (int i = 0; i < surfPointsFlatNum; ++i) {
          TransformToStart(&(cornerPointsSharp_curr.points[i]), &pointSel,
                           q_temp, t_temp);
          kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                         pointSearchSqDis);

          int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
          if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
            closestPointInd = pointSearchInd[0];

            // get closest point's scan
            // ID
            int closestPointScanID =
                int(surfPointsLessFlat_last.points[closestPointInd].intensity);
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD,
                   minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

            // search in the direction
            // of increasing scan line
            for (int j = closestPointInd + 1;
                 j < (int)surfPointsLessFlat_last.points.size(); ++j) {
              // if not in nearby scans,
              // end the loop
              if (int(surfPointsLessFlat_last.points[j].intensity) >
                  (closestPointScanID + NEARBY_SCAN))
                break;

              double pointSqDis =
                  (surfPointsLessFlat_last.points[j].x - pointSel.x) *
                      (surfPointsLessFlat_last.points[j].x - pointSel.x) +
                  (surfPointsLessFlat_last.points[j].y - pointSel.y) *
                      (surfPointsLessFlat_last.points[j].y - pointSel.y) +
                  (surfPointsLessFlat_last.points[j].z - pointSel.z) *
                      (surfPointsLessFlat_last.points[j].z - pointSel.z);

              // if in the same or lower
              // scan line
              if (int(surfPointsLessFlat_last.points[j].intensity) <=
                      closestPointScanID &&
                  pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
              // if in the higher scan
              // line
              else if (int(surfPointsLessFlat_last.points[j].intensity) >
                           closestPointScanID &&
                       pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }

            // search in the direction
            // of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j) {
              // if not in nearby scans,
              // end the loop
              if (int(surfPointsLessFlat_last.points[j].intensity) <
                  (closestPointScanID - NEARBY_SCAN))
                break;

              double pointSqDis =
                  (surfPointsLessFlat_last.points[j].x - pointSel.x) *
                      (surfPointsLessFlat_last.points[j].x - pointSel.x) +
                  (surfPointsLessFlat_last.points[j].y - pointSel.y) *
                      (surfPointsLessFlat_last.points[j].y - pointSel.y) +
                  (surfPointsLessFlat_last.points[j].z - pointSel.z) *
                      (surfPointsLessFlat_last.points[j].z - pointSel.z);

              // if in the same or
              // higher scan line
              if (int(surfPointsLessFlat_last.points[j].intensity) >=
                      closestPointScanID &&
                  pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              } else if (int(surfPointsLessFlat_last.points[j].intensity) <
                             closestPointScanID &&
                         pointSqDis < minPointSqDis3) {
                // find nearer point
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }

            if (minPointInd2 >= 0 && minPointInd3 >= 0) {
              Eigen::Vector3d curr_point(cornerPointsSharp_curr.points[i].x,
                                         cornerPointsSharp_curr.points[i].y,
                                         cornerPointsSharp_curr.points[i].z);
              Eigen::Vector3d last_point_a(
                  surfPointsLessFlat_last.points[closestPointInd].x,
                  surfPointsLessFlat_last.points[closestPointInd].y,
                  surfPointsLessFlat_last.points[closestPointInd].z);
              Eigen::Vector3d last_point_b(
                  surfPointsLessFlat_last.points[minPointInd2].x,
                  surfPointsLessFlat_last.points[minPointInd2].y,
                  surfPointsLessFlat_last.points[minPointInd2].z);
              Eigen::Vector3d last_point_c(
                  surfPointsLessFlat_last.points[minPointInd3].x,
                  surfPointsLessFlat_last.points[minPointInd3].y,
                  surfPointsLessFlat_last.points[minPointInd3].z);

              double s;
              if (DISTORTION)
                s = (cornerPointsSharp_curr.points[i].intensity -
                     int(surfPointsLessFlat_last.points[i].intensity)) /
                    SCAN_PERIOD;
              else
                s = 1.0;
              ceres::CostFunction *cost_function = LidarPlaneFactor2::Create(
                  curr_point, last_point_a, last_point_b, last_point_c, s,
                  t_lidar_to_pose, q_lidar_to_pose);
              problem.AddResidualBlock(cost_function, loss_function,
                                       q_last_curr.coeffs().data(),
                                       t_last_curr.data());
              plane_correspondence++;
            }
          }
        }

        printf(
            "coner_correspondance %d, "
            "plane_correspondence %d "
            "\n",
            corner_correspondence, plane_correspondence);

        if ((corner_correspondence + plane_correspondence) < 10) {
          printf(
              "less correspondence! "
              "************************"
              "************************"
              "*\n");
        }
        static int max_num_iter = 4;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = max_num_iter;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double solve_efficiency =
            (1. - summary.final_cost / summary.initial_cost) / max_num_iter;
        if (solve_efficiency > 0.05) max_num_iter = max_num_iter + 1;
        if (solve_efficiency < 0.01) max_num_iter = max_num_iter - 1;
        if (max_num_iter < 2) max_num_iter = 2;
        printf(
            "solve_efficiency = %f %%, "
            "max_num_iter = %d.\n",
            solve_efficiency * 100, max_num_iter);
      }
    }
    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
    q_w_curr = q_w_curr * q_last_curr;

    main_thread_mutex.lock();
    q_w_curr_buf = q_w_curr;
    t_w_curr_buf = t_w_curr;
    main_thread_mutex.unlock();

    //保存到位姿列表和唯一列表
    //用于设置初始位姿
    qlist_map_curr.push_back(q_w_curr);
    tlist_map_curr.push_back(t_w_curr);
    //用于添加约束
    qlist_last_curr.push_back(q_last_curr);
    tlist_last_curr.push_back(t_last_curr);

    descriptors_left_last = descriptors_left_curr;
    descriptors_right_last = descriptors_right_curr;
    imgPoints_left_last = imgPoints_left_curr;
    imgPoints_right_last = imgPoints_right_curr;
    good_matches_stereo_last = good_matches_stereo_curr;
    good_points_3d_last = good_points_3d_curr;
    cornerPointsSharp_last = cornerPointsSharp_curr;
    cornerPointsLessSharp_last = cornerPointsLessSharp_curr;
    surfPointsFlat_last = surfPointsFlat_curr;
    surfPointsLessFlat_last = surfPointsLessFlat_curr;

    ROS_INFO("odometryThread End\n");
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_duration = end_time - start_time;
  }
}

// 主函数
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

  std::string left_image_with_feature_pub;
  std::string point_cloud_with_feature_pub;
  std::string conner_points_pub;
  std::string conner_points_less_pub;
  std::string surf_points_pub;
  std::string surf_points_less_pub;
  std::string laser_odometry_pub;
  std::string camera_points_clouds_pub;
  nh.param<std::string>("left_image_with_feature_pub",
                        left_image_with_feature_pub,
                        "/left_image_with_feature_pub");
  nh.param<std::string>("point_cloud_with_feature_pub",
                        point_cloud_with_feature_pub,
                        "/point_cloud_with_feature_pub");
  nh.param<std::string>("conner_points_pub", conner_points_pub,
                        "/conner_points_pub");
  nh.param<std::string>("conner_points_less_pub", conner_points_less_pub,
                        "/conner_points_less_pub");
  nh.param<std::string>("surf_points_pub", surf_points_pub, "/surf_points_pub");
  nh.param<std::string>("surf_points_less_pub", surf_points_less_pub,
                        "/surf_points_less_pub");
  nh.param<std::string>("laser_odometry_pub", laser_odometry_pub,
                        "/laser_odometry_pub");
  nh.param<std::string>("camera_points_clouds_pub", camera_points_clouds_pub,
                        "/camera_points_clouds_pub");

  // 雷达参数
  std::string lidarType;
  nh.param<std::string>("lidar_type", lidarType, "HDL-64E");

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

  // 发布话题
  pubLeftImageWithFeature =
      nh.advertise<sensor_msgs::Image>(left_image_with_feature_pub, 10);
  pubPointCloudWithFeature =
      nh.advertise<sensor_msgs::PointCloud2>(point_cloud_with_feature_pub, 10);
  pubCornerPointsSharp =
      nh.advertise<sensor_msgs::PointCloud2>(conner_points_pub, 10);
  pubCornerPointsLessSharp =
      nh.advertise<sensor_msgs::PointCloud2>(conner_points_less_pub, 10);
  pubSurfPointsFlat =
      nh.advertise<sensor_msgs::PointCloud2>(surf_points_pub, 10);
  pubSurfPointsLessFlat =
      nh.advertise<sensor_msgs::PointCloud2>(surf_points_less_pub, 10);
  pubLaserOdometry = nh.advertise<nav_msgs::Odometry>(laser_odometry_pub, 10);
  pubCameraPointsCloud =
      nh.advertise<sensor_msgs::PointCloud2>(camera_points_clouds_pub, 10);

  // 传感器参数
  // 相机内参
  lidar_to_base_pose =
      (cv::Mat_<double>(3, 4) << 0, 0, 1, 0.27, -1, 0, 0, 0, 0, -1, 0, -0.08);

  preporcess_thread = new PreprocessThread(lidarType);
  // std::thread odometry_thread{odometryThread};
  // std::thread mapping_thread{ mappingThread };

  ros::spin();

  while (ros::ok()) {
    ros::spinOnce();
  }

  delete preporcess_thread;

  ROS_INFO("liso Stop\n");
}
