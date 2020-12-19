// Copyright 2020, Tang Yin, Nanjing University of Science and Technology

#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
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

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "liso/CeresSolveFactor.hpp"
#include "liso/common.h"
#include "ros/init.h"

//全局变量

//参数

using cv::Mat;

static int N_SCAN = 16;
static float MAX_RANGE = 100;
static float MIN_RANGE = 0;
static float RES_RANGE = .3f;
static float MAX_ANGLE = 15;
static float MIN_ANGLE = -15;
static float RES_ANGLE = 2;

static double MINIMUM_RANGE = 1.57;

//话题
static ros::Publisher pubLeftImageWithFeature;
static ros::Publisher pubPointCloudWithFeature;
static ros::Publisher pubCornerPointsSharp;
static ros::Publisher pubCornerPointsLessSharp;
static ros::Publisher pubSurfPointsFlat;
static ros::Publisher pubSurfPointsLessFlat;
static ros::Publisher pubLaserOdometry;
static ros::Publisher pubCameraPointsCloud;

//特征提取
static cv::Ptr<cv::FeatureDetector> detector;
static cv::Ptr<cv::DescriptorExtractor> descriptor;
static cv::Ptr<cv::DescriptorMatcher> matcher;

//相机参数
static Eigen::Matrix3d left_camera_matrix;
static Eigen::Matrix3d right_camera_matrix;
static double stereoDistanceThresh;
static cv::Mat left_camera_to_base_pose;
static cv::Mat right_camera_to_base_pose;
static cv::Mat lidar_to_base_pose;
static cv::Point2i image_size;

//激光提取边沿点和平面点
static float cloudCurvature[400000];
static int cloudSortInd[400000];
static int cloudNeighborPicked[400000];
static int cloudLabel[400000];

//激光里程计

//消息预处理线程的缓存变量
static sensor_msgs::Image left_image_msg_buf;
static sensor_msgs::CameraInfo left_cam_info_msg_buf;
static sensor_msgs::Image right_image_msg_buf;
static sensor_msgs::CameraInfo right_cam_info_msg_buf;
static sensor_msgs::PointCloud2 point_cloud_msg_buf;
static std::mutex preprocess_thread_mutex;
static bool is_preprocess_thread_ready;

//里程计线程的缓存变量
static cv::Mat descriptors_left_buf, descriptors_right_buf;
static std::vector<cv::Point2f> imgPoints_left_buf, imgPoints_right_buf;
static std::vector<cv::DMatch> good_matches_stereo_buf;
static std::vector<cv::Point3d> good_points_3d_buf;
static pcl::PointCloud<PointType> cornerPointsSharp_buf;
static pcl::PointCloud<PointType> cornerPointsLessSharp_buf;
static pcl::PointCloud<PointType> surfPointsFlat_buf;
static pcl::PointCloud<PointType> surfPointsLessFlat_buf;
static std::mutex odometry_thread_mutex;
static bool is_odometry_thread_ready;

inline cv::Scalar get_color(float depth) {
  static Accumulator<float> depth_range(50);
  if (depth < 4 * depth_range.mean() && depth > -2 * depth_range.mean())
    depth_range.addDataValue(depth);
  float up_th = 2 * depth_range.mean(), low_th = 0.f, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  // printf("depth_range.mean() = %f\n", depth_range.mean());
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// 归一化齐次点
float normalizeHomogeneousPoints(const Mat &points_4d,
                                 std::vector<cv::Point3d> &points_3d) {
  Accumulator<float> scale_count;
  for (int i = 0; i < points_4d.cols; i++) {
    cv::Mat x = points_4d.col(i);
    scale_count.addDataValue(x.at<float>(3, 0));
    x /= x.at<float>(3, 0);  // 归一化
    cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points_3d.push_back(p);
  }
  return scale_count.mean();
}

// 转换像素点
cv::Point2f pixel2cam(const cv::Point2d &p, const Eigen::Matrix3d &K) {
  return cv::Point2f((p.x - K(0, 2)) / K(0, 0), (p.y - K(1, 2)) / K(1, 1));
}

//根据匹配获得匹配点
void filterMatchPoints(const std::vector<cv::KeyPoint> &keypoints_1,
                       const std::vector<cv::KeyPoint> &keypoints_2,
                       const std::vector<cv::DMatch> &matches,
                       std::vector<cv::Point2f> &points_1,
                       std::vector<cv::Point2f> &points_2) {
  points_1.clear();
  points_2.clear();
  for (int i = 0; i < (int)matches.size(); i++) {
    int query_idx = matches[i].queryIdx;
    int train_idx = matches[i].trainIdx;
    points_1.push_back(
        pixel2cam(keypoints_1[query_idx].pt, left_camera_matrix));
    points_2.push_back(
        pixel2cam(keypoints_2[train_idx].pt, right_camera_matrix));
  }
}

void filterUsableKeyPoints(
    const std::vector<cv::KeyPoint> &keypoints_1,
    const std::vector<cv::KeyPoint> &keypoints_2, const cv::Mat &descriptors_1,
    const cv::Mat &descriptors_2, const std::vector<cv::DMatch> &matches,
    const std::vector<cv::Point3d> &points_3d,
    std::vector<cv::Point2f> &imgPoints_left,
    std::vector<cv::Point2f> &imgPoints_right, cv::Mat &descriptors_left,
    cv::Mat &descriptors_right, std::vector<cv::DMatch> &good_matches,
    std::vector<cv::Point3d> &good_points_3d) {
  for (size_t i = 0; i < matches.size(); i++) {
    double depth = points_3d[i].z;
    if (depth < stereoDistanceThresh && depth > 0.54) {
      int query_idx = matches[i].queryIdx;
      int train_idx = matches[i].trainIdx;
      descriptors_left.push_back(descriptors_1.row(query_idx));
      descriptors_right.push_back(descriptors_2.row(query_idx));
      imgPoints_left.push_back(keypoints_1[query_idx].pt);
      imgPoints_right.push_back(keypoints_2[train_idx].pt);
      good_matches.push_back(matches[i]);
      good_points_3d.push_back(points_3d[i]);
    }
  }
}

//带正反检查的描述子匹配
void robustMatch(const cv::Mat &queryDescriptors,
                 const cv::Mat &trainDescriptors,
                 std::vector<cv::DMatch> &matches) {
  std::vector<cv::DMatch> matches_1, matches_2;
  matches.clear();
  matcher->match(queryDescriptors, trainDescriptors, matches_1);
  matcher->match(trainDescriptors, queryDescriptors, matches_2);

  for (int i = 0; i < (int)matches_1.size(); i++) {
    for (int j = 0; j < (int)matches_2.size(); j++) {
      if (matches_1[i].queryIdx == matches_2[j].trainIdx &&
          matches_2[j].queryIdx == matches_1[i].trainIdx) {
        matches.push_back(matches_1[i]);
        break;
      }
    }
  }
}

//去除指定半径内的点
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out, float thres) {
  if (&cloud_in != &cloud_out) {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  size_t j = 0;

  for (size_t i = 0; i < cloud_in.points.size(); ++i) {
    if (cloud_in.points[i].x * cloud_in.points[i].x +
            cloud_in.points[i].y * cloud_in.points[i].y +
            cloud_in.points[i].z * cloud_in.points[i].z <
        thres * thres)
      continue;
    cloud_out.points[j] = cloud_in.points[i];
    j++;
  }
  if (j != cloud_in.points.size()) {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<uint32_t>(j);
  cloud_out.is_dense = true;
}

// 计算点云各点的曲率
template <typename PointType>
void calculatePointCurvature(pcl::PointCloud<PointType> &laserCloudIn) {
  int cloudSize = laserCloudIn.points.size();
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffX = laserCloudIn.points[i - 5].x + laserCloudIn.points[i - 4].x +
                  laserCloudIn.points[i - 3].x + laserCloudIn.points[i - 2].x +
                  laserCloudIn.points[i - 1].x - 10 * laserCloudIn.points[i].x +
                  laserCloudIn.points[i + 1].x + laserCloudIn.points[i + 2].x +
                  laserCloudIn.points[i + 3].x + laserCloudIn.points[i + 4].x +
                  laserCloudIn.points[i + 5].x;
    float diffY = laserCloudIn.points[i - 5].y + laserCloudIn.points[i - 4].y +
                  laserCloudIn.points[i - 3].y + laserCloudIn.points[i - 2].y +
                  laserCloudIn.points[i - 1].y - 10 * laserCloudIn.points[i].y +
                  laserCloudIn.points[i + 1].y + laserCloudIn.points[i + 2].y +
                  laserCloudIn.points[i + 3].y + laserCloudIn.points[i + 4].y +
                  laserCloudIn.points[i + 5].y;
    float diffZ = laserCloudIn.points[i - 5].z + laserCloudIn.points[i - 4].z +
                  laserCloudIn.points[i - 3].z + laserCloudIn.points[i - 2].z +
                  laserCloudIn.points[i - 1].z - 10 * laserCloudIn.points[i].z +
                  laserCloudIn.points[i + 1].z + laserCloudIn.points[i + 2].z +
                  laserCloudIn.points[i + 3].z + laserCloudIn.points[i + 4].z +
                  laserCloudIn.points[i + 5].z;
    float curve = diffX * diffX + diffY * diffY + diffZ * diffZ;

    cloudCurvature[i] = curve;
    cloudSortInd[i] = i;
    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;

    laserCloudIn.points[i].curvature = curve;
  }
}

// 生成整理好的点云和序号
template <typename PointType>
void generateFromPointCloudScans(
    const std::vector<pcl::PointCloud<PointType>> &laserCloudScans,
    pcl::PointCloud<PointType> &laserCloudOut, std::vector<int> &scanStartInd,
    std::vector<int> &scanEndInd) {
  scanStartInd.clear();
  scanStartInd.resize(N_SCAN);
  scanEndInd.clear();
  scanEndInd.resize(N_SCAN);
  for (int i = 0; i < N_SCAN; i++) {
    scanStartInd[i] = laserCloudOut.size() + 5;
    laserCloudOut += laserCloudScans[i];
    scanEndInd[i] = laserCloudOut.size() - 6;
  }
}

// 读取各行点云
template <typename PointT, typename PointType>
void parsePointCloudScans(
    const pcl::PointCloud<PointT> &laserCloudIn,
    std::vector<pcl::PointCloud<PointType>> &laserCloudScans, float startOri,
    float endOri) {
  bool halfPassed = false;
  int cloudSize = laserCloudIn.points.size();
  int count = cloudSize;
  PointType point;
  laserCloudScans.clear();
  laserCloudScans.resize(N_SCAN);
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;

    float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) *
                  180 / M_PI;
    int scanID = 0;

    scanID = int((angle - MIN_ANGLE + RES_ANGLE) /
                 (MAX_ANGLE + RES_ANGLE - MIN_ANGLE + RES_ANGLE) * N_SCAN);
    if (scanID > (N_SCAN - 1) || scanID < 0) {
      count--;
      continue;
    }
    // printf("angle %f scanID %d \n", angle, scanID);

    float ori = -atan2(point.y, point.x);
    if (!halfPassed) {
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      }
    }

    float relTime = (ori - startOri) / (endOri - startOri);
    point.intensity = scanID + SCAN_PERIOD * relTime;
    point.normal_x = 0.f;  //光流矢量
    point.normal_y = 0.f;
    point.normal_z = 0.f;
    point.curvature = 0.f;  //矢量长度
    laserCloudScans[scanID].push_back(point);
  }
  // printf("points size %d \n", count);
}

// 读取激光点云起始角和结束角
template <typename PointT>
void readPointCloudOrient(const pcl::PointCloud<PointT> &laserCloudIn,
                          float &startOri, float &endOri) {
  int cloudSize = laserCloudIn.points.size();
  startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                  laserCloudIn.points[cloudSize - 1].x) +
           2 * M_PI;

  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
}

// 按激光点的曲率排序
inline bool comp(int i, int j) {
  return (cloudCurvature[i] < cloudCurvature[j]);
}

// 从点云中间分割平面点和边缘点（锋利点）
void segmentSurfAndConner(
    const std::vector<int> &scanStartInd, const std::vector<int> &scanEndInd,
    const pcl::PointCloud<PointType> &laserCloudIn,
    pcl::PointCloud<PointType>::Ptr &cornerPointsSharp,
    pcl::PointCloud<PointType>::Ptr &cornerPointsLessSharp,
    pcl::PointCloud<PointType>::Ptr &surfPointsFlat,
    pcl::PointCloud<PointType>::Ptr &surfPointsLessFlat) {
  static float sharp_thresh = 3000;
  static float mid_thresh = 10;
  static float flat_thresh = 0.0005;
  for (int i = 0; i < N_SCAN; i++) {
    int sp = scanStartInd[i];
    int ep = scanEndInd[i];

    if (ep - sp < 0) continue;

    std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);

    for (int k = ep; k >= sp; k--) {
      int ind = cloudSortInd[k];
      if (cloudNeighborPicked[ind] == 0) {
        if (cloudCurvature[ind] > sharp_thresh) {
          cloudLabel[ind] = 2;
          cornerPointsSharp->push_back(laserCloudIn.points[ind]);
          cornerPointsLessSharp->push_back(laserCloudIn.points[ind]);
        } else if (cloudCurvature[ind] > mid_thresh) {
          cloudLabel[ind] = 1;
          cornerPointsLessSharp->push_back(laserCloudIn.points[ind]);
        } else {
          break;
        }

        for (int l = -5; l <= 5; l++) {
          cloudNeighborPicked[ind + l] = 1;
        }
      }
    }

    for (int k = sp; k <= ep; k++) {
      int ind = cloudSortInd[k];
      if (cloudNeighborPicked[ind] == 0) {
        if (cloudCurvature[ind] < flat_thresh) {
          cloudLabel[ind] = -2;
          surfPointsFlat->push_back(laserCloudIn.points[ind]);
          surfPointsLessFlat->push_back(laserCloudIn.points[ind]);
        } else if (cloudCurvature[ind] < mid_thresh) {
          cloudLabel[ind] = -1;
          surfPointsLessFlat->push_back(laserCloudIn.points[ind]);
        } else {
          break;
        }

        for (int l = -5; l <= 5; l++) {
          cloudNeighborPicked[ind + l] = 1;
        }
      }
    }
  }

  pcl::VoxelGrid<PointType> downSizeFilter;
  downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
  downSizeFilter.setInputCloud(cornerPointsSharp);
  downSizeFilter.filter(*cornerPointsSharp);
  downSizeFilter.setInputCloud(cornerPointsLessSharp);
  downSizeFilter.filter(*cornerPointsLessSharp);
  downSizeFilter.setInputCloud(surfPointsFlat);
  downSizeFilter.filter(*surfPointsFlat);
  downSizeFilter.setInputCloud(surfPointsLessFlat);
  downSizeFilter.filter(*surfPointsLessFlat);

  // 调整阈值，使得比值(numLessSharp:numSharp:numFlat:numLessFlat) = (1:5:10：2)
  float numSharp = cornerPointsSharp->size();
  float numLessSharp = cornerPointsLessSharp->size();
  float numFlat = surfPointsFlat->size();
  float numLessFlat = surfPointsLessFlat->size();
  // std::cout << "( " << numSharp << " , " << numLessSharp << " , " << numFlat
  // << " , " << numLessFlat << " )"
  //           << std::endl;

  if (numLessSharp / numSharp > 5.5)
    sharp_thresh *= 0.9;
  else if (numLessSharp / numSharp < 4.5)
    sharp_thresh *= 1.1;
  if (numLessFlat / numLessSharp > 2.2)
    mid_thresh *= 0.9;
  else if (numLessFlat / numLessSharp < 1.8)
    mid_thresh *= 1.1;
  if (numLessFlat / numFlat > 5.5)
    flat_thresh *= 1.1;
  else if (numLessFlat / numFlat < 4.5)
    flat_thresh *= 0.9;
  // std::cout << "( " << sharp_thresh << " , " << mid_thresh << " , " <<
  // flat_thresh << " )" << std::endl;
}

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
void addMatchPointToViews(const std::vector<cv::Point2f> &points_1,
                          const std::vector<cv::Point2f> &points_2,
                          const cv::Mat &descriptors_1,
                          const std::vector<cv::Point3d> &points_3d,
                          const Eigen::Quaterniond &q_w_curr,
                          const Eigen::Vector3d &t_w_curr,
                          cv::Mat &descriptorsInMap, int camera_idx,
                          std::vector<CameraView> &camera_views,
                          std::vector<Eigen::Vector3d> &points_3d_maps) {
  std::cout << "addMatchPointToViews() start." << std::endl;
  //描述子和观察表中的描述子进行匹配
  std::vector<bool> choosen_flag;
  choosen_flag.resize(points_1.size(), false);
  std::vector<cv::DMatch> matches_map;
  CameraView view1, view2;
  view1.camera_idx = camera_idx;
  view2.camera_idx = camera_idx + 1;
  if (camera_idx > 0) {
    robustMatch(descriptors_1, descriptorsInMap, matches_map);
    //根据匹配列表为已有特征点添加观测
    for (size_t i = 0; i < matches_map.size(); i++) {
      if (points_1[matches_map[i].queryIdx].x >= image_size.x ||
          points_1[matches_map[i].queryIdx].y >= image_size.y ||
          points_1[matches_map[i].queryIdx].x <= 1 ||
          points_1[matches_map[i].queryIdx].y <= 1)
        continue;

      if (points_2[matches_map[i].queryIdx].x >= image_size.x ||
          points_2[matches_map[i].queryIdx].y >= image_size.y ||
          points_2[matches_map[i].queryIdx].x <= 1 ||
          points_2[matches_map[i].queryIdx].y <= 1)
        continue;

      view1.point_idx = matches_map[i].trainIdx;
      view1.observation_x = points_1[matches_map[i].queryIdx].x;
      view1.observation_y = points_1[matches_map[i].queryIdx].y;
      camera_views.push_back(view1);

      view2.point_idx = matches_map[i].trainIdx;
      view2.observation_x = points_2[matches_map[i].queryIdx].x;
      view2.observation_y = points_2[matches_map[i].queryIdx].y;
      // camera_views.push_back(view1);

      descriptorsInMap.row(matches_map[i].trainIdx) =
          descriptors_1.row(matches_map[i].queryIdx);
      choosen_flag[i] = true;
    }
  }
  //根据未匹配列表添加新的特征点和观测
  for (size_t i = 0; i < choosen_flag.size(); i++) {
    //是否已匹配
    if (choosen_flag[i]) continue;

    cv::Point3d point = points_3d[i];
    cv::Mat pt_trans = left_camera_to_base_pose *
                       (cv::Mat_<double>(4, 1) << point.x, point.y, point.z, 1);
    double depth = pt_trans.at<double>(2, 0);

    if (depth > stereoDistanceThresh || depth < 0.54) continue;

    if (points_1[i].x >= image_size.x || points_1[i].y >= image_size.y ||
        points_1[i].x <= 1 || points_1[i].y <= 1)
      continue;

    if (points_2[i].x >= image_size.x || points_2[i].y >= image_size.y ||
        points_2[i].x <= 1 || points_2[i].y <= 1)
      continue;

    view1.point_idx = descriptorsInMap.rows;
    view1.observation_x = points_1[i].x;
    view1.observation_y = points_1[i].y;
    camera_views.push_back(view1);

    view2.point_idx = descriptorsInMap.rows;
    view2.observation_x = points_2[i].x;
    view2.observation_y = points_2[i].y;
    // camera_views.push_back(view2);

    descriptorsInMap.push_back(descriptors_1.row(i));

    //把点转换到地图坐标系
    CvPointTransform(point, point, q_w_curr, t_w_curr);
    //把点添加到地图特征点集中
    Eigen::Vector3d point_temp;
    point_temp.x() = point.x;
    point_temp.y() = point.y;
    point_temp.z() = point.z;
    points_3d_maps.push_back(point_temp);
  }
  std::cout << "addMatchPointToViews() end." << std::endl;
}

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

  preprocess_thread_mutex.lock();
  left_image_msg_buf = *left_image_msg;
  left_cam_info_msg_buf = *left_cam_info_msg;
  right_image_msg_buf = *right_image_msg;
  right_cam_info_msg_buf = *right_cam_info_msg;
  point_cloud_msg_buf = *point_cloud_msg;
  is_preprocess_thread_ready = true;
  preprocess_thread_mutex.unlock();

  static tf::TransformBroadcaster broadcaster;
  tf::Transform tf_left_camera_to_base_pose(tf::Quaternion(0.5, -0.5, 0.5, 0.5),
                                            tf::Vector3(0, -0.08, -0.27));
  broadcaster.sendTransform(
      tf::StampedTransform(tf_left_camera_to_base_pose, ros::Time::now(),
                           "/kitti_base_link", "/kitti_velo_link"));

  ROS_INFO("callbackHandle Stop\n");
}

// 激光雷达的参数
void parseLidarType(const std::string &lidarType) {
  printf("Lidar type is %s", lidarType.c_str());
  if (lidarType == "VLP-16") {
    N_SCAN = 16;
    MAX_RANGE = 100;
    MIN_RANGE = 0;
    RES_RANGE = 0.03f * 2;
    MAX_ANGLE = 15;
    MIN_ANGLE = -15;
    RES_ANGLE = 2 * 2;
  } else if (lidarType == "HDL-32E") {
    N_SCAN = 32;
    MAX_RANGE = 100;
    MIN_RANGE = 0;
    RES_RANGE = 0.02f * 2;
    MAX_ANGLE = 10.67f;
    MIN_ANGLE = -30.67f;
    RES_ANGLE = 1.33f * 2;
  } else if (lidarType == "HDL-64E") {
    N_SCAN = 64;
    MAX_RANGE = 120;
    MIN_RANGE = 0;
    RES_RANGE = 0.02f * 2;
    MAX_ANGLE = 2;
    MIN_ANGLE = -24.8f;
    RES_ANGLE = 0.4f * 2;
  } else {
    printf("， which is UNRECOGNIZED!!!\n");
    ROS_BREAK();
  }
  printf(".\n");
}

void preprocessThread() __attribute__((noreturn));
void odometryThread() __attribute__((noreturn));
void mappingThread() __attribute__((noreturn));

void preprocessThread() {
  while (1) {
    //-- 第0步：线程休眠,时长为线程运行时常
    static auto start_time = std::chrono::high_resolution_clock::now();
    static auto end_time = std::chrono::high_resolution_clock::now();
    static std::chrono::duration<double, std::milli> elapsed_duration =
        end_time - start_time;
    std::this_thread::sleep_for(elapsed_duration);

    //-- 第1步：判断是否有新消息
    preprocess_thread_mutex.lock();
    bool is_msg_updated_local = is_preprocess_thread_ready;
    preprocess_thread_mutex.unlock();
    if (!is_msg_updated_local) {
      continue;
    }

    ROS_INFO("preprocessThread Start\n");
    start_time = std::chrono::high_resolution_clock::now();

    //-- 第2步：从线程外读取新消息
    preprocess_thread_mutex.lock();
    sensor_msgs::Image left_image_msg = left_image_msg_buf;
    sensor_msgs::CameraInfo left_cam_info_msg = left_cam_info_msg_buf;
    sensor_msgs::Image right_image_msg = right_image_msg_buf;
    sensor_msgs::CameraInfo right_cam_info_msg = right_cam_info_msg_buf;
    sensor_msgs::PointCloud2 point_cloud_msg = point_cloud_msg_buf;
    is_preprocess_thread_ready = false;
    preprocess_thread_mutex.unlock();

    //-- 第3步：读取视觉图像数据
    cv_bridge::CvImage cv_ptr_1, cv_ptr_2;
    cv_ptr_1 = *cv_bridge::toCvCopy(left_image_msg, left_image_msg.encoding);
    cv_ptr_2 = *cv_bridge::toCvCopy(right_image_msg, right_image_msg.encoding);

    //-- 第4步：读取激光点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(point_cloud_msg, *laserCloudIn);

    //-- 第5步：检测 Oriented FAST 角点位置
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    detector->detect(cv_ptr_1.image, keypoints_1);
    detector->detect(cv_ptr_2.image, keypoints_2);

    //-- 第6步：根据角点位置计算 BRIEF 描述子
    cv::Mat descriptors_1, descriptors_2;
    descriptor->compute(cv_ptr_1.image, keypoints_1, descriptors_1);
    descriptor->compute(cv_ptr_2.image, keypoints_2, descriptors_2);

    //-- 第7步：对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> matches_stereo;
    robustMatch(descriptors_1, descriptors_2, matches_stereo);

    //-- 第8步：筛选出匹配点
    std::vector<cv::Point2f> points_1, points_2;
    filterMatchPoints(keypoints_1, keypoints_2, matches_stereo, points_1,
                      points_2);

    //-- 第9步：三角化计算
    cv::Mat points_4d;
    cv::triangulatePoints(left_camera_to_base_pose, right_camera_to_base_pose,
                          points_1, points_2, points_4d);

    //-- 第10步：齐次三维点归一化
    std::vector<cv::Point3d> points_3d;
    normalizeHomogeneousPoints(points_4d, points_3d);

    //-- 第10.1步：根据三维点的具体位置限制,筛选相机前方合理区间的特征点
    cv::Mat descriptors_left, descriptors_right;
    std::vector<cv::Point2f> imgPoints_left, imgPoints_right;
    std::vector<cv::DMatch> good_matches_stereo;
    std::vector<cv::Point3d> good_points_3d;
    filterUsableKeyPoints(
        keypoints_1, keypoints_2, descriptors_1, descriptors_2, matches_stereo,
        points_3d, imgPoints_left, imgPoints_right, descriptors_left,
        descriptors_right, good_matches_stereo, good_points_3d);

    //-- 第11步：消除无意义点和距离为零的点
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
    removeClosedPointCloud(*laserCloudIn, *laserCloudIn, MINIMUM_RANGE);

    //-- 第12步：计算点云每条线的曲率
    float startOri, endOri;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans;
    pcl::PointCloud<PointType> laserCloudOut;
    std::vector<int> scanStartInd, scanEndInd;
    readPointCloudOrient(*laserCloudIn, startOri, endOri);
    parsePointCloudScans(*laserCloudIn, laserCloudScans, startOri, endOri);
    generateFromPointCloudScans(laserCloudScans, laserCloudOut, scanStartInd,
                                scanEndInd);
    calculatePointCurvature(laserCloudOut);

    //-- 第13步：点云分割点云
    pcl::PointCloud<PointType>::Ptr cornerPointsSharp(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr surfPointsFlat(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(
        new pcl::PointCloud<PointType>);
    segmentSurfAndConner(scanStartInd, scanEndInd, laserCloudOut,
                         cornerPointsSharp, cornerPointsLessSharp,
                         surfPointsFlat, surfPointsLessFlat);

    //-- 第14步：输出结果给里程计线程
    odometry_thread_mutex.lock();
    descriptors_left_buf = descriptors_left;
    descriptors_right_buf = descriptors_right;
    imgPoints_left_buf = imgPoints_left;
    imgPoints_right_buf = imgPoints_right;
    good_matches_stereo_buf = good_matches_stereo;
    good_points_3d_buf = good_points_3d;
    cornerPointsSharp_buf = *cornerPointsSharp;
    cornerPointsLessSharp_buf = *cornerPointsLessSharp;
    surfPointsFlat_buf = *surfPointsFlat;
    surfPointsLessFlat_buf = *surfPointsLessFlat;
    is_odometry_thread_ready = true;
    odometry_thread_mutex.unlock();

    cv_bridge::CvImage img_plot = cv_ptr_1;
    for (int i = 0; i < imgPoints_left.size(); i++)
      cv::circle(img_plot.image, imgPoints_left[i], 2,
                 get_color(good_points_3d[i].z), 2);
    pubLeftImageWithFeature.publish(img_plot.toImageMsg());

    pcl::PointCloud<PointType>::Ptr points_3d_clouds(
        new pcl::PointCloud<PointType>);
    addPoints3DToCloud(good_points_3d, points_3d_clouds);

    sensor_msgs::PointCloud2 laserCloudOutput;
    pcl::toROSMsg(*points_3d_clouds, laserCloudOutput);
    laserCloudOutput.header.stamp = ros::Time::now();
    laserCloudOutput.header.frame_id = "/kitti_base_link";
    pubCameraPointsCloud.publish(laserCloudOutput);

    pcl::toROSMsg(*cornerPointsSharp, laserCloudOutput);
    laserCloudOutput.header.stamp = ros::Time::now();
    laserCloudOutput.header.frame_id = "/kitti_velo_link";
    pubCornerPointsSharp.publish(laserCloudOutput);

    pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutput);
    laserCloudOutput.header.stamp = ros::Time::now();
    laserCloudOutput.header.frame_id = "/kitti_velo_link";
    pubCornerPointsLessSharp.publish(laserCloudOutput);

    pcl::toROSMsg(*surfPointsFlat, laserCloudOutput);
    laserCloudOutput.header.stamp = ros::Time::now();
    laserCloudOutput.header.frame_id = "/kitti_velo_link";
    pubSurfPointsFlat.publish(laserCloudOutput);

    pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutput);
    laserCloudOutput.header.stamp = ros::Time::now();
    laserCloudOutput.header.frame_id = "/kitti_velo_link";
    pubSurfPointsLessFlat.publish(laserCloudOutput);

    ROS_INFO("preprocessThread End\n");
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_duration = end_time - start_time;
  }
}

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

    //-- 第3步：判断是否初始化
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
        printf("test point 1.");

        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization *q_parameterization =
            new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);

        printf("test point 2.");

        problem.AddParameterBlock(q_last_curr.coeffs().data(), 4,
                                  q_parameterization);
        problem.AddParameterBlock(t_last_curr.data(), 3);

        PointType pointSel;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // 建立边缘特征约束
        int cornerPointsSharpNum = cornerPointsSharp_curr.points.size();
        int corner_correspondence = 0;
        for (int i = 0; i < cornerPointsSharpNum; ++i) {
          TransformToStart(&(cornerPointsSharp_curr.points[i]), &pointSel,
                           q_last_curr, t_last_curr);
          kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                           pointSearchSqDis);

          int closestPointInd = -1, minPointInd2 = -1;
          if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
            closestPointInd = pointSearchInd[0];
            int closestPointScanID = int(
                cornerPointsLessSharp_last.points[closestPointInd].intensity);

            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
            // search in the direction of increasing scan line
            for (int j = closestPointInd + 1;
                 j < (int)cornerPointsLessSharp_last.points.size(); ++j) {
              // if in the same scan line, continue
              if (int(cornerPointsLessSharp_last.points[j].intensity) <=
                  closestPointScanID)
                continue;

              // if not in nearby scans, end the loop
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

            // search in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j) {
              // if in the same scan line, continue
              if (int(cornerPointsLessSharp_last.points[j].intensity) >=
                  closestPointScanID)
                continue;

              // if not in nearby scans, end the loop
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
          if (minPointInd2 >=
              0)  // both closestPointInd and minPointInd2 is valid
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
            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(
                curr_point, last_point_a, last_point_b, s);
            problem.AddResidualBlock(cost_function, loss_function,
                                     q_last_curr.coeffs().data(),
                                     t_last_curr.data());
            corner_correspondence++;
          }
        }

        printf("test point 4.");
        // find correspondence for plane features
        int surfPointsFlatNum = cornerPointsSharp_curr.points.size();
        int plane_correspondence = 0;
        for (int i = 0; i < surfPointsFlatNum; ++i) {
          TransformToStart(&(cornerPointsSharp_curr.points[i]), &pointSel,
                           q_last_curr, t_last_curr);
          kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                         pointSearchSqDis);

          int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
          if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
            closestPointInd = pointSearchInd[0];

            // get closest point's scan ID
            int closestPointScanID =
                int(surfPointsLessFlat_last.points[closestPointInd].intensity);
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD,
                   minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

            // search in the direction of increasing scan line
            for (int j = closestPointInd + 1;
                 j < (int)surfPointsLessFlat_last.points.size(); ++j) {
              // if not in nearby scans, end the loop
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

              // if in the same or lower scan line
              if (int(surfPointsLessFlat_last.points[j].intensity) <=
                      closestPointScanID &&
                  pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
              // if in the higher scan line
              else if (int(surfPointsLessFlat_last.points[j].intensity) >
                           closestPointScanID &&
                       pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }

            // search in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j) {
              // if not in nearby scans, end the loop
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

              // if in the same or higher scan line
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
              ceres::CostFunction *cost_function = LidarPlaneFactor::Create(
                  curr_point, last_point_a, last_point_b, last_point_c, s);
              problem.AddResidualBlock(cost_function, loss_function,
                                       q_last_curr.coeffs().data(),
                                       t_last_curr.data());
              plane_correspondence++;
            }
          }
        }

        printf("coner_correspondance %d, plane_correspondence %d \n",
               corner_correspondence, plane_correspondence);

        if ((corner_correspondence + plane_correspondence) < 10) {
          printf(
              "less correspondence! "
              "*************************************************\n");
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
        printf("solve_efficiency = %f %%, max_num_iter = %d.\n",
               solve_efficiency * 100, max_num_iter);
      }
    }
    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
    q_w_curr = q_w_curr * q_last_curr;

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
  parseLidarType(lidarType);

  //机器人的最大半径
  nh.param<double>("minimum_range", MINIMUM_RANGE, RES_RANGE);

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
  left_camera_matrix << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0,
      0.0, 1.0;
  right_camera_matrix << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0,
      0.0, 1.0;
  //传感器外参
  left_camera_to_base_pose =
      (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  right_camera_to_base_pose =
      (cv::Mat_<double>(3, 4) << 1, 0, 0, -0.54, 0, 1, 0, 0, 0, 0, 1, 0);

  lidar_to_base_pose =
      (cv::Mat_<double>(3, 4) << 0, 0, 1, 0.27, -1, 0, 0, 0, 0, -1, 0, -0.08);

  stereoDistanceThresh = 718.856 * 0.54 * 2;

  detector = cv::ORB::create();
  descriptor = cv::ORB::create();
  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  std::thread preprocess_thread{preprocessThread};
  // std::thread odometry_thread{odometryThread};
  // std::thread mapping_thread{ mappingThread };

  ros::spin();

  while (ros::ok()) {
    ros::spinOnce();
  }

  ROS_INFO("liso Stop\n");
}
