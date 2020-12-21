#ifndef LISO_PREPROCESS_THREAD_H_
#define LISO_PREPROCESS_THREAD_H_

#include <functional>
#include <mutex>
#include <thread>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include "liso/common.h"

class PreprocessThread {
 public:
  PreprocessThread(const std::string &lidarType);
  ~PreprocessThread(){};
  void run() __attribute__((noreturn));

  void set_left_image_msg_buf(const sensor_msgs::Image &tmp) {
    preprocess_thread_mutex.lock();
    left_image_msg_buf = tmp;
    preprocess_thread_mutex.unlock();
  };
  const sensor_msgs::Image &get_left_image_msg_buf() {
    preprocess_thread_mutex.lock();
    return left_image_msg_buf;
    preprocess_thread_mutex.unlock();
  };

  void set_left_cam_info_msg_buf(const sensor_msgs::CameraInfo &tmp) {
    preprocess_thread_mutex.lock();
    left_cam_info_msg_buf = tmp;
    preprocess_thread_mutex.unlock();
  };
  const sensor_msgs::CameraInfo &get_left_cam_info_msg_buf() {
    preprocess_thread_mutex.lock();
    return left_cam_info_msg_buf;
    preprocess_thread_mutex.unlock();
  };

  void set_right_image_msg_buf(const sensor_msgs::Image &tmp) {
    preprocess_thread_mutex.lock();
    right_image_msg_buf = tmp;
    preprocess_thread_mutex.unlock();
  };
  const sensor_msgs::Image &get_right_image_msg_buf() {
    preprocess_thread_mutex.lock();
    return right_image_msg_buf;
    preprocess_thread_mutex.unlock();
  };

  void set_right_cam_info_msg_buf(const sensor_msgs::CameraInfo &tmp) {
    preprocess_thread_mutex.lock();
    right_cam_info_msg_buf = tmp;
    preprocess_thread_mutex.unlock();
  };
  const sensor_msgs::CameraInfo &get_right_cam_info_msg_buf() {
    preprocess_thread_mutex.lock();
    return right_cam_info_msg_buf;
    preprocess_thread_mutex.unlock();
  };

  void set_point_cloud_msg_buf(const sensor_msgs::PointCloud2 &tmp) {
    preprocess_thread_mutex.lock();
    point_cloud_msg_buf = tmp;
    preprocess_thread_mutex.unlock();
  };
  const sensor_msgs::PointCloud2 &get_point_cloud_msg_buf() {
    preprocess_thread_mutex.lock();
    return point_cloud_msg_buf;
    preprocess_thread_mutex.unlock();
  };

  void set_is_preprocess_thread_ready(const bool tmp) {
    preprocess_thread_mutex.lock();
    is_preprocess_thread_ready = tmp;
    preprocess_thread_mutex.unlock();
  };
  const bool get_is_preprocess_thread_ready() {
    preprocess_thread_mutex.lock();
    return is_preprocess_thread_ready;
    preprocess_thread_mutex.unlock();
  };

 private:
  // 按激光点的曲率排序
  bool comp(int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }

  void parseLidarType(const std::string &lidarType);
  void robustMatch(const cv::Mat &queryDescriptors,
                   const cv::Mat &trainDescriptors,
                   std::vector<cv::DMatch> &matches);
  void filterMatchPoints(const std::vector<cv::KeyPoint> &keypoints_1,
                         const std::vector<cv::KeyPoint> &keypoints_2,
                         const std::vector<cv::DMatch> &matches,
                         std::vector<cv::Point2f> &points_1,
                         std::vector<cv::Point2f> &points_2);
  float normalizeHomogeneousPoints(const cv::Mat &points_4d,
                                   std::vector<cv::Point3d> &points_3d);
  void filterUsableKeyPoints(const std::vector<cv::KeyPoint> &keypoints_1,
                             const std::vector<cv::KeyPoint> &keypoints_2,
                             const cv::Mat &descriptors_1,
                             const cv::Mat &descriptors_2,
                             const std::vector<cv::DMatch> &matches,
                             const std::vector<cv::Point3d> &points_3d,
                             std::vector<cv::Point2f> &imgPoints_left,
                             std::vector<cv::Point2f> &imgPoints_right,
                             cv::Mat &descriptors_left,
                             cv::Mat &descriptors_right,
                             std::vector<cv::DMatch> &good_matches,
                             std::vector<cv::Point3d> &good_points_3d);
  template <typename PointT>
  void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres);
  template <typename PointT>
  void readPointCloudOrient(const pcl::PointCloud<PointT> &laserCloudIn,
                            float &startOri, float &endOri);
  template <typename PointT, typename PointType>
  void parsePointCloudScans(
      const pcl::PointCloud<PointT> &laserCloudIn,
      std::vector<pcl::PointCloud<PointType>> &laserCloudScans, float startOri,
      float endOri);
  template <typename PointType>
  void generateFromPointCloudScans(
      const std::vector<pcl::PointCloud<PointType>> &laserCloudScans,
      pcl::PointCloud<PointType> &laserCloudOut, std::vector<int> &scanStartInd,
      std::vector<int> &scanEndInd);
  template <typename PointType>
  void calculatePointCurvature(pcl::PointCloud<PointType> &laserCloudIn);
  void segmentSurfAndConner(
      const std::vector<int> &scanStartInd, const std::vector<int> &scanEndInd,
      const pcl::PointCloud<PointType> &laserCloudIn,
      pcl::PointCloud<PointType>::Ptr &cornerPointsSharp,
      pcl::PointCloud<PointType>::Ptr &cornerPointsLessSharp,
      pcl::PointCloud<PointType>::Ptr &surfPointsFlat,
      pcl::PointCloud<PointType>::Ptr &surfPointsLessFlat);
  cv::Point2f pixel2cam(const cv::Point2d &p, const Eigen::Matrix3d &K);

  sensor_msgs::Image left_image_msg_buf;
  sensor_msgs::CameraInfo left_cam_info_msg_buf;
  sensor_msgs::Image right_image_msg_buf;
  sensor_msgs::CameraInfo right_cam_info_msg_buf;
  sensor_msgs::PointCloud2 point_cloud_msg_buf;
  bool is_preprocess_thread_ready;
  std::mutex preprocess_thread_mutex;

  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  cv::Mat left_camera_to_base_pose;
  cv::Mat right_camera_to_base_pose;

  int N_SCAN = 16;
  float MAX_RANGE = 100;
  float MIN_RANGE = 0;
  float RES_RANGE = .3f;
  float MAX_ANGLE = 15;
  float MIN_ANGLE = -15;
  float RES_ANGLE = 2;

  //相机参数
  Eigen::Matrix3d left_camera_matrix;
  Eigen::Matrix3d right_camera_matrix;
  double stereoDistanceThresh;

  //激光提取边沿点和平面点
  float cloudCurvature[400000];
  int cloudSortInd[400000];
  int cloudNeighborPicked[400000];
  int cloudLabel[400000];
};
#endif  // LISO_PREPROCESS_THREAD_H_