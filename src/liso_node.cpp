#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

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

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "liso/CeresSolveFactor.hpp"
#include "liso/common.h"

//全局变量

//参数
static std::string left_image_sub;
static std::string left_camera_info_sub;
static std::string right_image_sub;
static std::string right_camera_info_sub;
static std::string point_cloud_sub;

static std::string left_image_with_feature_pub;
static std::string point_cloud_with_feature_pub;
static std::string conner_points_pub;
static std::string conner_points_less_pub;
static std::string surf_points_pub;
static std::string surf_points_less_pub;
static std::string laser_odometry_pub;

static std::string lidarType;
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

//特征提取
static cv::Ptr<cv::FeatureDetector> detector;
static cv::Ptr<cv::DescriptorExtractor> descriptor;
static cv::Ptr<cv::DescriptorMatcher> matcher;

//相机参数
static cv::Mat left_camera_matrix;
static cv::Mat right_camera_matrix;
static double stereoDistanceThresh;
static cv::Mat base_to_left_camera_pose;
static cv::Mat base_to_right_camera_pose;

//激光提取边沿点和平面点
static Accumulator<float> curvature_range(0.1);
static float cloudCurvature[400000];
static int cloudSortInd[400000];
static int cloudNeighborPicked[400000];
static int cloudLabel[400000];

//激光里程计

// 前后帧参数
// q_curr_last(x, y, z, w), t_curr_last(x,y,z)
double para_q[4] = { 0, 0, 0, 1 };
double para_t[3] = { 0, 0, 0 };
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

inline cv::Scalar get_color(float depth)
{
  static Accumulator<float> depth_range(50);
  if (depth < 4 * depth_range.mean() && depth > -2 * depth_range.mean())
    depth_range.addDataValue(depth);
  float up_th = 2 * depth_range.mean(), low_th = 0.f, th_range = up_th - low_th;
  if (depth > up_th)
    depth = up_th;
  if (depth < low_th)
    depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// void callbackLeftCameraInfo(const sensor_msgs::CameraInfoConstPtr &cam_info_msg)
// {
//   ROS_INFO("callbackLeftCameraInfo Start\n");

//   std::cout<<cam_info_msg->K[0]<<cam_info_msg->K[1]<<cam_info_msg->K[2]<<cam_info_msg->K[3]<<cam_info_msg->K[4]<<cam_info_msg->K[5]<<cam_info_msg->K[6]<<cam_info_msg->K[7]<<cam_info_msg->K[8];
//   std::cout<<std::endl;
//   last_descriptors =
//   cv::Mat_<double>(3,3)<<cam_info_msg->K[0],cam_info_msg->K[1],cam_info_msg->K[2],cam_info_msg->K[3],cam_info_msg->K[4],cam_info_msg->K[5],cam_info_msg->K[6],cam_info_msg->K[7],cam_info_msg->K[8];
//   std::cout << "cam_mat is " << std::endl << last_descriptors << std::endl;

//   ROS_INFO("callbackLeftImage Stop\n");
// }

// 归一化齐次点
float normalizeHomogeneousPoints(const cv::Mat &points_4d, std::vector<cv::Point3d> &points_3d)
{
  Accumulator<float> scale_count;
  for (int i = 0; i < points_4d.cols; i++)
  {
    cv::Mat x = points_4d.col(i);
    scale_count.addDataValue(x.at<float>(3, 0));
    x /= x.at<float>(3, 0);  // 归一化
    cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points_3d.push_back(p);
  }
  return scale_count.mean();
}

// 转换像素点
cv::Point2f pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{
  return cv::Point2f((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

//根据匹配获得匹配点
void filterMatchPoints(const std::vector<cv::KeyPoint> &keypoints_1, const std::vector<cv::KeyPoint> &keypoints_2,
                       const std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &points_1,
                       std::vector<cv::Point2f> &points_2)
{
  points_1.clear();
  points_2.clear();
  for (int i = 0; i < (int)matches.size(); i++)
  {
    points_1.push_back(pixel2cam(keypoints_1[matches[i].queryIdx].pt, left_camera_matrix));
    points_2.push_back(pixel2cam(keypoints_2[matches[i].trainIdx].pt, right_camera_matrix));
  }
}

//带正反检查的描述子匹配
void robustMatch(const cv::Mat &keypoints_desc1, const cv::Mat &keypoints_desc2, std::vector<cv::DMatch> &matches)
{
  std::vector<cv::DMatch> matches_1, matches_2;
  matches.clear();
  matcher->match(keypoints_desc1, keypoints_desc2, matches_1);
  matcher->match(keypoints_desc2, keypoints_desc1, matches_2);

  for (int i = 0; i < (int)matches_1.size(); i++)
  {
    for (int j = 0; j < (int)matches_2.size(); j++)
    {
      if (matches_1[i].queryIdx == matches_2[j].trainIdx && matches_2[j].queryIdx == matches_1[i].trainIdx)
      {
        matches.push_back(matches_1[i]);
        break;
      }
    }
  }
  std::cout << "matches.size() = " << matches.size() << std::endl;
}

//去除指定半径内的点
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, float thres)
{
  if (&cloud_in != &cloud_out)
  {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  size_t j = 0;

  for (size_t i = 0; i < cloud_in.points.size(); ++i)
  {
    if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y +
            cloud_in.points[i].z * cloud_in.points[i].z <
        thres * thres)
      continue;
    cloud_out.points[j] = cloud_in.points[i];
    j++;
  }
  if (j != cloud_in.points.size())
  {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<uint32_t>(j);
  cloud_out.is_dense = true;
}

// 计算点云各点的曲率
template <typename PointType>
void calculatePointCurvature(pcl::PointCloud<PointType> &laserCloudIn)
{
  int cloudSize = laserCloudIn.points.size();
  for (int i = 5; i < cloudSize - 5; i++)
  {
    float diffX = laserCloudIn.points[i - 5].x + laserCloudIn.points[i - 4].x + laserCloudIn.points[i - 3].x +
                  laserCloudIn.points[i - 2].x + laserCloudIn.points[i - 1].x - 10 * laserCloudIn.points[i].x +
                  laserCloudIn.points[i + 1].x + laserCloudIn.points[i + 2].x + laserCloudIn.points[i + 3].x +
                  laserCloudIn.points[i + 4].x + laserCloudIn.points[i + 5].x;
    float diffY = laserCloudIn.points[i - 5].y + laserCloudIn.points[i - 4].y + laserCloudIn.points[i - 3].y +
                  laserCloudIn.points[i - 2].y + laserCloudIn.points[i - 1].y - 10 * laserCloudIn.points[i].y +
                  laserCloudIn.points[i + 1].y + laserCloudIn.points[i + 2].y + laserCloudIn.points[i + 3].y +
                  laserCloudIn.points[i + 4].y + laserCloudIn.points[i + 5].y;
    float diffZ = laserCloudIn.points[i - 5].z + laserCloudIn.points[i - 4].z + laserCloudIn.points[i - 3].z +
                  laserCloudIn.points[i - 2].z + laserCloudIn.points[i - 1].z - 10 * laserCloudIn.points[i].z +
                  laserCloudIn.points[i + 1].z + laserCloudIn.points[i + 2].z + laserCloudIn.points[i + 3].z +
                  laserCloudIn.points[i + 4].z + laserCloudIn.points[i + 5].z;
    float curve = diffX * diffX + diffY * diffY + diffZ * diffZ;

    cloudCurvature[i] = curve;
    cloudSortInd[i] = i;
    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;

    laserCloudIn.points[i].curvature = curve;
    if (curve < 4 * curvature_range.mean() && curve > -2 * curvature_range.mean())
      curvature_range.addDataValue(cloudCurvature[i]);
  }

  std::cout << "curvature_range = " << curvature_range.mean() << std::endl;
  std::cout << "curvature_range std = " << curvature_range.stddev() << std::endl;
}

// 生成整理好的点云和序号
template <typename PointType>
void generateFromPointCloudScans(const std::vector<pcl::PointCloud<PointType>> &laserCloudScans,
                                 pcl::PointCloud<PointType> &laserCloudOut, std::vector<int> &scanStartInd,
                                 std::vector<int> &scanEndInd)
{
  scanStartInd.clear();
  scanStartInd.resize(N_SCAN);
  scanEndInd.clear();
  scanEndInd.resize(N_SCAN);
  for (int i = 0; i < N_SCAN; i++)
  {
    scanStartInd[i] = laserCloudOut.size() + 5;
    laserCloudOut += laserCloudScans[i];
    scanEndInd[i] = laserCloudOut.size() - 6;
  }
}

// 读取各行点云
template <typename PointT, typename PointType>
void parsePointCloudScans(const pcl::PointCloud<PointT> &laserCloudIn,
                          std::vector<pcl::PointCloud<PointType>> &laserCloudScans, float startOri, float endOri)
{
  bool halfPassed = false;
  int cloudSize = laserCloudIn.points.size();
  int count = cloudSize;
  PointType point;
  laserCloudScans.clear();
  laserCloudScans.resize(N_SCAN);
  for (int i = 0; i < cloudSize; i++)
  {
    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;

    float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
    int scanID = 0;

    scanID = int((angle - MIN_ANGLE + RES_ANGLE) / (MAX_ANGLE + RES_ANGLE - MIN_ANGLE + RES_ANGLE) * N_SCAN);
    if (scanID > (N_SCAN - 1) || scanID < 0)
    {
      count--;
      continue;
    }
    // printf("angle %f scanID %d \n", angle, scanID);

    float ori = -atan2(point.y, point.x);
    if (!halfPassed)
    {
      if (ori < startOri - M_PI / 2)
      {
        ori += 2 * M_PI;
      }
      else if (ori > startOri + M_PI * 3 / 2)
      {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI)
      {
        halfPassed = true;
      }
    }
    else
    {
      ori += 2 * M_PI;
      if (ori < endOri - M_PI * 3 / 2)
      {
        ori += 2 * M_PI;
      }
      else if (ori > endOri + M_PI / 2)
      {
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
  printf("points size %d \n", count);
}

// 读取激光点云起始角和结束角
template <typename PointT>
void readPointCloudOrient(const pcl::PointCloud<PointT> &laserCloudIn, float &startOri, float &endOri)
{
  int cloudSize = laserCloudIn.points.size();
  startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

  if (endOri - startOri > 3 * M_PI)
  {
    endOri -= 2 * M_PI;
  }
  else if (endOri - startOri < M_PI)
  {
    endOri += 2 * M_PI;
  }
}

// 按激光点的曲率排序
inline bool comp(int i, int j)
{
  return (cloudCurvature[i] < cloudCurvature[j]);
}

// 从点云中间分割平面点和边缘点（锋利点）
void segmentSurfAndConner(const std::vector<int> &scanStartInd, const std::vector<int> &scanEndInd,
                          const pcl::PointCloud<PointType> &laserCloudIn,
                          pcl::PointCloud<PointType>::Ptr &cornerPointsSharp,
                          pcl::PointCloud<PointType>::Ptr &cornerPointsLessSharp,
                          pcl::PointCloud<PointType>::Ptr &surfPointsFlat,
                          pcl::PointCloud<PointType>::Ptr &surfPointsLessFlat)
{
  static float sharp_thresh = 3000;
  static float mid_thresh = 10;
  static float flat_thresh = 0.0005;
  for (int i = 0; i < N_SCAN; i++)
  {
    int sp = scanStartInd[i];
    int ep = scanEndInd[i];

    if (ep - sp < 0)
      continue;

    std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);

    for (int k = ep; k >= sp; k--)
    {
      int ind = cloudSortInd[k];
      if (cloudNeighborPicked[ind] == 0)
      {
        if (cloudCurvature[ind] > sharp_thresh)
        {
          cloudLabel[ind] = 2;
          cornerPointsSharp->push_back(laserCloudIn.points[ind]);
          cornerPointsLessSharp->push_back(laserCloudIn.points[ind]);
        }
        else if (cloudCurvature[ind] > mid_thresh)
        {
          cloudLabel[ind] = 1;
          cornerPointsLessSharp->push_back(laserCloudIn.points[ind]);
        }
        else
        {
          break;
        }

        for (int l = -5; l <= 5; l++)
        {
          cloudNeighborPicked[ind + l] = 1;
        }
      }
    }

    for (int k = sp; k <= ep; k++)
    {
      int ind = cloudSortInd[k];
      if (cloudNeighborPicked[ind] == 0)
      {
        if (cloudCurvature[ind] < flat_thresh)
        {
          cloudLabel[ind] = -2;
          surfPointsFlat->push_back(laserCloudIn.points[ind]);
          surfPointsLessFlat->push_back(laserCloudIn.points[ind]);
        }
        else if (cloudCurvature[ind] < mid_thresh)
        {
          cloudLabel[ind] = -1;
          surfPointsLessFlat->push_back(laserCloudIn.points[ind]);
        }
        else
        {
          break;
        }

        for (int l = -5; l <= 5; l++)
        {
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

  float numSharp = cornerPointsSharp->size();
  float numLessSharp = cornerPointsLessSharp->size();
  float numFlat = surfPointsFlat->size();
  float numLessFlat = surfPointsLessFlat->size();
  std::cout << "( " << numSharp << " , " << numLessSharp << " , " << numFlat << " , " << numLessFlat << " )"
            << std::endl;

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

  std::cout << "( " << sharp_thresh << " , " << mid_thresh << " , " << flat_thresh << " )" << std::endl;

  // for (int i = 0; i < N_SCAN; i++)
  // {
  //   if( scanEndInd[i] - scanStartInd[i] < 6)
  //     continue;
  //   pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
  //   for (int j = 0; j < 6; j++)
  //   {
  //     int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
  //     int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

  //     std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);

  //     int largestPickedNum = 0;
  //     for (int k = ep; k >= sp; k--)
  //     {
  //       int ind = cloudSortInd[k];

  //       if (cloudNeighborPicked[ind] == 0 &&
  //           cloudCurvature[ind] > 0.1)
  //       {

  //         largestPickedNum++;
  //         if (largestPickedNum <= 2)
  //         {
  //           cloudLabel[ind] = 2;
  //           cornerPointsSharp.push_back(laserCloudIn.points[ind]);
  //           cornerPointsLessSharp.push_back(laserCloudIn.points[ind]);
  //         }
  //         else if (largestPickedNum <= 20)
  //         {
  //           cloudLabel[ind] = 1;
  //           cornerPointsLessSharp.push_back(laserCloudIn.points[ind]);
  //         }
  //         else
  //         {
  //           break;
  //         }

  //         cloudNeighborPicked[ind] = 1;

  //         for (int l = 1; l <= 5; l++)
  //         {
  //           // float diffX = laserCloudIn.points[ind + l].x - laserCloudIn.points[ind + l - 1].x;
  //           // float diffY = laserCloudIn.points[ind + l].y - laserCloudIn.points[ind + l - 1].y;
  //           // float diffZ = laserCloudIn.points[ind + l].z - laserCloudIn.points[ind + l - 1].z;
  //           // if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
  //           // {
  //           //   break;
  //           // }

  //           cloudNeighborPicked[ind + l] = 1;
  //         }
  //         for (int l = -1; l >= -5; l--)
  //         {
  //           // float diffX = laserCloudIn.points[ind + l].x - laserCloudIn.points[ind + l + 1].x;
  //           // float diffY = laserCloudIn.points[ind + l].y - laserCloudIn.points[ind + l + 1].y;
  //           // float diffZ = laserCloudIn.points[ind + l].z - laserCloudIn.points[ind + l + 1].z;
  //           // if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
  //           // {
  //           //   break;
  //           // }

  //           cloudNeighborPicked[ind + l] = 1;
  //         }
  //       }
  //     }

  //     int smallestPickedNum = 0;
  //     for (int k = sp; k <= ep; k++)
  //     {
  //       int ind = cloudSortInd[k];

  //       if (cloudNeighborPicked[ind] == 0 &&
  //           cloudCurvature[ind] < 0.1)
  //       {

  //         cloudLabel[ind] = -1;
  //         surfPointsFlat.push_back(laserCloudIn.points[ind]);

  //         smallestPickedNum++;
  //         if (smallestPickedNum >= 4)
  //         {
  //           break;
  //         }

  //         cloudNeighborPicked[ind] = 1;
  //         for (int l = 1; l <= 5; l++)
  //         {
  //           // float diffX = laserCloudIn.points[ind + l].x - laserCloudIn.points[ind + l - 1].x;
  //           // float diffY = laserCloudIn.points[ind + l].y - laserCloudIn.points[ind + l - 1].y;
  //           // float diffZ = laserCloudIn.points[ind + l].z - laserCloudIn.points[ind + l - 1].z;
  //           // if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
  //           // {
  //           //   break;
  //           // }

  //           cloudNeighborPicked[ind + l] = 1;
  //         }
  //         for (int l = -1; l >= -5; l--)
  //         {
  //           // float diffX = laserCloudIn.points[ind + l].x - laserCloudIn.points[ind + l + 1].x;
  //           // float diffY = laserCloudIn.points[ind + l].y - laserCloudIn.points[ind + l + 1].y;
  //           // float diffZ = laserCloudIn.points[ind + l].z - laserCloudIn.points[ind + l + 1].z;
  //           // if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
  //           // {
  //           //   break;
  //           // }

  //           cloudNeighborPicked[ind + l] = 1;
  //         }
  //       }
  //     }

  //     for (int k = sp; k <= ep; k++)
  //     {
  //       if (cloudLabel[k] <= 0)
  //       {
  //         surfPointsLessFlatScan->push_back(laserCloudIn.points[k]);
  //       }
  //     }
  //   }

  //   pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
  //   pcl::VoxelGrid<PointType> downSizeFilter;
  //   downSizeFilter.setInputCloud(surfPointsLessFlatScan);
  //   downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
  //   downSizeFilter.filter(surfPointsLessFlatScanDS);

  //   surfPointsLessFlat += surfPointsLessFlatScanDS;
  // }
}

// 把点转环到上一帧的坐标系上
void TransformToStart(PointType const *const pi, PointType *const po)
{
  // interpolation ratio
  double s;
  if (DISTORTION)
    s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
  else
    s = 1.0;
  // s = 1;
  Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
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

//激光里程计优化
void lidarOdometryOptimism(const pcl::PointCloud<PointType>::Ptr &cornerPointsSharp,
                           const pcl::PointCloud<PointType>::Ptr &surfPointsFlat,
                           const pcl::PointCloud<PointType>::Ptr &cornerPointsLast,
                           const pcl::PointCloud<PointType>::Ptr &surfPointsLast,
                           const pcl::KdTreeFLANN<PointType>::Ptr &kdtreeCornerLast,
                           const pcl::KdTreeFLANN<PointType>::Ptr &kdtreeSurfLast, double *para_q, double *para_t)

{
  int cornerPointsSharpNum = cornerPointsSharp->points.size();
  int surfPointsFlatNum = surfPointsFlat->points.size();
  int corner_correspondence = 0;
  int plane_correspondence = 0;

  ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
  ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  problem.AddParameterBlock(para_q, 4, q_parameterization);
  problem.AddParameterBlock(para_t, 3);

  PointType pointSel;
  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  // 建立边缘特征约束
  for (int i = 0; i < cornerPointsSharpNum; ++i)
  {
    TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
    kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

    int closestPointInd = -1, minPointInd2 = -1;
    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
    {
      closestPointInd = pointSearchInd[0];
      int closestPointScanID = int(cornerPointsLast->points[closestPointInd].intensity);

      double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
      // search in the direction of increasing scan line
      for (int j = closestPointInd + 1; j < (int)cornerPointsLast->points.size(); ++j)
      {
        // if in the same scan line, continue
        if (int(cornerPointsLast->points[j].intensity) <= closestPointScanID)
          continue;

        // if not in nearby scans, end the loop
        if (int(cornerPointsLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
          break;

        double pointSqDis =
            (cornerPointsLast->points[j].x - pointSel.x) * (cornerPointsLast->points[j].x - pointSel.x) +
            (cornerPointsLast->points[j].y - pointSel.y) * (cornerPointsLast->points[j].y - pointSel.y) +
            (cornerPointsLast->points[j].z - pointSel.z) * (cornerPointsLast->points[j].z - pointSel.z);

        if (pointSqDis < minPointSqDis2)
        {
          // find nearer point
          minPointSqDis2 = pointSqDis;
          minPointInd2 = j;
        }
      }

      // search in the direction of decreasing scan line
      for (int j = closestPointInd - 1; j >= 0; --j)
      {
        // if in the same scan line, continue
        if (int(cornerPointsLast->points[j].intensity) >= closestPointScanID)
          continue;

        // if not in nearby scans, end the loop
        if (int(cornerPointsLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
          break;

        double pointSqDis =
            (cornerPointsLast->points[j].x - pointSel.x) * (cornerPointsLast->points[j].x - pointSel.x) +
            (cornerPointsLast->points[j].y - pointSel.y) * (cornerPointsLast->points[j].y - pointSel.y) +
            (cornerPointsLast->points[j].z - pointSel.z) * (cornerPointsLast->points[j].z - pointSel.z);

        if (pointSqDis < minPointSqDis2)
        {
          // find nearer point
          minPointSqDis2 = pointSqDis;
          minPointInd2 = j;
        }
      }
    }
    if (minPointInd2 >= 0)  // both closestPointInd and minPointInd2 is valid
    {
      Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x, cornerPointsSharp->points[i].y,
                                 cornerPointsSharp->points[i].z);
      Eigen::Vector3d last_point_a(cornerPointsLast->points[closestPointInd].x,
                                   cornerPointsLast->points[closestPointInd].y,
                                   cornerPointsLast->points[closestPointInd].z);
      Eigen::Vector3d last_point_b(cornerPointsLast->points[minPointInd2].x, cornerPointsLast->points[minPointInd2].y,
                                   cornerPointsLast->points[minPointInd2].z);

      double s;
      if (DISTORTION)
        s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
      else
        s = 1.0;
      ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
      problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
      corner_correspondence++;
    }
  }

  // find correspondence for plane features
  for (int i = 0; i < surfPointsFlatNum; ++i)
  {
    TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
    kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

    int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
    {
      closestPointInd = pointSearchInd[0];

      // get closest point's scan ID
      int closestPointScanID = int(surfPointsLast->points[closestPointInd].intensity);
      double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

      // search in the direction of increasing scan line
      for (int j = closestPointInd + 1; j < (int)surfPointsLast->points.size(); ++j)
      {
        // if not in nearby scans, end the loop
        if (int(surfPointsLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
          break;

        double pointSqDis = (surfPointsLast->points[j].x - pointSel.x) * (surfPointsLast->points[j].x - pointSel.x) +
                            (surfPointsLast->points[j].y - pointSel.y) * (surfPointsLast->points[j].y - pointSel.y) +
                            (surfPointsLast->points[j].z - pointSel.z) * (surfPointsLast->points[j].z - pointSel.z);

        // if in the same or lower scan line
        if (int(surfPointsLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
        {
          minPointSqDis2 = pointSqDis;
          minPointInd2 = j;
        }
        // if in the higher scan line
        else if (int(surfPointsLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
        {
          minPointSqDis3 = pointSqDis;
          minPointInd3 = j;
        }
      }

      // search in the direction of decreasing scan line
      for (int j = closestPointInd - 1; j >= 0; --j)
      {
        // if not in nearby scans, end the loop
        if (int(surfPointsLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
          break;

        double pointSqDis = (surfPointsLast->points[j].x - pointSel.x) * (surfPointsLast->points[j].x - pointSel.x) +
                            (surfPointsLast->points[j].y - pointSel.y) * (surfPointsLast->points[j].y - pointSel.y) +
                            (surfPointsLast->points[j].z - pointSel.z) * (surfPointsLast->points[j].z - pointSel.z);

        // if in the same or higher scan line
        if (int(surfPointsLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
        {
          minPointSqDis2 = pointSqDis;
          minPointInd2 = j;
        }
        else if (int(surfPointsLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
        {
          // find nearer point
          minPointSqDis3 = pointSqDis;
          minPointInd3 = j;
        }
      }

      if (minPointInd2 >= 0 && minPointInd3 >= 0)
      {
        Eigen::Vector3d curr_point(surfPointsFlat->points[i].x, surfPointsFlat->points[i].y,
                                   surfPointsFlat->points[i].z);
        Eigen::Vector3d last_point_a(surfPointsLast->points[closestPointInd].x,
                                     surfPointsLast->points[closestPointInd].y,
                                     surfPointsLast->points[closestPointInd].z);
        Eigen::Vector3d last_point_b(surfPointsLast->points[minPointInd2].x, surfPointsLast->points[minPointInd2].y,
                                     surfPointsLast->points[minPointInd2].z);
        Eigen::Vector3d last_point_c(surfPointsLast->points[minPointInd3].x, surfPointsLast->points[minPointInd3].y,
                                     surfPointsLast->points[minPointInd3].z);

        double s;
        if (DISTORTION)
          s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
        else
          s = 1.0;
        ceres::CostFunction *cost_function =
            LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
        plane_correspondence++;
      }
    }
  }

  printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);

  if ((corner_correspondence + plane_correspondence) < 10)
  {
    printf("less correspondence! *************************************************\n");
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 4;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

// 将激光点添加到观察列表
void addMatchPointToViews(const std::vector<cv::KeyPoint> &keypoints_1, const std::vector<cv::KeyPoint> &keypoints_2,
                          const cv::Mat &descriptors_1, const std::vector<cv::DMatch> &matches_stereo,
                          const std::vector<cv::Point3d> &points_3d)
{
  //描述子和观察表中的描述子进行匹配
  //
}

// 传感器消息同步处理
void callbackHandle(const sensor_msgs::ImageConstPtr &left_image_msg,
                    const sensor_msgs::CameraInfoConstPtr &left_cam_info_msg,
                    const sensor_msgs::ImageConstPtr &right_image_msg,
                    const sensor_msgs::CameraInfoConstPtr &right_cam_info_msg,
                    const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg)
{
  ROS_INFO("callbackLeftImage Start\n");

  //-- 第零步：读取图像和点云
  cv_bridge::CvImage cv_ptr_1, cv_ptr_2;
  cv_ptr_1 = *cv_bridge::toCvCopy(left_image_msg, left_image_msg->encoding);
  cv_ptr_2 = *cv_bridge::toCvCopy(right_image_msg, right_image_msg->encoding);

  pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*point_cloud_msg, *laserCloudIn);
  std::vector<int> indices;

  //-- 第一步:检测 Oriented FAST 角点位置
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  detector->detect(cv_ptr_1.image, keypoints_1);
  detector->detect(cv_ptr_2.image, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  cv::Mat descriptors_1, descriptors_2;
  descriptor->compute(cv_ptr_1.image, keypoints_1, descriptors_1);
  descriptor->compute(cv_ptr_2.image, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  std::vector<cv::DMatch> matches_stereo;
  robustMatch(descriptors_1, descriptors_2, matches_stereo);
  // cv::drawKeypoints(cv_ptr.image, keypoints, cv_ptr.image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

  //-- 第四步:筛选出匹配点
  std::vector<cv::Point2f> points_1, points_2;
  filterMatchPoints(keypoints_1, keypoints_2, matches_stereo, points_1, points_2);

  //-- 第五步:三角化计算
  cv::Mat points_4d;
  cv::triangulatePoints(base_to_left_camera_pose, base_to_right_camera_pose, points_1, points_2, points_4d);

  //-- 第六步:齐次三维点归一化
  std::vector<cv::Point3d> points_3d;
  normalizeHomogeneousPoints(points_4d, points_3d);

  //-- 第六点一步：将激光点添加到观察列表
  // addMatchPointToViews(keypoints_1, keypoints_2,  descriptors_1, matches_stereo, points_3d);

  //-- 第七步：激光点云预处理
  float startOri, endOri;
  std::vector<pcl::PointCloud<PointType>> laserCloudScans;
  pcl::PointCloud<PointType> laserCloudOut;
  std::vector<int> scanStartInd, scanEndInd;

  pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
  removeClosedPointCloud(*laserCloudIn, *laserCloudIn, MINIMUM_RANGE);
  readPointCloudOrient(*laserCloudIn, startOri, endOri);
  parsePointCloudScans(*laserCloudIn, laserCloudScans, startOri, endOri);
  generateFromPointCloudScans(laserCloudScans, laserCloudOut, scanStartInd, scanEndInd);
  calculatePointCurvature(laserCloudOut);

  //-- 第八步：点云分割点云
  pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>);
  segmentSurfAndConner(scanStartInd, scanEndInd, laserCloudOut, cornerPointsSharp, cornerPointsLessSharp,
                       surfPointsFlat, surfPointsLessFlat);

  //-- 第九步：定位初始化
  static bool isSystemInitial = false;

  static pcl::PointCloud<PointType>::Ptr cornerPointsLast(new pcl::PointCloud<PointType>);
  static pcl::PointCloud<PointType>::Ptr surfPointsLast(new pcl::PointCloud<PointType>);
  static pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointType>());
  static pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointType>());

  if (!isSystemInitial)
  {
    // 初始化
    isSystemInitial = true;
  }
  else
  {
    //-- 第十步：激光里程计优化
    for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
    {
      lidarOdometryOptimism(cornerPointsSharp, surfPointsFlat, cornerPointsLast, surfPointsLast, kdtreeCornerLast,
                            kdtreeSurfLast, para_q, para_t);
    }
    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
    q_w_curr = q_w_curr * q_last_curr;
  }

  //-- 第十一步：更新最新的坐标和地图

  cornerPointsLast = cornerPointsLessSharp;
  surfPointsLast = surfPointsLessFlat;

  kdtreeCornerLast->setInputCloud(cornerPointsLast);
  kdtreeSurfLast->setInputCloud(surfPointsLast);

  //   isLeftImageInitial = true;
  // }

  // static cv::Mat descriptors_last,descriptors_curr;
  // static std::vector<cv::KeyPoint> last_keypoints;
  // static cv::Mat last_descriptors;

  // descriptors_curr = matched_desc_1;
  // if(!isSensorInitial)
  // {
  //   // 初始化
  //   last_cv_ptr = cv_ptr;
  //   last_keypoints = keypoints;
  //   last_descriptors = descriptors;

  //   isLeftImageInitial = true;
  // }
  // else
  // {
  //-- 上一帧左图的描述子，

  //-- 上一帧
  // std::vector<cv::DMatch> matches;
  // cv::Mat matched_desc_last,matched_desc_curr;
  // robustMatch(descriptors_1,descriptors_curr,matches,matched_desc_last,matched_desc_curr);

  // //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  // std::vector<cv::DMatch> matches;
  // matcher->match(descriptors, last_descriptors, matches);

  // //-- 第四步:匹配点对筛选
  // //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  // std::vector<cv::DMatch> good_matches;
  // for (int i = 0; i < descriptors.rows; i++) {
  //   matches_thresh.addDataValue(matches[i].distance);
  //   if (matches[i].distance <= matches_thresh.mean())
  //     good_matches.push_back(matches[i]);
  // }
  // // std::cout << "good_matches rate = " <<
  // 100.0*good_matches.size()/(keypoints.size()+last_keypoints.size()-good_matches.size())<< "%. " << std::endl;
  // // std::cout << "matches_thresh.mean() = " << matches_thresh.mean() << ". " << std::endl;

  // //筛选出匹配点
  // std::vector<cv::Point2f> points1;
  // std::vector<cv::Point2f> points2;
  // for (int i = 0; i < (int) good_matches.size(); i++) {
  //   points1.push_back(keypoints[good_matches[i].queryIdx].pt);
  //   points2.push_back(last_keypoints[good_matches[i].trainIdx].pt);
  // }

  //   //-- 第五步:计算本质矩阵
  //   cv::Mat essential_matrix, R, t;
  //   essential_matrix = cv::findEssentialMat(points1, points2, left_camera_matrix);
  //   std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

  //   //-- 第六步:从本质矩阵中恢复旋转和平移信息.
  //   // 此函数仅在Opencv3中提供
  //   cv::recoverPose(essential_matrix, points1, points2, left_camera_matrix, R, t);
  //   std::cout << "R is " << std::endl << R << std::endl;
  //   std::cout << "t is " << std::endl << t << std::endl;

  //   //-- 第七步:三角化计算
  //   cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
  //   1, 0, 0, 0,
  //   0, 1, 0, 0,
  //   0, 0, 1, 0);

  //   cv::Mat T2 = left_camera_to_right_camera_trans;

  //   cv::Mat pts_4d;
  //   cv::triangulatePoints(T1, T2, points1, points2, pts_4d);

  //   std::vector<cv::Point3d> points3d;
  //   for (int i = 0; i < pts_4d.cols; i++) {
  //     cv::Mat x = pts_4d.col(i);
  //     x /= x.at<float>(3, 0); // 归一化
  //     cv::Point3d p(
  //       x.at<float>(0, 0),
  //       x.at<float>(1, 0),
  //       x.at<float>(2, 0)
  //     );
  //     points3d.push_back(p);
  //     depth_filter.addDataValue(x.at<float>(2, 0));
  //   }

  //   //-- 第八步：PnP求解
  //   cv::Mat r2, t2;
  //   cv::solvePnP(points3d, points2, left_camera_matrix, cv::Mat(), r2, t2, false); // 调用OpenCV 的 PnP
  //   求解，可选择EPNP，DLS等方法 cv::Mat R2; cv::Rodrigues(r2, R2); // r为旋转向量形式，用Rodrigues公式转换为矩阵

  //   std::cout << "R2=" << std::endl << R2 << std::endl;
  //   std::cout << "t2=" << std::endl << t2 << std::endl;

  //   -- 最后一步:绘制匹配结果
  //   cv_bridge::CvImage img_goodmatch = cv_ptr;
  //   cv::drawMatches(cv_ptr.image, keypoints, last_cv_ptr.image, last_keypoints, good_matches, img_goodmatch.image);

  // }
  // descriptors_last = descriptors_curr;

  cv_bridge::CvImage img_plot = cv_ptr_1;

  // cv::drawMatches(cv_ptr_1.image, keypoints_1, cv_ptr_2.image, keypoints_2, matches_stereo, img_plot.image);

  int good_point_count = 0;
  for (int i = 0; i < matches_stereo.size(); i++)
  {
    // 第二个图
    cv::Mat pt_trans1 =
        base_to_left_camera_pose * (cv::Mat_<double>(4, 1) << points_3d[i].x, points_3d[i].y, points_3d[i].z, 1);
    cv::Mat pt_trans2 =
        base_to_right_camera_pose * (cv::Mat_<double>(4, 1) << points_3d[i].x, points_3d[i].y, points_3d[i].z, 1);
    float depth1 = pt_trans1.at<double>(2, 0);
    float depth2 = pt_trans2.at<double>(2, 0);
    // if (depth1>0 && depth2>0)
    // {
    //   good_point_count++;
    cv::circle(img_plot.image, keypoints_1[matches_stereo[i].queryIdx].pt, 2, get_color(depth1), 2);
    // }
  }
  std::cout << "good_point_count = " << good_point_count << std::endl;

  pubLeftImageWithFeature.publish(img_plot.toImageMsg());

  sensor_msgs::PointCloud2 laserCloudOutput;
  pcl::toROSMsg(laserCloudOut, laserCloudOutput);
  laserCloudOutput.header.stamp = point_cloud_msg->header.stamp;
  laserCloudOutput.header.frame_id = point_cloud_msg->header.frame_id;
  pubPointCloudWithFeature.publish(laserCloudOutput);

  pcl::toROSMsg(*cornerPointsSharp, laserCloudOutput);
  laserCloudOutput.header.stamp = point_cloud_msg->header.stamp;
  laserCloudOutput.header.frame_id = point_cloud_msg->header.frame_id;
  pubCornerPointsSharp.publish(laserCloudOutput);

  pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutput);
  laserCloudOutput.header.stamp = point_cloud_msg->header.stamp;
  laserCloudOutput.header.frame_id = point_cloud_msg->header.frame_id;
  pubCornerPointsLessSharp.publish(laserCloudOutput);

  pcl::toROSMsg(*surfPointsFlat, laserCloudOutput);
  laserCloudOutput.header.stamp = point_cloud_msg->header.stamp;
  laserCloudOutput.header.frame_id = point_cloud_msg->header.frame_id;
  pubSurfPointsFlat.publish(laserCloudOutput);

  pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutput);
  laserCloudOutput.header.stamp = point_cloud_msg->header.stamp;
  laserCloudOutput.header.frame_id = point_cloud_msg->header.frame_id;
  pubSurfPointsLessFlat.publish(laserCloudOutput);

  // publish odometry
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/velo_init";
  laserOdometry.child_frame_id = "/velo_link";
  laserOdometry.header.stamp = point_cloud_msg->header.stamp;
  laserOdometry.pose.pose.orientation.x = q_w_curr.x();
  laserOdometry.pose.pose.orientation.y = q_w_curr.y();
  laserOdometry.pose.pose.orientation.z = q_w_curr.z();
  laserOdometry.pose.pose.orientation.w = q_w_curr.w();
  laserOdometry.pose.pose.position.x = t_w_curr.x();
  laserOdometry.pose.pose.position.y = t_w_curr.y();
  laserOdometry.pose.pose.position.z = t_w_curr.z();
  pubLaserOdometry.publish(laserOdometry);

  static tf::TransformBroadcaster tf_bc;
  tf::Transform tf_bc_t;
  tf::Quaternion tf_bc_q;
  tf_bc_t.setOrigin(tf::Vector3(t_w_curr(0), t_w_curr(1), t_w_curr(2)));
  tf_bc_q.setW(q_w_curr.w());
  tf_bc_q.setX(q_w_curr.x());
  tf_bc_q.setY(q_w_curr.y());
  tf_bc_q.setZ(q_w_curr.z());
  tf_bc_t.setRotation(tf_bc_q);
  tf_bc.sendTransform(tf::StampedTransform(tf_bc_t, point_cloud_msg->header.stamp, "/velo_init", "/velo_link"));

  ROS_INFO("callbackLeftImage Stop\n");
}

// 激光雷达的参数
void parseLidarType()
{
  printf("Lidar type is %s", lidarType.c_str());
  if (lidarType == "VLP-16")
  {
    N_SCAN = 16;
    MAX_RANGE = 100;
    MIN_RANGE = 0;
    RES_RANGE = 0.03f * 2;
    MAX_ANGLE = 15;
    MIN_ANGLE = -15;
    RES_ANGLE = 2 * 2;
  }
  else if (lidarType == "HDL-32E")
  {
    N_SCAN = 32;
    MAX_RANGE = 100;
    MIN_RANGE = 0;
    RES_RANGE = 0.02f * 2;
    MAX_ANGLE = 10.67f;
    MIN_ANGLE = -30.67f;
    RES_ANGLE = 1.33f * 2;
  }
  else if (lidarType == "HDL-64E")
  {
    N_SCAN = 64;
    MAX_RANGE = 120;
    MIN_RANGE = 0;
    RES_RANGE = 0.02f * 2;
    MAX_ANGLE = 2;
    MIN_ANGLE = -24.8f;
    RES_ANGLE = 0.4f * 2;
  }
  else
  {
    printf("， which is UNRECOGNIZED!!!\n");
    ROS_BREAK();
  }
  printf(".\n");
}

// 主函数
int main(int argc, char **argv)
{
  ros::init(argc, argv, "liso_node");
  ros::NodeHandle nh;

  ROS_INFO("liso Start\n");

  // 解析参数
  nh.param<std::string>("left_image_sub", left_image_sub, "/kitti/camera_color_left/image_rect");
  nh.param<std::string>("left_camera_info_sub", left_camera_info_sub, "/kitti/camera_color_left/camera_info");
  nh.param<std::string>("right_image_sub", right_image_sub, "/kitti/camera_color_right/image_rect");
  nh.param<std::string>("right_camera_info_sub", right_camera_info_sub, "/kitti/camera_color_right/camera_info");
  nh.param<std::string>("point_cloud_sub", point_cloud_sub, "/kitti/velo/pointcloud");

  nh.param<std::string>("left_image_with_feature_pub", left_image_with_feature_pub, "/left_image_with_feature_pub");
  nh.param<std::string>("point_cloud_with_feature_pub", point_cloud_with_feature_pub, "/point_cloud_with_feature_pub");
  nh.param<std::string>("conner_points_pub", conner_points_pub, "/conner_points_pub");
  nh.param<std::string>("conner_points_less_pub", conner_points_less_pub, "/conner_points_less_pub");
  nh.param<std::string>("surf_points_pub", surf_points_pub, "/surf_points_pub");
  nh.param<std::string>("surf_points_less_pub", surf_points_less_pub, "/surf_points_less_pub");
  nh.param<std::string>("laser_odometry_pub", laser_odometry_pub, "/laser_odometry_pub");

  // 雷达参数
  nh.param<std::string>("lidar_type", lidarType, "HDL-64E");
  parseLidarType();

  //机器人的最大半径
  nh.param<double>("minimum_range", MINIMUM_RANGE, RES_RANGE);

  // 订阅话题
  message_filters::Subscriber<sensor_msgs::Image> subLeftImage(nh, left_image_sub, 10);
  message_filters::Subscriber<sensor_msgs::CameraInfo> subLeftCameraInfo(nh, left_camera_info_sub, 10);
  message_filters::Subscriber<sensor_msgs::Image> subRightImage(nh, right_image_sub, 10);
  message_filters::Subscriber<sensor_msgs::CameraInfo> subRightCameraInfo(nh, right_camera_info_sub, 10);
  message_filters::Subscriber<sensor_msgs::PointCloud2> subPointCloud(nh, point_cloud_sub, 10);
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image,
                                    sensor_msgs::CameraInfo, sensor_msgs::PointCloud2>
      sync(subLeftImage, subLeftCameraInfo, subRightImage, subRightCameraInfo, subPointCloud, 10);
  sync.registerCallback(boost::bind(&callbackHandle, _1, _2, _3, _4, _5));

  // 传感器参数
  left_camera_matrix = (cv::Mat_<double>(3, 3) << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0);
  right_camera_matrix = (cv::Mat_<double>(3, 3) << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0);

  stereoDistanceThresh = 718.856 * 0.45 * 2;

  base_to_left_camera_pose = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  base_to_right_camera_pose = (cv::Mat_<double>(3, 4) << 1, 0, 0, -0.45, 0, 1, 0, 0, 0, 0, 1, 0);
  // left_camera_to_lidar_trans = (cv::Mat_<double>(3,4)<<
  //   0, 0, 1, 0.06,
  //  -1, 0, 0, 0,
  //   0,-1, 0, 0);

  detector = cv::ORB::create();
  descriptor = cv::ORB::create();
  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  // 发布话题
  pubLeftImageWithFeature = nh.advertise<sensor_msgs::Image>(left_image_with_feature_pub, 10);
  pubPointCloudWithFeature = nh.advertise<sensor_msgs::PointCloud2>(point_cloud_with_feature_pub, 10);
  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>(conner_points_pub, 10);
  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>(conner_points_less_pub, 10);
  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>(surf_points_pub, 10);
  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>(surf_points_less_pub, 10);
  pubLaserOdometry = nh.advertise<nav_msgs::Odometry>(laser_odometry_pub, 10);
  // cv::Mat points_test;
  // std::vector<cv::Point2f> point_test1,point_test2;
  // point_test1.push_back(cv::Point2f(1,0));
  // point_test2.push_back(cv::Point2f(-1,0));
  // cv::triangulatePoints(base_to_left_camera_pose, base_to_right_camera_pose, point_test1, point_test2, points_test);
  // std::cout << "points_test is " << std::endl << points_test << std::endl;

  ros::spin();

  ROS_INFO("liso Stop\n");
}
