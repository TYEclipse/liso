// Copyright 2020, Tang Yin, Nanjing University of Science and Technology

#include "liso/liso_preprocess_thread.h"

PreprocessThread::PreprocessThread(const std::string &lidarType) {
  detector = cv::ORB::create();
  descriptor = cv::ORB::create();
  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  //传感器外参
  left_camera_to_base_pose =
      (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  right_camera_to_base_pose =
      (cv::Mat_<double>(3, 4) << 1, 0, 0, -0.54, 0, 1, 0, 0, 0, 0, 1, 0);

  left_camera_matrix << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0,
      0.0, 1.0;
  right_camera_matrix << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0,
      0.0, 1.0;

  stereoDistanceThresh = 718.856 * 0.54 * 2;

  parseLidarType(lidarType);

  auto bound_run = std::bind(&PreprocessThread::run, this);
  std::thread preprocess_thread(bound_run);
}

void PreprocessThread::run() {
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

    start_time = std::chrono::high_resolution_clock::now();
    printf("preprocessThread Start\n");

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
    removeClosedPointCloud(*laserCloudIn, *laserCloudIn, RES_RANGE);

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

    // //-- 第14步：输出结果给里程计线程
    // odometry_thread_mutex.lock();
    // descriptors_left_buf = descriptors_left;
    // descriptors_right_buf = descriptors_right;
    // imgPoints_left_buf = imgPoints_left;
    // imgPoints_right_buf = imgPoints_right;
    // good_matches_stereo_buf = good_matches_stereo;
    // good_points_3d_buf = good_points_3d;
    // cornerPointsSharp_buf = *cornerPointsSharp;
    // cornerPointsLessSharp_buf = *cornerPointsLessSharp;
    // surfPointsFlat_buf = *surfPointsFlat;
    // surfPointsLessFlat_buf = *surfPointsLessFlat;
    // is_odometry_thread_ready = true;
    // odometry_thread_mutex.unlock();

    // cv_bridge::CvImage img_plot = cv_ptr_1;
    // for (int i = 0; i < imgPoints_left.size(); i++)
    //   cv::circle(img_plot.image, imgPoints_left[i], 2,
    //              get_color(good_points_3d[i].z), 2);
    // pubLeftImageWithFeature.publish(img_plot.toImageMsg());

    // pcl::PointCloud<PointType>::Ptr points_3d_clouds(
    //     new pcl::PointCloud<PointType>);
    // addPoints3DToCloud(good_points_3d, points_3d_clouds);

    // sensor_msgs::PointCloud2 laserCloudOutput;
    // pcl::toROSMsg(*points_3d_clouds, laserCloudOutput);
    // laserCloudOutput.header.stamp = ros::Time::now();
    // laserCloudOutput.header.frame_id = "kitti_base_link";
    // pubCameraPointsCloud.publish(laserCloudOutput);

    // pcl::toROSMsg(*cornerPointsSharp, laserCloudOutput);
    // laserCloudOutput.header.stamp = ros::Time::now();
    // laserCloudOutput.header.frame_id = "kitti_velo_link";
    // pubCornerPointsSharp.publish(laserCloudOutput);

    // pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutput);
    // laserCloudOutput.header.stamp = ros::Time::now();
    // laserCloudOutput.header.frame_id = "kitti_velo_link";
    // pubCornerPointsLessSharp.publish(laserCloudOutput);

    // pcl::toROSMsg(*surfPointsFlat, laserCloudOutput);
    // laserCloudOutput.header.stamp = ros::Time::now();
    // laserCloudOutput.header.frame_id = "kitti_velo_link";
    // pubSurfPointsFlat.publish(laserCloudOutput);

    // pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutput);
    // laserCloudOutput.header.stamp = ros::Time::now();
    // laserCloudOutput.header.frame_id = "kitti_velo_link";
    // pubSurfPointsLessFlat.publish(laserCloudOutput);

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_duration = end_time - start_time;
    printf("preprocessThread End， elapsed_duration = %f ms\n",
           elapsed_duration.count());
  }
}

//带正反检查的描述子匹配
void PreprocessThread::robustMatch(const cv::Mat &queryDescriptors,
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

//根据匹配获得匹配点
void PreprocessThread::filterMatchPoints(
    const std::vector<cv::KeyPoint> &keypoints_1,
    const std::vector<cv::KeyPoint> &keypoints_2,
    const std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &points_1,
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

// 归一化齐次点
float PreprocessThread::normalizeHomogeneousPoints(
    const cv::Mat &points_4d, std::vector<cv::Point3d> &points_3d) {
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

void PreprocessThread::filterUsableKeyPoints(
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

//去除指定半径内的点
template <typename PointT>
void PreprocessThread::removeClosedPointCloud(
    const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out,
    float thres) {
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

// 读取激光点云起始角和结束角
template <typename PointT>
void PreprocessThread::readPointCloudOrient(
    const pcl::PointCloud<PointT> &laserCloudIn, float &startOri,
    float &endOri) {
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

// 读取各行点云
template <typename PointT, typename PointType>
void PreprocessThread::parsePointCloudScans(
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

// 生成整理好的点云和序号
template <typename PointType>
void PreprocessThread::generateFromPointCloudScans(
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

// 计算点云各点的曲率
template <typename PointType>
void PreprocessThread::calculatePointCurvature(
    pcl::PointCloud<PointType> &laserCloudIn) {
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

void PreprocessThread::segmentSurfAndConner(
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

    auto bound_cmp = std::bind(&PreprocessThread::comp, this,
                               std::placeholders::_1, std::placeholders::_2);
    std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, bound_cmp);

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

// 转换像素点
cv::Point2f PreprocessThread::pixel2cam(const cv::Point2d &p,
                                        const Eigen::Matrix3d &K) {
  return cv::Point2f((p.x - K(0, 2)) / K(0, 0), (p.y - K(1, 2)) / K(1, 1));
}

// 激光雷达的参数
void PreprocessThread::parseLidarType(const std::string &lidarType) {
  printf("Lidar type is %s", lidarType.c_str());
  if (lidarType == "VLP-16") {
    preprocess_thread_mutex.lock();
    N_SCAN = 16;
    MAX_RANGE = 100;
    MIN_RANGE = 0;
    RES_RANGE = 0.03f * 2;
    MAX_ANGLE = 15;
    MIN_ANGLE = -15;
    RES_ANGLE = 2 * 2;
    preprocess_thread_mutex.unlock();
  } else if (lidarType == "HDL-32E") {
    preprocess_thread_mutex.lock();
    N_SCAN = 32;
    MAX_RANGE = 100;
    MIN_RANGE = 0;
    RES_RANGE = 0.02f * 2;
    MAX_ANGLE = 10.67f;
    MIN_ANGLE = -30.67f;
    RES_ANGLE = 1.33f * 2;
    preprocess_thread_mutex.unlock();
  } else if (lidarType == "HDL-64E") {
    preprocess_thread_mutex.lock();
    N_SCAN = 64;
    MAX_RANGE = 120;
    MIN_RANGE = 0;
    RES_RANGE = 0.02f * 2;
    MAX_ANGLE = 2;
    MIN_ANGLE = -24.8f;
    RES_ANGLE = 0.4f * 2;
    preprocess_thread_mutex.unlock();
  } else {
    printf("， which is UNRECOGNIZED!!!\n");
    ROS_BREAK();
  }
  printf(".\n");
}