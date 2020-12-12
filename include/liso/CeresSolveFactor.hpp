#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct LidarEdgeFactor
{
  LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_, double s_)
	: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_)
  {
  }

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const
  {
	Eigen::Matrix<T, 3, 1> cp{ T(curr_point.x()), T(curr_point.y()), T(curr_point.z()) };
	Eigen::Matrix<T, 3, 1> lpa{ T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z()) };
	Eigen::Matrix<T, 3, 1> lpb{ T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z()) };

	// Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
	Eigen::Quaternion<T> q_last_curr{ q[3], q[0], q[1], q[2] };
	Eigen::Quaternion<T> q_identity{ T(1), T(0), T(0), T(0) };
	q_last_curr = q_identity.slerp(T(s), q_last_curr);
	Eigen::Matrix<T, 3, 1> t_last_curr{ T(s) * t[0], T(s) * t[1], T(s) * t[2] };

	Eigen::Matrix<T, 3, 1> lp;
	lp = q_last_curr * cp + t_last_curr;

	Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
	Eigen::Matrix<T, 3, 1> de = lpa - lpb;

	residual[0] = nu.x() / de.norm();
	residual[1] = nu.y() / de.norm();
	residual[2] = nu.z() / de.norm();

	return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									 const Eigen::Vector3d last_point_b_, const double s_)
  {
	return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(
		new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
  }

  Eigen::Vector3d curr_point, last_point_a, last_point_b;
  double s;
};

struct LidarPlaneFactor
{
  LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d last_point_l_,
				   Eigen::Vector3d last_point_m_, double s_)
	: curr_point(curr_point_)
	, last_point_j(last_point_j_)
	, last_point_l(last_point_l_)
	, last_point_m(last_point_m_)
	, s(s_)
  {
	ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
	ljm_norm.normalize();
  }

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const
  {
	Eigen::Matrix<T, 3, 1> cp{ T(curr_point.x()), T(curr_point.y()), T(curr_point.z()) };
	Eigen::Matrix<T, 3, 1> lpj{ T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z()) };
	// Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
	// Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
	Eigen::Matrix<T, 3, 1> ljm{ T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z()) };

	// Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
	Eigen::Quaternion<T> q_last_curr{ q[3], q[0], q[1], q[2] };
	Eigen::Quaternion<T> q_identity{ T(1), T(0), T(0), T(0) };
	q_last_curr = q_identity.slerp(T(s), q_last_curr);
	Eigen::Matrix<T, 3, 1> t_last_curr{ T(s) * t[0], T(s) * t[1], T(s) * t[2] };

	Eigen::Matrix<T, 3, 1> lp;
	lp = q_last_curr * cp + t_last_curr;

	residual[0] = (lp - lpj).dot(ljm);

	return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									 const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									 const double s_)
  {
	return (new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(
		new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
  }

  Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
  Eigen::Vector3d ljm_norm;
  double s;
};

struct PoseGraph3dFactor
{
  PoseGraph3dFactor(const Eigen::Quaterniond last_curr_q_, Eigen::Vector3d last_curr_p_)
	: last_curr_q(last_curr_q_), last_curr_p(last_curr_p_)
  {
  }

  // q_Quaternion(x, y, z, w), p_Vector3d(x,y,z)
  template <typename T>
  bool operator()(const T *p_a_ptr, const T *q_a_ptr, const T *p_b_ptr, const T *q_b_ptr, T *residuals_ptr) const
  {
	Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
	Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

	Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
	Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

	// 计算两个帧之间的相对变换
	Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
	Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

	// ‎表示在A帧中两个帧之间的位移‎
	Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

	//‎计算两个方向估计值之间的误差
	Eigen::Quaternion<T> delta_q = last_curr_q.template cast<T>() * q_ab_estimated.conjugate();

	//‎计算残差。‎
	// [ position         ]   [ delta_p          ]
	// [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
	Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
	residuals.template block<3, 1>(0, 0) = p_ab_estimated - last_curr_p.template cast<T>();
	residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

	return true;
  }

  static ceres::CostFunction *Create(const Eigen::Quaterniond last_curr_q_, const Eigen::Vector3d last_curr_p_)
  {
	return (new ceres::AutoDiffCostFunction<PoseGraph3dFactor, 6, 3, 4, 3, 4>(
		new PoseGraph3dFactor(last_curr_q_, last_curr_p_)));
  }

  Eigen::Quaterniond last_curr_q;
  Eigen::Vector3d last_curr_p;
};

struct ReprojectionError
{
  ReprojectionError(double observed_x_, double observed_y_, Eigen::Matrix3d camera_matrix_)
	: observed_x(observed_x_), observed_y(observed_y_), camera_matrix(camera_matrix_)
  {
	focal_x = camera_matrix(0, 0);
	focal_y = camera_matrix(1, 1);
  }

  // q_Quaternion(x, y, z, w), p_Vector3d(x,y,z),point(x,y)
  template <typename T>
  bool operator()(const T *p_ptr, const T *q_ptr, const T *point_ptr, T *residuals) const
  {
	Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_pose(p_ptr);
	Eigen::Map<const Eigen::Quaternion<T>> q_pose(q_ptr);
	Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(point_ptr);

	//将点从地图坐标系转换到世界坐标系
	T p[3];
	ceres::QuaternionRotatePoint(q_ptr, point_ptr, p);
	p[0] += p_ptr[0];
	p[1] += p_ptr[1];
	p[2] += p_ptr[2];

	//求出与相机坐标系的夹角
	T xp = -p[0] / p[2];
	T yp = -p[1] / p[2];

	residuals[0] = focal_x * xp - T(observed_x);
	residuals[1] = focal_y * yp - T(observed_y);

	return true;
  }

  static ceres::CostFunction *Create(const double observed_x_, const double observed_y_,
									 const Eigen::Matrix3d camera_matrix_)
  {
	return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 4, 3>(
		new ReprojectionError(observed_x_, observed_y_, camera_matrix_)));
  }

  Eigen::Matrix3d camera_matrix;

  double observed_x, observed_y, focal_x, focal_y;
};