#ifndef FRAME_H
#define FRAME_H

#include <string>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <memory>
#include "IMU/imudata.h"
#include <sophus/se3.hpp>
#include "PLANE/peac_plane_detect.h"

using namespace Eigen;

typedef Eigen::Vector3i ChunkID;
typedef std::vector<ChunkID, Eigen::aligned_allocator<ChunkID> > ChunkIDList;
typedef std::vector<Eigen::Vector3f , Eigen::aligned_allocator<Eigen::Vector3d> > Point3fList;
typedef std::vector<Eigen::Vector3d , Eigen::aligned_allocator<Eigen::Vector3d> > Point3dList;
typedef std::vector<Sophus::SE3d , Eigen::aligned_allocator<Eigen::Vector3d> > PoseSE3dList;
typedef Sophus::SE3d PoseSE3d;



inline Eigen::Vector3d applyPose( const Sophus::SE3d &pose, const Eigen::Vector3d &point )
{
    return pose.so3() * point + pose.translation();
}


class Frame
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    //平面部分
    cv::Mat seg_plane;
    std::vector<PLANE::Plane_param > plane_v;
    int number_fit; //匹配的特征点的个数，个数越多，视觉部分的优化越可信
    double xishu_v_imu; //视觉和imu优化的系数

    //these only useful for keyframe    
    Vector3d _V;
    Vector3d angular_V;
    // keep unchanged during optimization
    //initialize with the last keyframe
    Vector3d _BiasGyr;   // bias of gyroscope
    Vector3d _BiasAcc;   // bias of accelerometer

    // update below term during optimization
    Vector3d _dBias_g;  // delta bias of gyroscope, correction term computed in optimization
    Vector3d _dBias_a;  // delta bias of accelerometer

    IMUPreintegrator imu_res;

    // dense scene info
    int frame_index;
    int keyframe_index;   // 第几个关键帧
    cv::Mat rgb;
    cv::Mat depth;
    cv::Mat refined_depth;
    cv::Mat normal_map;
    cv::Mat weight;
    cv::Mat colorValidFlag;
    // sparse feature info
    std::vector<cv::KeyPoint > keypoints;
    cv::Mat descriptor;
    Point3dList local_points;



    PoseSE3dList pose_sophus; // pose_sophus[0] for current pose, pose_sophus[1] for next pose
    // time stamp
    double time_stamp;

    int tracking_success;
    int blur_flag;
    int is_keyframe;
    int is_fixed_frame;
    int origin_index;
    float depth_scale;

    int imu_locality;
    float bluriness;     //清晰程度  用在建图的判断

    //for tsdf fusion
    ChunkIDList validChunks;
    std::vector<void *> validChunksPtr;

    int GetOccupiedMemorySize()
    {
    //      printf("memory occupied: %d %d %d %d      %d %d %d %d     %d %d %d %d\r\n",
    //             (rgb.datalimit - rgb.data),
    //             (depth.datalimit - depth.data) ,
    //             (refined_depth.datalimit - refined_depth.data) ,
    //             (normal_map.datalimit - normal_map.data) ,

    //             (weight.datalimit - weight.data) ,
    //             (descriptor.datalimit - descriptor.data) ,
    //             keypoints.size() * sizeof(cv::KeyPoint) ,
    //             feature_tracked_flag.size() * sizeof(unsigned char) ,

    //             local_points.size() * sizeof(Eigen::Vector3d) ,
    //             validChunks.size() * sizeof(ChunkID) ,
    //             pose_sophus.size() * sizeof(Sophus::SE3d),
    //             validChunksPtr.size() * sizeof(void *));
        return ( (rgb.datalimit - rgb.data) +
                (depth.datalimit - depth.data) +
                (refined_depth.datalimit - refined_depth.data) +
                (normal_map.datalimit - normal_map.data) +
                (weight.datalimit - weight.data) +
                (descriptor.datalimit - descriptor.data) +
                keypoints.size() * sizeof(cv::KeyPoint) +
                local_points.size() * sizeof(Eigen::Vector3d) +
                validChunks.size() * sizeof(ChunkID) +
                pose_sophus.size() * sizeof(Sophus::SE3d)+
                validChunksPtr.size() * sizeof(void *) +
                (colorValidFlag.datalimit - colorValidFlag.data)
                                );
    }

    // preserve feature/rgb/depth
    void clear_keyframe_memory()
    {
        depth.release();
        weight.release();
        normal_map.release();
    }

    // preserve local depth
    void clear_redudent_memoery()
    {
        rgb.release();
        colorValidFlag.release();
        depth.release();
        weight.release();
        normal_map.release();
        keypoints.clear();
        descriptor.release();
        local_points.clear();
    }

    // remove frames totally
    void clear_memory()
    {
        rgb.release();
        colorValidFlag.release();
        depth.release();
        refined_depth.release();
        weight.release();
        normal_map.release();
        keypoints.clear();
        descriptor.release();
        local_points.clear();
    }

    Eigen::Vector3d localToGlobal(const Eigen::Vector3d &point)
    {
        return pose_sophus[0].so3() * point + pose_sophus[0].translation();
    }

	Frame()
	{
        
        _V.setZero();
        _BiasGyr.setZero();
        _BiasAcc.setZero();
        _dBias_g.setZero();
        _dBias_a.setZero();

        
		frame_index = 0;
        keyframe_index=0;
		is_fixed_frame = 0;
		origin_index = 1e8;	// no origin by default
		keypoints.clear(); 
        descriptor.release();
		rgb.release();
		depth.release();
        refined_depth.release();
        local_points.clear();
		tracking_success = 0;
		blur_flag = 0;
        is_keyframe = 0;
        imu_locality=0;
        bluriness=500;

        pose_sophus.push_back(Sophus::SE3d());
        pose_sophus.push_back(Sophus::SE3d());
	}
};

class Corr_plane
{
    public:
    Frame frame_1;
    int plane_1;
    int frame_pose1;    // the order in the keyframe sequence
    Frame frame_2;  
    int plane_2;
    int frame_pose2;

    //这里的frame可以改成引用   Frame &frame_1;
};




#endif
