#ifndef SPARCE_SHOW
#define SPARCE_SHOW


#include <unistd.h>
#include <pangolin/pangolin.h>

#include <iostream>
#include <opencv/cv.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <list>
#include <omp.h>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include "GCSLAM/frame.h"


struct Pose_flag
{
    Sophus::SE3d pose;

    int tracking_success;
    int origin_index;
    int is_keyframe;

    std::vector<Eigen::Vector3d , Eigen::aligned_allocator<Eigen::Vector3d>> local_points;
};

extern vector<Pose_flag> g_frame_pose;

// extern vector<Frame> *g_frame;
extern pthread_mutex_t mutex_show;


extern void *Viewer_Run(void *ptr);
extern void *show_cam_imu(void *ptr);

extern void DrawmatchingPoints();
extern void DrawMapPoints();
extern void DrawKeyFrames();
extern void Draw_framecorres();

extern void DrawCurrentCamera();
extern void GetCurrentOpenGLCameraMatrix( Eigen::Matrix3d R_temp,Eigen::Vector3d t_temp,pangolin::OpenGlMatrix &M);
extern void GetCurrentOpenGLCameraMatrix(Sophus::SE3d pose,pangolin::OpenGlMatrix &M);

#endif // DEBUG