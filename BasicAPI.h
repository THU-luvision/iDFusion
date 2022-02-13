#ifndef BASICAPI_H
#define BASICAPI_H


#include "Tools/LogReader.h"
#include "Tools/LiveLogReader.h"
#include "Tools/RawLogReader.h"
#include <librealsense2/rs.hpp>          // Include RealSense Cross Platform API
#include "GCSLAM/MultiViewGeometry.h"
#include <Eigen/Dense>


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

#include "sensor_msgs/Imu.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cv_bridge/cv_bridge.h"
#include <list>
#include <pthread.h>
#include <Eigen/Dense>
#include "string.h"
#include <fstream>
#include <sys/stat.h> 　
#include <sys/types.h> 　


using namespace sensor_msgs;
using namespace message_filters;


using namespace Eigen;



namespace BasicAPI
{
struct vec8
{
  __m256 xmm;

  vec8 (__m256 v) : xmm (v) {}

  vec8 (float v) { xmm = _mm256_set1_ps(v); }

  vec8 (float a, float b, float c, float d, float e, float f, float g, float h)
  { xmm = _mm256_set_ps(h,g,f,e,d,c,b,a); }

  vec8 floor()
  {
      return _mm256_floor_ps(xmm);
  }

  vec8 (const float *v) { xmm = _mm256_load_ps(v); }

  vec8 operator & (const vec8 &v) const
  { return vec8(_mm256_and_ps(xmm,v.xmm)); }


  vec8 operator > (const vec8 &v) const
  { return vec8(_mm256_cmp_ps(xmm,v.xmm,_CMP_GT_OS)); }
  vec8 operator < (const vec8 &v) const
  { return vec8(_mm256_cmp_ps(v.xmm,xmm,_CMP_GT_OS)); }

  vec8 operator* (const vec8 &v) const
  { return vec8(_mm256_mul_ps(xmm, v.xmm)); }

  vec8 operator+ (const vec8 &v) const
  { return vec8(_mm256_add_ps(xmm, v.xmm)); }

  vec8 operator- (const vec8 &v) const
  { return vec8(_mm256_sub_ps(xmm, v.xmm)); }

  vec8 operator/ (const vec8 &v) const
  { return vec8(_mm256_div_ps(xmm, v.xmm)); }

  void operator*= (const vec8 &v)
  { xmm = _mm256_mul_ps(xmm, v.xmm); }

  void operator+= (const vec8 &v)
  { xmm = _mm256_add_ps(xmm, v.xmm); }

  void operator-= (const vec8 &v)
  { xmm = _mm256_sub_ps(xmm, v.xmm); }

  void operator/= (const vec8 &v)
  { xmm = _mm256_div_ps(xmm, v.xmm); }

  void operator>> (float *v)
  { _mm256_store_ps(v, xmm); }

};

// must be initialized
void loadGlobalParameters(MultiViewGeometry::GlobalParameters &g_para, std::string para_file);

// save trajectory
void saveTrajectoryFrameList(std::vector<Frame> &F,std::string fileName);
void saveTrajectoryKeyFrameList(std::vector<Frame> &F,std::string fileName);


void findCubeCorner(Frame &frame_ref, MultiViewGeometry::CameraPara &cameraModel);
// dump PLY model
void savePLYFiles(std::string fileName,
                  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3d> > p,
                  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3d> > color);
void  savePLYFrame(std::string fileName, const Frame &f, const MultiViewGeometry::CameraPara &para);
void saveRefinedFrame(std::string fileName, const Frame &frame_new, const  MultiViewGeometry::CameraPara &para);

int detectAndExtractFeatures(Frame &t, int feature_num, MultiViewGeometry::CameraPara para);



void refineDepthUseNormal(float *normal, float *depth,
                                 float fx, float fy, float cx, float cy,float width, float height);
//eliminate outliers
void refineNewframes(Frame &frame_ref, Frame &frame_new,MultiViewGeometry::CameraPara &cameraModel);

void refineNewframesSIMD(Frame &frame_ref, Frame &frame_new,MultiViewGeometry::CameraPara &cameraModel);
void refineKeyframesSIMD(Frame &frame_ref, Frame &frame_new,MultiViewGeometry::CameraPara &cameraModel);
void refineKeyframes(Frame &frame_ref, Frame &frame_new,MultiViewGeometry::CameraPara &cameraModel);
void refineDepthUseNormalSIMD(float *normal, float *depth,
                                 float fx, float fy, float cx, float cy,float width, float height);
void extractNormalMapSIMD(const cv::Mat &depthMap, cv::Mat &normalMap,
                             float fx, float fy, float cx, float cy);
void checkColorQuality(const cv::Mat &normalMap, cv::Mat &validColorFlag,
                       float fx, float fy, float cx, float cy);
// calculate reprojection error between two frames: (p1+x1) - (p2+x2)





void framePreprocess(Frame &t, MultiViewGeometry::CameraPara &camera);
int LoadOnlineOPENNIData(Frame &fc,
                         LogReader *liveLogReader,
                         MultiViewGeometry::CameraPara &camera);
int LoadOnlineRS2Data(Frame &fc,
                      rs2::pipeline &pipe,
                      MultiViewGeometry::CameraPara &camera);

int LoadRawData(Frame &t,
                const std::vector <std::string> &rgb_files,
                const std::vector<std::string > &depth_files,
                const std::vector<double> &time_stamp,
                MultiViewGeometry::CameraPara &camera);
                
void  spilt_word(std::string ori, std::vector<std::string> &res);
void initOfflineData(std::string work_folder, std::vector <std::string> &rgb_files, std::vector<std::string > &depth_files, std::vector<double> &time_stamp,
    Eigen::MatrixXd &ground_truth);

void  read_camera_param(MultiViewGeometry::CameraPara &camera);



bool DirectoryExists( const char* pzPath );
void makeDir(const std::string & directory);
void printHelpFunctions();
/* sensorType: 0 for offline data
 *             1 for xtion, 2 for realsense
 *
 */
void parse_Input(int argc, char **argv,
              float &ipnutVoxelResolution,
              std::string &basepath,
              MultiViewGeometry::GlobalParameters &para,
              int &sensorType);


void initOpenNICamera(LogReader *logReader, MultiViewGeometry::CameraPara &camera);
int initRS2Camera(rs2::pipeline &pipe, MultiViewGeometry::CameraPara &camera);


int get_ros_data(Frame &fc, MultiViewGeometry::CameraPara &camera);
void  write_pic_time_txt(vector<double>  time,string name);
void  spilt_word1(string ori, vector<double> &res);
void get_imu_data(string lujing);


void callback_imu(const sensor_msgs::ImuConstPtr& gyro_msg,const sensor_msgs::ImuConstPtr& acc_msg);
int callback_cam(const ImageConstPtr& image, const ImageConstPtr& image2);
void *get_d435i_data(void *ptr);
void *get_xtion_imu_data(void *ptr);


void *exit_control(void *ptr);
}
#endif // BASICAPI_HPP
