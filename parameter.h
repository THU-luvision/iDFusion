
#ifndef PARAMETER
#define PARAMETER

#include <iostream>
#include <opencv/cv.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "GCSLAM/IMU/imudata.h"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/filesystem.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;


#define MULTI_THREAD 1

#define INPUT_SOURCE_DATABASE 0
#define INPUT_SOURCE_ROS_IMU   1


#define   CANSHU   "../Parameters/canshu.yaml"

extern Eigen::Vector3d  initial_gravity;
extern Eigen::Matrix3d  rota_gravity;

extern Eigen::Matrix3d ini_imu_to_cam_rota;
extern Eigen::Vector3d ini_imu_to_cam_trans;

extern Eigen::Matrix3d imu_to_cam_rota;
extern Eigen::Vector3d imu_to_cam_trans;

extern vector<IMUData>  IMU_data_raw;     //从数据集得到的全部imu数据

extern vector<double>  pic_time;    
extern vector<Mat>  color_data_raw;    
extern vector<Mat>  depth_data_raw;   

extern pthread_mutex_t mutex_pic;
extern pthread_mutex_t mutex_imu;

extern pthread_mutex_t mutex_pose;
extern pthread_mutex_t mutex_g_R_T;

extern pthread_mutex_t mutex_current;

extern int g_global_start;
extern int count_global_opti;
extern boost::mutex mutex_global_opti;
extern boost::condition_variable mutex_global_opti_condi;

extern Matrix<double,6,6> cov_bias_noise;
extern Matrix<double,6,6> cov_bias_instability;



using namespace std;
class Global_parameter
{
 public:

//cost function coefficient
    double xishu_visual ;
    double xishu_imu_rtv ;
    double xishu_imu_bias_change;
    double xishu_plane ;



//update coefficient
    double xishu_V ;
    double xishu_R ;
    double xishu_T ;

    double xishu_bg_d ;
    double xishu_ba_d ;

    double xishu_gravity ;
    double xishu_rote ;
    double xishu_trans ;
 
//route parameter：
    string dataset_route;  
    string imu_file_name;
    int TUMDATASET;
    string setting_route;  


//working parameter
    int sliding_window_length;
    int ini_window_length;    

    int GN_number;
    int PIC_NUMBER;
    int global_opti;

    double gravity_norm ;
    int drop_wrong_loop_relevant;
    int drop_corres_length;
    double keyframe_track_threshold;

    double vixel;  
    double time_delay;

//运行标志位:
    int on_line_ros;
    int flag_youhua ;
    int slove_method ;
    int imu_locality;

    int use_cov;
    int drop_wrong_loop;
    int visual_loop ;  //在部分优化的时候，是否进行视觉的全局优化


//debug：
    int exit_thread;
    int pose_show_thread;
    int out_bias;
    int show_loop ;
    int show_cam_imu;
    int show_trajectory;
    int show_loop_number;
    int out_transformation ;
    int out_residual;

    int save_pic_time;
    int save_ply;

    int exit_flag;    

//mapping
    float blur_threshold;

//show
    int showCaseMode ;

//camera paramter 

    int camera_width;
    int camera_height;
    double camera_c_fx;
    double camera_c_fy ;
    double camera_c_cx ;
    double camera_c_cy ;
    double camera_depth_scale ;
    double camera_maximum_depth;

    double camera_d0 ;
    double camera_d1 ;
    double camera_d2 ;
    double camera_d3 ;
    double camera_d4 ;

    int gravity_opti_method;


    Global_parameter();

};


extern Global_parameter G_parameter;

#endif