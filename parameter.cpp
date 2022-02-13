#include <iostream>
#include <iomanip>

#include <vector>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <list>
#include <omp.h>
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "parameter.h"


Eigen::Vector3d  initial_gravity;
Eigen::Matrix3d  rota_gravity;


Eigen::Matrix3d imu_to_cam_rota;
Eigen::Vector3d imu_to_cam_trans;

Eigen::Matrix3d ini_imu_to_cam_rota;
Eigen::Vector3d ini_imu_to_cam_trans;


vector<IMUData>  IMU_data_raw;     //imu数据  从数据集一次性读取，或者从ros在线获取

//存储ros的图片数据,可以更改为只保留当前的
vector<double>  pic_time;    
vector<Mat>  color_data_raw;    
vector<Mat>  depth_data_raw;   


//获取数据的锁
pthread_mutex_t mutex_pic;
pthread_mutex_t mutex_imu;
//线程里位姿和变换矩阵的锁
pthread_mutex_t mutex_pose;
pthread_mutex_t mutex_g_R_T;

pthread_mutex_t mutex_current;


int g_global_start=0;
int count_global_opti=0;

boost::mutex mutex_global_opti;
boost::condition_variable mutex_global_opti_condi;



Matrix<double,6,6> cov_bias_noise;
Matrix<double,6,6> cov_bias_instability;


Global_parameter G_parameter;


Global_parameter::Global_parameter()
{
    // cov_raw_imu
    cov_bias_noise.setZero();
    cov_bias_instability.setZero();

    //10 hours calibration result
    //bg ba wihte noise
    cov_bias_noise(0,0)=5.1139567119853316e-04;
    cov_bias_noise(1,1)=6.0610991075244652e-04 ;
    cov_bias_noise(2,2)=5.7900587387976039e-04;
    cov_bias_noise(3,3)=1.3233292212738251e-02;
    cov_bias_noise(4,4)=1.3343115613847923e-02;
    cov_bias_noise(5,5)=1.9740761877965859e-02;

    //bg ba instability
    cov_bias_instability(0,0)=9.4556072822705115e-06;
    cov_bias_instability(1,1)=2.0615735179337626e-05;
    cov_bias_instability(2,2)=1.1398053881235606e-05;
    cov_bias_instability(3,3)=1.9830106675990003e-04;
    cov_bias_instability(4,4)=2.5750643938033958e-04;
    cov_bias_instability(5,5)=5.5799332457334456e-04;

    // //10 hours calibration average result
    // //bg ba wihte noise
    // cov_bias_noise(0,0)=5.6550381861024662e-04;
    // cov_bias_noise(1,1)=5.6550381861024662e-04 ;
    // cov_bias_noise(2,2)=5.6550381861024662e-04;
    // cov_bias_noise(3,3)=1.5439056568184012e-02;
    // cov_bias_noise(4,4)=1.5439056568184012e-02;
    // cov_bias_noise(5,5)=1.5439056568184012e-02;

    // //bg ba instability
    // cov_bias_instability(0,0)=1.3823132114281247e-05;
    // cov_bias_instability(1,1)=1.3823132114281247e-05;
    // cov_bias_instability(2,2)=1.3823132114281247e-05;
    // cov_bias_instability(3,3)=3.3793361023786139e-04;
    // cov_bias_instability(4,4)=3.3793361023786139e-04;
    // cov_bias_instability(5,5)=3.3793361023786139e-04;

    // //2 hours calibration result
    // //bg ba wihte noise
    // cov_bias_noise(0,0)=3.8106230059692028e-04;
    // cov_bias_noise(1,1)=2.9200298095396083e-04;
    // cov_bias_noise(2,2)=2.1578613642294237e-04;
    // cov_bias_noise(3,3)=1.3303700969258581e-02;
    // cov_bias_noise(4,4)=1.3916430771251031e-02;
    // cov_bias_noise(5,5)=2.0199944893677095e-02;

    // //bg ba instability
    // cov_bias_instability(0,0)=1.5344004341408872e-05;
    // cov_bias_instability(1,1)=1.5558996427106962e-05;
    // cov_bias_instability(2,2)=2.6572129177411749e-06;
    // cov_bias_instability(3,3)=1.9780260848072001e-04;
    // cov_bias_instability(4,4)=2.6305990601538999e-04;
    // cov_bias_instability(5,5)=6.4837656724543052e-04;

    // Matrix<double,6,6> danwei=Matrix<double,6,6>::Identity();
    // cov_bias_noise=danwei.norm()*cov_bias_noise.normalized();
    // cout<<cov_bias_noise.norm()<<endl;
    // cout<<danwei.norm()<<endl;
    // cout<<cov_bias_instability.norm()<<endl;
    // cout<<cov_bias_noise.normalized()<<endl;
    // cout<<cov_bias_noise.norm()<<endl;
    // cout<<cov_bias_instability.norm()<<endl;

    // cov_bias_noise=1000*cov_bias_noise;
    // cov_bias_instability=1000*cov_bias_instability;

    // cout<<cov_bias_noise.inverse()<<endl<<endl;
    // cout<<cov_bias_instability.inverse()<<endl;
    // exit(1);


    cv::FileStorage fSettings;
    fSettings = cv::FileStorage(CANSHU, cv::FileStorage::READ); 

//global variable：
    imu_to_cam_rota.setIdentity();
    imu_to_cam_trans.setZero();
    rota_gravity.setIdentity();
   
    Matrix3d rota;  
    // rota<<0.02824729928246547, -0.9995993777573096, -0.0017815921098786047,
    //         -0.028381539190637406, 0.0009795641670923005, -0.9995966830113097, 
    //         0.9991979675301671, 0.028286470993068413, -0.028342498872561517;

    rota<< -0.0270527,  -0.999245, -0.0278698,
        0.0215538, -0.0284566,  0.999363,
        -0.999402,  0.0264348,  0.0223073;




    imu_to_cam_rota=rota;
    imu_to_cam_trans<<0.04777362000000000108, 0.00223730999999999991, 0.00160071000000000008;


	// // Vector3d rota_(  1.44217, -0.801876,  0.769252);  //13
    // Vector3d rota_( 0.753807, -1.78184 , 1.74913);     //14
	// Sophus::SO3d rota2 = Sophus::SO3d::exp(rota_);
    // imu_to_cam_rota=rota2.matrix();
    // imu_to_cam_trans<< -0.0324982, -0.00719391 , -0.0897905;     //14


    // // 初始偏差不能太大，如果比较大，那么结果重力优化不行，否则可以
    // Eigen::Vector3d ro_gravity( 2.5,0.0 , 0.1);
    // Sophus::SO3d G_ro=Sophus::SO3d::exp(ro_gravity);
    // rota=G_ro.matrix();

    ini_imu_to_cam_rota=imu_to_cam_rota;   
    ini_imu_to_cam_trans=imu_to_cam_trans;

    pthread_mutex_init (&mutex_pose,NULL);
    pthread_mutex_init (&mutex_g_R_T,NULL);

    pthread_mutex_init (&mutex_pic,NULL);
    pthread_mutex_init (&mutex_imu,NULL);
    
    pthread_mutex_init (&mutex_imu,NULL);

    //jacobian coefficient
    xishu_visual = fSettings["xishu_visual"];
    xishu_imu_rtv = fSettings["xishu_imu_rtv"];
    xishu_imu_bias_change = fSettings["xishu_imu_bias_change"];
    xishu_plane = fSettings["xishu_plane"];

    //update coefficient
    xishu_V = fSettings["xishu_V"];
    xishu_R = fSettings["xishu_R"];
    xishu_T = fSettings["xishu_T"];

    xishu_bg_d = fSettings["xishu_bg_d"];
    xishu_ba_d = fSettings["xishu_ba_d"];   

    xishu_gravity = fSettings["xishu_gravity"];
    xishu_rote = fSettings["xishu_rote"];
    xishu_trans = fSettings["xishu_trans"];

    //路径参数：
    string dataset_route1 = fSettings["dataset_route"];  
    string imu_file_name1 = fSettings["imu_file_name"];  
    string setting_route1 = fSettings["setting_route"];  
    dataset_route = dataset_route1;  
    imu_file_name=imu_file_name1;
    setting_route = setting_route1;  
    TUMDATASET = fSettings["TUMDATASET"];
   

    //运行参数：
    sliding_window_length= fSettings["sliding_window_length"];    
    ini_window_length= fSettings["ini_window_length"];    

    
    GN_number = fSettings["GN_number"];
    PIC_NUMBER= fSettings["PIC_NUMBER"];
    global_opti= fSettings["global_opti"];

    gravity_norm = fSettings["gravity_norm"];
    drop_wrong_loop_relevant = fSettings["drop_wrong_loop_relevant"];
    drop_corres_length = fSettings["drop_corres_length"];
    
    keyframe_track_threshold= fSettings["keyframe_track_threshold"];

    vixel = fSettings["vixel"];  
    time_delay = fSettings["time_delay"];
   
    //运行标志位:
    on_line_ros= fSettings["on_line_ros"];
    flag_youhua = fSettings["flag_youhua"];
    slove_method = fSettings["slove_method"];
    imu_locality = fSettings["imu_locality"];


    use_cov = fSettings["use_cov"];
    drop_wrong_loop = fSettings["drop_wrong_loop"];
    visual_loop= fSettings["visual_loop"];
    

    //debug：
    exit_thread = fSettings["exit_thread"];
    pose_show_thread = fSettings["pose_show_thread"];
    out_bias = fSettings["out_bias"];
    show_loop = fSettings["show_loop"];
    show_cam_imu=fSettings["show_cam_imu"];
    show_trajectory=fSettings["show_trajectory"];
    show_loop_number = fSettings["show_loop_number"];
    out_transformation = fSettings["out_transformation"];
    out_residual = fSettings["out_residual"];

    save_ply = fSettings["save_ply"];
    save_pic_time= fSettings["save_pic_time"];
    
    exit_flag = fSettings["exit_flag"];
    
    //mapping 
    blur_threshold = fSettings["blur_threshold"];

    //show
    showCaseMode = fSettings["showCaseMode"];


    //camera parameter

    camera_width= fSettings["camera_width"];
    camera_height= fSettings["camera_height"];
    camera_c_fx= fSettings["camera_c_fx"];
    camera_c_fy = fSettings["camera_c_fy"];
    camera_c_cx = fSettings["camera_c_cx"];
    camera_c_cy = fSettings["camera_c_cy"];
    camera_depth_scale = fSettings["camera_depth_scale"];
    camera_maximum_depth= fSettings["camera_maximum_depth"];

    camera_d0 = fSettings["camera_d0"];
    camera_d1 = fSettings["camera_d1"];
    camera_d2 = fSettings["camera_d2"];
    camera_d3 = fSettings["camera_d3"];
    camera_d4 = fSettings["camera_d4"];

    gravity_opti_method = fSettings["gravity_opti_method"];

    //如果视觉优化，那么没有imu定位
    if(flag_youhua==2)
    {
        imu_locality=0;
    }


    fSettings.release();
}