#include <xmmintrin.h>
#include <iostream>
#include <smmintrin.h>
#include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Sparse>
#include "IMU/imudata.h"
#include "IMU/so3_cal.h"
#include <pthread.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <list>
#include <omp.h>
#include <ctime>
#include "matrix_youhua.h"



using namespace std;
using namespace cv;



// 初始化矩阵尺寸
void via_temp::ini_var()
{
    Eigen::MatrixXd delta1(delta_row , 1 );
    Eigen::MatrixXd err1(error_row , 1 );
    Eigen::MatrixXd J1(error_row , delta_row );
    Eigen::MatrixXd J_cov1(error_row , delta_row );
    Eigen::MatrixXd Jm1(jm_row , jm_col );
    Eigen::MatrixXd Jm_cov1(jm_row , jm_col );
    Eigen::MatrixXd rx1(jm_row , 1);
    
    delta=delta1;
    err=err1;
    J=J1;
    J_cov=J_cov1;
    Jm=Jm1;
    Jm_cov=Jm_cov1;
    rx=rx1;

    Eigen::MatrixXd cov1(error_row , error_row);
    cov=cov1;
    cov.setIdentity();
    // Eigen::MatrixXd JTJ1(delta_row , delta_row );
    // Eigen::MatrixXd JTR1(delta_row , 1 );
    // JTJ=JTJ1;
    // JTR=JTR1;
}

void via_temp::set_size(int delta1,int err1,int j1,int j2)
{
    delta_row=delta1;
    error_row=err1;
    jm_row=j1;
    jm_col=j2;
    ini_var();
}

//变量置零
void via_temp::set_zero_big()
{      
    delta.setZero();
    err.setZero();
    J.setZero();
    cov.setIdentity();
    // JTJ.setZero();
    // JTR.setZero();
}

void via_temp::set_zero_small()
{
    Jm.setZero();
    rx.setZero();
}


//************************************初始化变量
void VIA_ALL::initial_variable()
{
    via_g_R_P.set_size(9,9 * (imu_youhua_count - 1),9,9);
    via_bias.set_size(6 *imu_youhua_count,15 * (imu_youhua_count - 1),15,12);
    via_t_r.set_size(6* imu_youhua_count,9 * (imu_youhua_count - 1),9,12);
    via_v2.set_size(3* imu_youhua_count,9 * (imu_youhua_count - 1),9,6);
}

//************************************变量置零
void VIA_ALL::set_zero_big()
{   
    via_t_r.set_zero_big();
    via_v2.set_zero_big();
    via_g_R_P.set_zero_big();
    via_bias.set_zero_big();
}

//************************************变量置零
void VIA_ALL::set_zero_small()
{
    via_t_r.set_zero_small();
    via_v2.set_zero_small();
    via_g_R_P.set_zero_small();
    via_bias.set_zero_small();
}

//************************************变量求解
void VIA_ALL::solve_GN()
{
    //零偏  重力 旋转矩阵
    // GN_calculate(via_bias.J,via_bias.err,via_bias.delta,via_bias.cov);
    GN_calculate(via_g_R_P.J,via_g_R_P.err,via_g_R_P.delta,via_g_R_P.cov);
}


int get_Jacobian_by_numerical()
{
    



}