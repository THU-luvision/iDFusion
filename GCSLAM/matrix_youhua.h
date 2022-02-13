#ifndef MATRIX_YOUHUA
#define MATRIX_YOUHUA

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

class  via_temp
{
    public:
        void ini_var();
        void set_size(int delta1,int err1,int j1,int j2);
        void set_zero_big();
        void set_zero_small();

        int delta_row;
        int error_row;
        int jm_row;
        int jm_col;

        Eigen::MatrixXd delta;
        Eigen::MatrixXd err;  
        Eigen::MatrixXd J;
        // Eigen::MatrixXd JTJ;
        // Eigen::MatrixXd JTR;

        Eigen::MatrixXd Jm;
        Eigen::MatrixXd rx;    
    
        Eigen::MatrixXd cov;

        Eigen::MatrixXd Jm_cov;
        Eigen::MatrixXd J_cov;
};

class VIA_ALL
{
    public:
        VIA_ALL(int  count):imu_youhua_count(count){} 
        
        void initial_variable();
        void set_zero_big();
        void set_zero_small();
        void solve_GN();

        int imu_youhua_count;

    
        //只用到这四个
        via_temp via_t_r;
        via_temp via_v2;
        via_temp via_bias;
        via_temp via_g_R_P;

};

#endif