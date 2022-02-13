#ifndef IMUDATA_H
#define IMUDATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include <string.h>


using namespace Eigen;
using namespace std;


extern int judge_orthogonal(Matrix3d matrix,string a,int flag=0);
extern int set_orthogonal(Sophus::SE3d &temp);
extern int set_orthogonal(Matrix3d &r1);

extern void matrix_transform(Eigen::MatrixXd com_matrix, Eigen::SparseMatrix<double> &s_matrix);

extern void GN_calculate(Eigen::MatrixXd jab_,Eigen::MatrixXd error_,Eigen::MatrixXd &delta ,Eigen::MatrixXd cov,
                        int youhua_method=0,int cov_flag=0);

class IMUData
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUData(){}

    IMUData(const IMUData& imudata):_g(imudata._g),_a(imudata._a),time_stamp(imudata.time_stamp) {}

    IMUData(const double& gx, const double& gy, const double& gz,
            const double& ax, const double& ay, const double& az,
            const double& t) :
            _g(gx,gy,gz), _a(ax,ay,az), time_stamp(t)   { }

    void set_all(const IMUData& imudata);
    // Raw data of imu's
    Vector3d    _g;                //gyr data
    Vector3d    _a;                //acc data
    double      time_stamp;        //timestamp
};


typedef Eigen::Matrix<double, 9, 9>  Matrix9d;

class IMUPreintegrator
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    IMUPreintegrator()
    {
        // delta measurements, position/velocity/rotation(matrix)
        _delta_P.setZero();    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        _delta_V.setZero();    // V_k+1 = V_k + R_k*a_k*dt
        _delta_R.setIdentity();    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x
      
        // jacobian of delta measurements w.r.t bias of gyro/acc
        _J_P_Biasg.setZero();     // position / gyro
        _J_P_Biasa.setZero();     // position / acc
        _J_V_Biasg.setZero();     // velocity / gyro
        _J_V_Biasa.setZero();     // velocity / acc
        _J_R_Biasg.setZero();   // rotation / gyro

        // noise covariance propagation of delta measurements
        _cov_rvp.setZero();

        _delta_time = 0;

    }
        
    IMUPreintegrator(const IMUPreintegrator& pre):
        _delta_P(pre._delta_P),
        _delta_V(pre._delta_V),
        _delta_R(pre._delta_R),
        _J_P_Biasg(pre._J_P_Biasg),
        _J_P_Biasa(pre._J_P_Biasa),
        _J_V_Biasg(pre._J_V_Biasg),
        _J_V_Biasa(pre._J_V_Biasa),
        _J_R_Biasg(pre._J_R_Biasg),
        _cov_rvp(pre._cov_rvp),
        _delta_time(pre._delta_time)  { }


    void set_all(const IMUPreintegrator& pre)
    {
        _delta_P=pre._delta_P;
        _delta_V=pre._delta_V;
        _delta_R=pre._delta_R;
       
        _J_P_Biasg=pre._J_P_Biasg;
        _J_P_Biasa=pre._J_P_Biasa;
        _J_V_Biasg=pre._J_V_Biasg;
        _J_V_Biasa=pre._J_V_Biasa;
        _J_R_Biasg=pre._J_R_Biasg;
        _cov_rvp=pre._cov_rvp;
        _delta_time=pre._delta_time;
    }

    // reset to initial state
    void reset()
    {
        // delta measurements, position/velocity/rotation(matrix)
        _delta_P.setZero();    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        _delta_V.setZero();    // V_k+1 = V_k + R_k*a_k*dt
        _delta_R.setIdentity();    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x
       
        // jacobian of delta measurements w.r.t bias of gyro/acc
        _J_P_Biasg.setZero();     // position / gyro
        _J_P_Biasa.setZero();     // position / acc
        _J_V_Biasg.setZero();     // velocity / gyro
        _J_V_Biasa.setZero();     // velocity / acc
        _J_R_Biasg.setZero();   // rotation / gyro

        // noise covariance propagation of delta measurements
        _cov_rvp.setZero();

        _delta_time = 0;

    }

    // incrementally update 1、delta measurements  2、jacobians  3、covariance matrix
    int update(const Vector3d& omega, const Vector3d& acc, const double& dt);


    // delta measurements, position/velocity/rotation(matrix)
    Eigen::Vector3d _delta_P;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    Eigen::Vector3d _delta_V;    // V_k+1 = V_k + R_k*a_k*dt
    Eigen::Matrix3d _delta_R;    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x
  
    // jacobian of delta measurements w.r.t bias of gyro/acc
    Eigen::Matrix3d _J_P_Biasg;     // position / gyro
    Eigen::Matrix3d _J_P_Biasa;     // position / acc
    Eigen::Matrix3d _J_V_Biasg;     // velocity / gyro
    Eigen::Matrix3d _J_V_Biasa;     // velocity / acc
    Eigen::Matrix3d _J_R_Biasg;     // rotation / gyro

    // noise covariance propagation of delta measurements
    Eigen::Matrix<double, 9, 9> _cov_rvp;

    double _delta_time;
    int frame_index_qian;
    int frame_index_hou;
    int imu_index_qian;
    int imu_index_hou;
};


#endif // IMUDATA_H
