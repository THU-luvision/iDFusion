#include "imudata.h"
#include "so3_cal.h"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include <Eigen/Sparse>
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
#include "../../parameter.h"

using namespace std;
using namespace Eigen;


int judge_orthogonal(Matrix3d matrix,string a,int flag)
{
    Matrix3d trans=matrix.transpose();
    Matrix3d res=trans*matrix;
    Matrix3d unit;
    unit.setIdentity();
    res=unit-res;

    double norm=res.norm();
    if(flag==1)
    {
        cout<<a<<" "<<norm<<endl;
    }
    else
    {
        if(norm>0.00001)
        {
            cout<<a<<" "<<norm<<endl;
            // cout<<endl<<matrix<<endl<<endl;

            if(norm>0.01)
            {
                exit(1);
            }
        }   

    }
}


int set_orthogonal(Sophus::SE3d &temp)
{
  Eigen::Matrix3d r1 = temp.matrix().block<3, 3>(0, 0);
  Eigen::Vector3d t1 = temp.matrix().block<3, 1>(0, 3);

  Eigen::JacobiSVD< Eigen::MatrixXd > svd(r1,Eigen::ComputeThinU | Eigen::ComputeThinV); 
  Matrix3d V = svd.matrixV();
  Matrix3d U = svd.matrixU();
  Matrix3d r2;
  //Matrix3f  S = U.inverse() * A * V.transpose().inverse();
  r2=V*U.inverse();
  r1=r2.transpose();
  Sophus::SE3d SE3_Rt(r1, t1);
  temp = SE3_Rt;
}   

int set_orthogonal(Matrix3d &r1)
{
  Eigen::JacobiSVD< Eigen::MatrixXd > svd(r1,Eigen::ComputeThinU | Eigen::ComputeThinV); 
  Matrix3d V = svd.matrixV();
  Matrix3d U = svd.matrixU();
  Matrix3d r2;
  //Matrix3f  S = U.inverse() * A * V.transpose().inverse();
  r2=V*U.inverse();
  r1=r2.transpose();
}




void IMUData::set_all(const IMUData& imudata)
{
    _g=imudata._g;
    _a=imudata._a;
    time_stamp=imudata.time_stamp;
}

void matrix_transform(Eigen::MatrixXd com_matrix, Eigen::SparseMatrix<double> &s_matrix)
{
    s_matrix.resize( com_matrix.rows(),com_matrix.cols());
    for (int i1 = 0; i1 < com_matrix.rows(); ++i1)
    {
        for (int i2 = 0; i2 < com_matrix.cols(); ++i2)
        {
            double temp=com_matrix(i1, i2);
            if ( temp!= 0)
            {
                s_matrix.insert(i1,i2) = temp;
            }
        }
    }
}

void GN_calculate(Eigen::MatrixXd jab_,Eigen::MatrixXd error_,Eigen::MatrixXd &delta ,Eigen::MatrixXd cov,
                    int youhua_method,int cov_flag)
{
    Eigen::MatrixXd JTJ;
    Eigen::MatrixXd JTR;

    //添加协方差的影响
    if(cov_flag==1)
    {
        JTJ=jab_.transpose()*cov.inverse()*jab_;
        JTR=jab_.transpose()*cov.inverse()*error_;
    }
    else
    {
        JTJ=jab_.transpose()*jab_;
        JTR=jab_.transpose()*error_;
    }     

// #ifdef USE_COV
//     //直接在这里使用协方差矩阵是不行的,the matrix is too big ,the error of solving inverse matrix is too big to believe.
//     //maybe this is right. the inverse matrix of the big matrix is right
//     JTJ=jab_.transpose()*cov.inverse()*jab_;
//     JTR=jab_.transpose()*cov.inverse()*error_;
// #else
//     JTJ=jab_.transpose()*jab_;
//     JTR=jab_.transpose()*error_;
// #endif

    delta.setZero();

    if(youhua_method==0)
    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > SimplicialLDLTSolver;
        Eigen::SparseMatrix<double> JTJ_s;
        matrix_transform(JTJ,JTJ_s);
        SimplicialLDLTSolver.compute(JTJ_s);
        delta = SimplicialLDLTSolver.solve(-JTR);
    }
    else
    {
        delta = JTJ.colPivHouseholderQr().solve(-JTR);   
    }
}


int IMUPreintegrator::update(const Vector3d& omega, const Vector3d& acc, const double& dt)
{
    if(dt==0)
    {
        return 0;
    }

    double dt2 = dt*dt;

    Matrix3d dR = Exp(omega*dt);     //rotation vector to matrix
    Matrix3d Jr = JacobianR(omega*dt);
    Matrix3d I3x3 = Matrix3d::Identity();

    Eigen::MatrixXd A(9,9);
    A.setZero();

    A.block<3,3>(0,0) = dR.transpose();//求矩阵的转置
    A.block<3,3>(3,0) = -_delta_R*Sophus::SO3d::hat(acc)*dt;
    A.block<3,3>(6,0) = -0.5*_delta_R*Sophus::SO3d::hat(acc)*dt2;
    A.block<3,3>(3,3) = I3x3;
    A.block<3,3>(6,3) = I3x3*dt;
    A.block<3,3>(6,6) = I3x3;

    // Matrix<double,9,6> B = Matrix<double,9,6>::Zero();
    Eigen::MatrixXd B(9,6);
    B.setZero();

    B.block<3,3>(0,0) = Jr*dt;
    B.block<3,3>(3,3) =_delta_R*dt;
    B.block<3,3>(6,3) =  0.5*_delta_R*dt2;

    Matrix<double,9,9> K_temp;
    K_temp = A*_cov_rvp*A.transpose() + B*cov_bias_noise*B.transpose();
    _cov_rvp=K_temp;

    //对应论文中的70式，通过累加获得零偏的雅格比矩阵 
    _J_P_Biasa += _J_V_Biasa*dt - 0.5*_delta_R*dt2;
    _J_P_Biasg += _J_V_Biasg*dt - 0.5*_delta_R* Sophus::SO3d::hat(acc)*_J_R_Biasg*dt2;
    _J_V_Biasa += -_delta_R*dt;
    _J_V_Biasg += -_delta_R*Sophus::SO3d::hat(acc)*_J_R_Biasg*dt;
    _J_R_Biasg = dR.transpose()*_J_R_Biasg - Jr*dt;


    //对应论文中的33式，通过积分获得运动变化量    这个可以采用更高级的积分方法
    _delta_P += _delta_V*dt + 0.5*_delta_R*acc*dt2;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    _delta_V += _delta_R*acc*dt;
    // _delta_R = normalizeRotationM(_delta_R*dR);  // normalize rotation, in case of numerical error accumulation 

    _delta_R=_delta_R*dR;
    set_orthogonal(_delta_R);  // normalize rotation, in case of numerical error accumulation 

    _delta_time += dt;
    return 1;
}

