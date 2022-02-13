
#include <iostream>
#include "so3_cal.h"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

using namespace Eigen;

// right jacobian of SO(3)
Matrix3d JacobianR(const Vector3d& w)
{
    Matrix3d Jr = Matrix3d::Identity();
    double theta = w.norm();
    if(theta<0.00001)
    {
        return Jr;// = Matrix3d::Identity();
    }
    else
    {
        Vector3d k = w.normalized();  // k - unit direction vector of w
        Matrix3d K = Sophus::SO3d::hat(k);
        Jr =   Matrix3d::Identity() - (1-cos(theta))/theta*K + (1-sin(theta)/theta)*K*K;
    }
    return Jr;
}


Matrix3d JacobianRInv(const Vector3d& w)
{
    Matrix3d Jrinv = Matrix3d::Identity();
    double theta = w.norm();

    // very small angle
    if(theta < 0.00001)
    {
        return Jrinv;
    }
    else
    {
        Vector3d k = w.normalized();  // k - unit direction vector of w
        Matrix3d K = Sophus::SO3d::hat(k);
        Jrinv = Matrix3d::Identity() + 0.5*Sophus::SO3d::hat(w) + ( 1.0 - (1.0+cos(theta))*theta / (2.0*sin(theta)) ) *K*K;
    }
    return Jrinv;
}

// exponential map from vec3 to mat3x3 (Rodrigues formula)
Matrix3d Exp(const Vector3d& v)
{
    return Sophus::SO3d::exp(v).matrix();
}

// left jacobian of SO(3), Jl(x) = Jr(-x)
Matrix3d JacobianL(const Vector3d& w)
{
    return JacobianR(-w);
}
// left jacobian inverse
Matrix3d JacobianLInv(const Vector3d& w)
{
    return JacobianRInv(-w);
}


//四元数中，任意的旋转都可以由两个互为相反数的四元数表示
Quaterniond normalizeRotationQ(const Quaterniond& r)
{
    Quaterniond _r(r);
    if (_r.w()<0)
    {
        _r.coeffs() *= -1;   //保证w的符号为正
    }
    return _r.normalized();
}

//通过旋转矩阵初始化四元数，四元数归一化后再转化为旋转矩阵
Matrix3d normalizeRotationM (const Matrix3d& R)
{
    Quaterniond qr(R);
    return normalizeRotationQ(qr).toRotationMatrix();
}

