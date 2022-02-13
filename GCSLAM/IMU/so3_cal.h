
#ifndef SO3_CAL_H
#define SO3_CAL_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>

using namespace Eigen;
extern Matrix3d JacobianR(const Vector3d& w);
extern Matrix3d JacobianL(const Vector3d& w);
extern Matrix3d JacobianRInv(const Vector3d& w);
extern Matrix3d Exp(const Vector3d& v);
extern Matrix3d normalizeRotationM (const Matrix3d& R);



#endif