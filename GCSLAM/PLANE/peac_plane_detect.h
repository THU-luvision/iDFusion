#ifndef PEAC_PLANE_DETECT
#define PEAC_PLANE_DETECT 



namespace PLANE
{
class Plane_param
{
    public:
    std::vector<int> point_order; 
    Eigen::Vector3d normal_cam;
    double dis_cam;
    
    Eigen::Vector3d normal_world;
    double dis_world;
    int flag_=1; 

    //平面参数转换到世界坐标系 只用于平面匹配,在jacobian中直接使用原平面参数和相机位姿
    int transform_plane( Eigen::Matrix3d R,Eigen::Vector3d T)
    {
        flag_=1;
        normal_world=R*normal_cam;
        dis_world=dis_cam-normal_cam.transpose()*R.transpose()*T;
        //转换之后不能保证 dis_world为正的,那么转化为正值后也就是原点到平面的距离
        //这里要保留转换的符号falg,因为优化的是计算dis_world过程中用到的位姿 
        if(dis_world<0)
        {
            normal_world=-normal_world;
            dis_world=-dis_world;
            flag_=-1;
        }
    }
};

double angle_cal(Eigen::Vector3d obj1,Eigen::Vector3d obj2,int return_hudu=0);
int cal_plane_error(Plane_param plane1,Plane_param plane2,double &ang_int,double &dis_int);  // 判断两个平面是否是同一个
void show_by_order(std::string name,cv::Mat pic,int order);
cv::Mat get_single_plane(int height,int width,std::vector<int> point_order);    //画出单个平面


}


#endif // !PEAC_PLANE_DETECT
