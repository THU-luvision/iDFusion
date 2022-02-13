#include "AHCPlaneFitter.hpp"
#include <pangolin/pangolin.h>
#include "peac_plane_detect.h"
#include "../MultiViewGeometry.h"


using namespace std;
using namespace cv;

namespace PLANE
{

//用于onMouse 鼠标点击事件的传参
struct Pic_data
{
    Mat pic;
    int x;
    int y;
    std::vector<std::vector<int>> pMembership;
};

int show_point_cloud(cv::Mat_<cv::Vec3f> cloud,cv::Mat  color,Plane_param plane,int flag);
int cal_plane_param(cv::Mat_<cv::Vec3f> cloud,std::vector<int> point_order,Eigen::Vector3d &X,double &dis);


//验证二维向量的内容： 第一维表示了哪个平面，第二位表示图片中的哪些像素属于这个平面
void onMouse(int event, int x, int y, int flags, void *utsc)
{
	//也可以使用结构体来作为参数 同样的方法
	// cv::Mat image_t = *(cv::Mat*)utsc;
	if (event == CV_EVENT_LBUTTONUP)   
	{
		Pic_data temp_data=*(Pic_data*)utsc;
		cv::Mat image_t=temp_data.pic;

			std::cout << "x:" << x << "  y:" << y << std::endl;	
			std::cout << "rgb:" << image_t.at<cv::Vec3b>(y, x) <<std::endl;   
			//circle(image_t, Point(x, y), 2.5, Scalar(0, 0, 255), 2.5);   
			//imshow("Source Image", image_t);

		int i=0;
		for(i=0;i<temp_data.pMembership.size();i++)
		{
		    vector<int>::iterator it;
		    int value=y*temp_data.pic.cols+x;

		    it=find(temp_data.pMembership[i].begin(),temp_data.pMembership[i].end(),value);
		    if (it!=temp_data.pMembership[i].end())
		    {
                cout<<*it<<endl;
                int x=*it % image_t.cols;
                int y=*it/ image_t.cols;
                cout<<x<<" "<<y<<endl;
		        //vec中存在value值
                cout<<"belong to :"<<i<<endl;
		        break;
		    }
		    else
		    {
		        //vec中不存在value值
		    }
           
		}
	}
}

void show_by_order(string name,Mat pic,int order)
{
    double prop=0.5;
    int  screen_cols =1400; 
    int  screen_rows=1000; 

	char *_name;
	_name = &name[0];
	int rows_number = pic.rows;
	int cols_number = pic.cols;
	int pic_rows, pic_cols;
	pic_rows = rows_number * prop;
	pic_cols = cols_number * prop;

	namedWindow(_name, 0);
	cvResizeWindow(_name, pic_cols, pic_rows);

	int number_cols, pir_row_order, pir_col_order;

	number_cols = screen_cols / pic_cols;  
	
	pir_row_order = order / number_cols;  
	pir_col_order = order%number_cols;   
	 
	// cout<<(pic_cols+30)*pir_col_order<<" "<<(pic_rows + 60)*pir_row_order<<endl;
	cvMoveWindow(_name, (pic_cols+30)*pir_col_order, (pic_rows + 100)*pir_row_order+100);
	imshow(name, pic);
	waitKey(1);
}

struct OrganizedImage3D 
{
    const cv::Mat_<cv::Vec3f>& cloud;
    //note: ahc::PlaneFitter assumes mm as unit!!!
    OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c): cloud(c) {}
    inline int width() const { return cloud.cols; }
    inline int height() const { return cloud.rows; }
    inline bool get(const int row, const int col, double& x, double& y, double& z) const {
        const cv::Vec3f& p = cloud.at<cv::Vec3f>(row,col);
        x = p[0];
        y = p[1];
        z = p[2];
        return z > 0 && std::isnan(z)==0; //return false if current depth is NaN
    }
};
typedef ahc::PlaneFitter< OrganizedImage3D > PlaneFitter;


// 通过 深度图、相机参数 计算点云,提取平面;  通过鼠标事件返回平面序号;  验证获取平面中点的行列
int main_k()
{
    //zr300
    cv::Mat depth = cv::imread("../resource/d.png",cv::IMREAD_ANYDEPTH);
    cv:Mat color = cv::imread("../resource/c.png");
    const float c_cx = 277.317;
    const float c_cy = 223.081;
    const float c_fx = 402.528;
    const float c_fy = 402.567;


    const float max_use_range = 10;

    cv::Mat_<cv::Vec3f> cloud(depth.rows, depth.cols);
    for(int r=0; r<depth.rows; r++)
    {
        const unsigned short* depth_ptr = depth.ptr<unsigned short>(r);
        cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
        for(int c=0; c<depth.cols; c++)
        {
            float z = (float)depth_ptr[c]/1000.0;   //深度图尺度
            if(z>max_use_range){z=0;}
            pt_ptr[c][0] = (c-c_cx)/c_fx*z*1000.0;//m->mm 米转化为毫米
            pt_ptr[c][1] = (r-c_cy)/c_fy*z*1000.0;//m->mm  
            pt_ptr[c][2] = z*1000.0;//m->mm

        }
    }

    PlaneFitter pf;
    pf.minSupport = 3000;
    pf.windowWidth = 20;
    pf.windowHeight = 20;
    pf.doRefine = true;

    std::vector<std::vector<int>> pMembership;
    cv::Mat seg(depth.rows, depth.cols, CV_8UC3);
    OrganizedImage3D Ixyz(cloud);
    pf.run(&Ixyz, &pMembership, &seg);
    std::cout<<"平面个数： "<<pMembership.size()<<std::endl;

    cv::Mat depth_color;
    depth.convertTo(depth_color, CV_8UC1, 50.0/5000);
    applyColorMap(depth_color, depth_color, cv::COLORMAP_JET);

    Plane_param plane;
    plane.point_order=pMembership[2];
    cal_plane_param(cloud,plane.point_order,plane.normal_cam,plane.dis_cam);
    show_point_cloud(cloud,color,plane,1);


    show_by_order("seg",seg,0);
    show_by_order("depth",depth_color,3);

    Pic_data pic_data;
    pic_data.pic=seg;
    pic_data.pMembership=pMembership;
    setMouseCallback("seg", onMouse,&pic_data);
    cv::waitKey();
}


//通过点云和平面向量计算法向和距离 法向是单位法向,距离的单位是m
int cal_plane_param(cv::Mat_<cv::Vec3f> cloud,std::vector<int> point_order,Eigen::Vector3d &X,double &dis)
{
    //注意:不能求解4个未知数,那么求解结果为0; 因为系数整体成比例,所以要先把D设为1

    //求解平面系数
    Eigen::MatrixXd A(point_order.size(), 3 );
    Eigen::MatrixXd B(point_order.size(), 1 );
    A.setZero();
    X.setZero();
    for(int i=0; i<point_order.size(); i++)
    {
        int x=point_order[i] % cloud.cols;
        int y=point_order[i]/ cloud.cols;
        A(i,0)=cloud.at<Vec3f>(y, x)[0];
        A(i,1)=cloud.at<Vec3f>(y, x)[1];
        A(i,2)=cloud.at<Vec3f>(y, x)[2];
        B(i,0)=1;
    }
    X = A.colPivHouseholderQr().solve(B);   
    // cout<<"方程组的解:"<<endl;
    // cout<<X<<endl;

    //归一化得到平面参数 则X[0] X[1] X[3] 表示平面法向 X[3]表示距离
    double sum=pow(X[0],2)+pow(X[1],2)+pow(X[2],2);
    sum=pow(sum,0.5);
    X=X/sum;
    dis=1/sum;     //原本设定到平面的距离为1mm,通过归一化得到真实的原点到平面的距离
    dis=dis/1000;  //mm转化为m
}



//根据图片类型返回类型字符串
string type2str(int type) 
{
	string r;
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
	switch ( depth )
	{
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}
	r += "C";
	r += (chans+'0');
	return r;
}
    
//flag 0 画出点云 1 画出平面 距离和法向量
int show_point_cloud(cv::Mat_<cv::Vec3f> cloud,cv::Mat  color,Plane_param plane,int flag)
{
    // string pic_type=type2str(color.type());
    // cout<<pic_type<<endl;

    //表示点是否在平面上
    Mat flag_plane(color.rows,color.cols,CV_8UC1,cv::Scalar::all(0));
    if(flag==1)
    {
        for(int i=0;i<plane.point_order.size();i++)
        {
                int x=plane.point_order[i] % color.cols;
                int y=plane.point_order[i] / color.cols;
                flag_plane.at<uchar>(y, x)=1;
        }
    }

   pangolin::CreateWindowAndBind("zdw",640,480); 
   glEnable(GL_DEPTH_TEST);
   pangolin::OpenGlRenderState s_cam(
           pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
           pangolin::ModelViewLookAt(10,0,0.1,0,0,0,pangolin::AxisNegZ)
   );
   pangolin::Handler3D handler(s_cam);
   pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0.0,1.0,0.0,1.0,-640.0f/480.0f)
           .SetHandler(&handler);

   while(!pangolin::ShouldQuit())
   {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        glPointSize(1.0f);
        glBegin(GL_POINTS);
        for(int r=0; r<color.rows; r++)
        {
            Vec3b* color_data = color.ptr<Vec3b>(r);
            Vec3f* cloud_data = cloud.ptr<Vec3f>(r);
            for(int c=0; c<color.cols; c++)
            {
                float r_=(float)color_data[c][0]/255.0;
                float g_=(float)color_data[c][1]/255.0;
                float b_=(float)color_data[c][2]/255.0;
                //表明是平面
                if ( flag_plane.at<uchar>(r, c)==1)
                {
                    r_+=0;
                    g_+=0.2;
                    b_+=0;
                }
                glColor3f(r_,g_,b_);
                glVertex3f( cloud_data[c][0]/1000,cloud_data[c][1]/1000,cloud_data[c][2]/1000);
            }
        }
        glEnd();

        //法向量和距离
        if(flag==1)
        {
            glLineWidth(4);
            glColor3f(1.0,1.0,0);
            glBegin(GL_LINES);
            Eigen::Vector3d point_=plane.dis_cam*plane.normal_cam;
            // cout<<point_<<endl;
            glVertex3f(0,0,0);
            glVertex3f(point_[0],point_[1],point_[2]);
            glEnd();
        }

       pangolin::glDrawAxis(3);
       pangolin::FinishFrame();  
   }

   return 0;
}

Mat get_single_plane(int height,int width,std::vector<int> point_order)
{
    Mat pci_t(height, width, CV_8UC1,Scalar(0));
    for(int i=0; i<point_order.size(); i++)
    {
        int x=point_order[i] % width;
        int y=point_order[i]/ width;
        pci_t.at<uchar>(y, x)=255;
    }
    return pci_t;
}

//提取平面,存储在Frame中
int detect_plane(Frame &frame ,MultiViewGeometry::CameraPara camera)
{
    //得到局部点云
    cv::Mat_<cv::Vec3f> cloud(frame.refined_depth.rows, frame.refined_depth.cols);
    for(int r=0; r<frame.refined_depth.rows; r++)
    {
        const float* depth_ptr = frame.refined_depth.ptr<float>(r);
        cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
        for(int c=0; c<frame.refined_depth.cols; c++)
        {
            float z = depth_ptr[c];
            pt_ptr[c][0] = (c-camera.c_cx)/camera.c_fx*z*1000.0;//m->mm 米转化为毫米
            pt_ptr[c][1] = (r-camera.c_cy)/camera.c_fy*z*1000.0;//m->mm  
            pt_ptr[c][2] = z*1000.0;//m->mm
        }
    }
    
    PlaneFitter pf;
    pf.minSupport = 10000;
    pf.windowWidth = 20;
    pf.windowHeight = 20;
    pf.doRefine = true;

    std::vector<std::vector<int>> pMembership;
    cv::Mat seg(frame.refined_depth.rows, frame.refined_depth.cols, CV_8UC3);
    OrganizedImage3D Ixyz(cloud);
    pf.run(&Ixyz, &pMembership, &seg);

    // 平面过滤 去除面积较小 

    //计算平面参数
    frame.seg_plane=seg;
    for(int k=0;k<pMembership.size();k++)
    {
        Plane_param plane_t;
        plane_t.point_order=pMembership[k];
        cal_plane_param(cloud,plane_t.point_order,plane_t.normal_cam,plane_t.dis_cam);
        frame.plane_v.push_back(plane_t);
    }

    // cout<<"平面个数： "<<pMembership.size()<<endl;
    // show_by_order("color",frame.rgb,0);
    // show_by_order("seg",seg,1);
    // cv::waitKey(0);

    // Matrix3d rota_=frame.pose_sophus[0].rotationMatrix();
    // Vector3d trans_=frame.pose_sophus[0].translation();
    return 0;
}

// 求向量夹角 1 返回弧度 0 返回角度
double angle_cal(Vector3d obj1,Vector3d obj2,int return_hudu)
{   
    double temp=obj1.dot(obj2)/(obj1.norm()*obj2.norm());
    double angle=acos(temp);
    if(return_hudu==1)
        return angle;
    else
        return angle*180.0/3.1415926535;
}

//判断向量   1 平行  2 垂直  0 不相关   
int vector_compare(Vector3d obj1,Vector3d obj2)
{
    obj1=obj1/obj1.norm();
    obj2=obj2/obj2.norm();
    if(obj1[0]<0)obj1=-obj1;
    if(obj2[0]<0)obj2=-obj2;

    double cha=5;
    // 求向量夹角
    double angle=angle_cal(obj1,obj2,0);
    //地面
    if(angle<cha)
        return 1;  
    // 墙壁
    else if(angle>(90-cha)&&angle<(90+cha))
        return 2;
    else 
        return 0;
}


/*
关键帧平面提取-> 计算平面参数 -> 平面构建相关 -> 构建平面约束
 done            done         done            
*/

//判断是否是同一个平面
int cal_plane_error(Plane_param plane1,Plane_param plane2,double &ang_int,double &dis_int) 
{
    ang_int=angle_cal(plane1.normal_world,plane2.normal_world);
	dis_int=plane1.dis_world-plane2.dis_world;
    if(dis_int<0)dis_int=-dis_int;

    if(ang_int<1&&dis_int<0.05)
    {
        // cout<<"角度误差:"<<ang_int<<endl;
        // cout<<"距离误差:"<<dis_int<<endl;
        return 1;
    }
    else
        return 0;
}


//根据color图在转换后是否能够重合来判断是否是同一个平面
// int check_plane_color((Plane_param plane1,Plane_param plane2)
// {
	
	
    
// }



// 什么时候使用相机位姿来转换平面
// 每次得到相机位姿的时候,根据相机位姿更新平面参数
// 平面属于frame

}




//  corr_plane.clear();  
//  不清空这个向量,只是检查这个向量中原有元素是否符合条件



