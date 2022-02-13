#include <unistd.h>
#include <pangolin/pangolin.h>
#include <pthread.h>
#include <iostream>
#include <opencv/cv.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <list>
#include <omp.h>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include "sparce_show.h"
#include "GCSLAM/MultiViewGeometry.h"
#include "GCFusion/MobileFusion.h"

#include<GL/gl.h>
#include<GL/glu.h>
#include<GL/glut.h>
#include <GLFW/glfw3.h>
using namespace std;

vector<Pose_flag> g_frame_pose;

// vector<Frame> *g_frame;
pthread_mutex_t mutex_show;

extern Eigen::Matrix3d imu_to_cam_rota;
extern Eigen::Vector3d imu_to_cam_trans;
extern pthread_mutex_t mutex_g_R_T;

extern MobileFusion gcFusion;


void *show_cam_imu(void *ptr)
{
    pangolin::Params windowParams;
    windowParams.Set("SAMPLE_BUFFERS", 0);
    windowParams.Set("SAMPLES", 0);

    //创建一个窗口  名称 窗口大小 (是相机projection的缩放)
    pangolin::CreateWindowAndBind("zdw",640,480,windowParams); 

  
    glutInitWindowPosition(100,100);



    // // 获取当前窗口的尺寸
    // GLint m_viewport[4];
    // glGetIntegerv( GL_VIEWPORT, m_viewport );
    // cout<<"screen width: "<<m_viewport[2]<<" height:"<<m_viewport[3]<<endl;

//   获取屏幕的尺寸
//     std::cout << "glut screen width(DEFUALT): " << GLUT_SCREEN_WIDTH << std::endl;
//     std::cout << "glut screen height(DEFAULT): " << GLUT_SCREEN_HEIGHT << std::endl;
//    std::cout << "glut screen width: " <<  glutGet(GLUT_SCREEN_HEIGHT)<< std::endl;
//     std::cout << "glut screen height: " <<  glutGet(GLUT_SCREEN_WIDTH)<< std::endl;


    //启动深度测试 不显示遮挡的物体
    glEnable(GL_DEPTH_TEST);
    // pangolin::SetFullscreen(0);

    // Define Projection and initial ModelView matrix
    //这里不清楚具体是什么意思,可以看opengl参考
    pangolin::OpenGlRenderState s_cam(
            // 相机投影矩阵 长 宽 fu fv u0 v0
            pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
            //对应的是opengl的gluLookAt,前三个参数为摄像机位置,后三个为看向的位置,最后的参数为正对的轴
            // pangolin::ModelViewLookAt(5,5,5,0,0,0,pangolin::AxisNegZ)
        
            // 第一组eyex, eyey,eyez为相机在世界坐标的位置
            // 第二组centerx,centery,centerz为相机镜头对准的物体在世界坐标的位置
            // 第三组upx,upy,upz 为相机向上的方向在世界坐标中的方向(头顶朝向的方向)
            pangolin::ModelViewLookAt(0.2,-0.2,0.2  ,0,0,0,   0,-1,0)
    );



  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);
  //setBounds 跟opengl的viewport 有关
  //看SimpleDisplay中边界的设置就知道
  pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0.0,1.0,0.0,1.0,-640.0f/480.0f)
          .SetHandler(&handler);

  // 注意单位是m  xtion
  float x_cam = 0.175;
  float y_cam = 0.0225; 
  float z_cam = 0.036;

  //  独立imu
  float x_imu = 0.0385;
  float y_imu =  0.0385;
  float z_imu = 0.025;

//   Eigen::Vector3d r_(0.5,0,0);
//   Sophus::SO3d Rd = Sophus::SO3d::exp(r_);
    
//   Eigen::Matrix3d R_1=Rd.matrix();
//   Eigen::Vector3d t_(0.5*(length-length_imu),  0,  height);


//   Sophus::SE3d SE3_Rt(R_1, t_);    
//   Eigen::MatrixXd  Q_=SE3_Rt.matrix();
//   Eigen::MatrixXd Q_1=Q_.transpose();

//   std::vector<GLfloat > Twc1 = {  Q_1(0,0),Q_1(0,1),Q_1(0,2),Q_1(0,3),
//                                   Q_1(1,0),Q_1(1,1),Q_1(1,2),Q_1(1,3) ,
//                                   Q_1(2,0),Q_1(2,1),Q_1(2,2),Q_1(2,3) ,
//                                   Q_1(3,0),Q_1(3,1),Q_1(3,2),Q_1(3,3) };


  while(!pangolin::ShouldQuit())
  {
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    pangolin::glDrawAxis(3);   // 画出坐标轴  x red  y green z blue
      
    glPushMatrix();

      
    // CAMERA

    glLineWidth(2);
    glColor3f(0.8,0.8,0.8);
    glBegin(GL_LINES);

    //底面矩形
    glVertex3f(-x_cam/2,-y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,-z_cam/2);
    glVertex3f(-x_cam/2,y_cam/2,-z_cam/2);
    glVertex3f(-x_cam/2,y_cam/2,-z_cam/2);
    glVertex3f(-x_cam/2,-y_cam/2,-z_cam/2);

    //顶面矩形
    glVertex3f(-x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,z_cam/2);
    glVertex3f(-x_cam/2,y_cam/2,z_cam/2);
    glVertex3f(-x_cam/2,y_cam/2,z_cam/2);
    glVertex3f(-x_cam/2,-y_cam/2,z_cam/2);

    //底面和顶面的连线
    glVertex3f(-x_cam/2,-y_cam/2,-z_cam/2);
    glVertex3f(-x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,z_cam/2);
    glVertex3f(-x_cam/2,y_cam/2,-z_cam/2);
    glVertex3f(-x_cam/2,y_cam/2,z_cam/2);
    glEnd();

    glBegin(GL_QUADS); //绘制多组独立的填充四边形
    // 画出xoy平面的顶面
    glColor3f(0.5,1.0,0);
    glVertex3f(x_cam/4,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/2,y_cam/2,z_cam/2);
    glVertex3f(x_cam/4,y_cam/2,z_cam/2);
 
    // 画出xoz平面的底面
    glColor3f(0.2,0.6,0.6);
    glVertex3f(x_cam/4,-y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,-z_cam/2);
    glVertex3f(x_cam/2,-y_cam/2,z_cam/2);
    glVertex3f(x_cam/4,-y_cam/2,z_cam/2);

    glEnd();


    // IMU
    // 进行位姿变换

    pangolin::OpenGlMatrix Twc;
    pthread_mutex_lock (&mutex_g_R_T);

 
    // Eigen::Vector3d t_temp=imu_to_cam_trans; 
    Eigen::Vector3d t_temp(-0.01,-(y_cam+z_imu)/2,0);

    GetCurrentOpenGLCameraMatrix(imu_to_cam_rota,t_temp,Twc);
    pthread_mutex_unlock(&mutex_g_R_T);

    // glMultMatrixf(Twc1.data());
    glMultMatrixd(Twc.m);
    //先平移，在旋转，在下面加上imu和相机的旋转就可以了

    glLineWidth(2);
    glColor3f(1.0,1.0,0);
    glBegin(GL_LINES);

    //底面矩形
    glVertex3f(-x_imu/2,-y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,-z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,-z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,-z_imu/2);
    glVertex3f(-x_imu/2,-y_imu/2,-z_imu/2);

    //顶面矩形
    glVertex3f(-x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(-x_imu/2,-y_imu/2,z_imu/2);

    //底面和顶面的连线
    glVertex3f(-x_imu/2,-y_imu/2,-z_imu/2);
    glVertex3f(-x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,-z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,z_imu/2);

    glEnd();

    // 画面 表明方向   
    glBegin(GL_QUADS); //绘制多组独立的填充四边形
    
    // 画出xoy平面的顶面
    glColor3f(0.2,0.6,0.6);
    glVertex3f(-x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(-x_imu/2,y_imu/2,z_imu/2);
 
    // 画出yoz平面的顶面
    glColor3f(0.5,1.0,0);
    glVertex3f(x_imu/2,-y_imu/2,-z_imu/2);
    glVertex3f(x_imu/2,-y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,z_imu/2);
    glVertex3f(x_imu/2,y_imu/2,-z_imu/2);
 
    glEnd();



    // glPointSize(10.0f);
    // glBegin(GL_POINTS);
    // glColor3f(1.0,0.0,0.0);
    // glVertex3f(0.0f,0.0f,0.0f);
    // glVertex3f(1,0,0);
    // glVertex3f(0,2,0);
    // glEnd();

    glPopMatrix();

    // Swap frames and Process Events
    pangolin::FinishFrame();

    sleep(1);
  }
  return 0;
}


void *Viewer_Run(void *ptr)
{
    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowmatchingPoints("menu.Show matching Points",true,true);
    pangolin::Var<bool> menuShowallPoints("menu.Show all Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowFramecorres("menu.Show Framecorres",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
                pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0)
                );  

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    // pangolin::SetFullscreen(1);
    // 显示当前帧的特征点情况
    // cv::namedWindow("ORB-SLAM2: Current Frame");

    pangolin::SetFullscreen(1);

    bool bFollow = true;
    int i=0;
    while(1)
    {
        i++;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        pthread_mutex_lock(&mutex_show);
        if(g_frame_pose.size()>1)
            GetCurrentOpenGLCameraMatrix(g_frame_pose.back().pose,Twc);
        pthread_mutex_unlock(&mutex_show);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        d_cam.Activate(s_cam);
        // // 设置背景色
        // glClear(GL_COLOR_BUFFER_BIT); // 清除背景色
        // glClearColor(1.0f,1.0f,1.0f,1.0f);

        if(menuShowKeyFrames)        
        DrawKeyFrames();
        // DrawCurrentCamera();

        if(menuShowallPoints)
            DrawMapPoints();
        if(menuShowmatchingPoints)
            DrawmatchingPoints();
 

        if(menuShowFramecorres)
        Draw_framecorres();


        pangolin::FinishFrame();

        // cv::Mat im = mpFrameDrawer->DrawFrame();
        // cv::imshow("ORB-SLAM2: Current Frame",im);
        // cv::waitKey(1);

        if(menuReset)
        {
            menuShowKeyFrames = true;
            menuShowmatchingPoints = true;
            menuShowallPoints = true;
            bFollow = true;
            menuFollowCamera = true;
            menuReset = false;
        }
        usleep(10000); //线程休眠10ms  100HZ
    }
}


void DrawMapPoints()
{
    // 均匀的画出局部点云
    glPointSize(2);
    glBegin(GL_POINTS);

    double r_=151/0xFF;
    double g_=255/0xFF;
    double b_=255/0xFF;

    glColor3f(r_,g_,b_); 
    pthread_mutex_lock(&mutex_show);
	for (int i=0;i<g_frame_pose.size();i=i+1)
    {
        // if(g_frame_pose[i].is_keyframe==1)
        // {
            Eigen::Matrix3d R_temp=g_frame_pose[i].pose.matrix().block<3, 3>(0, 0);
            Eigen::Vector3d t_temp=g_frame_pose[i].pose.matrix().block<3, 1>(0, 3);
            for (int j=0;j<g_frame_pose[i].local_points.size();j=j+1)
            {
                Eigen::Vector3d point_=R_temp*g_frame_pose[i].local_points[j]+t_temp;
                glVertex3f(point_(0),point_(1),point_(2));
            }
        // }
    }
    pthread_mutex_unlock(&mutex_show);
    glEnd();

}


void DrawmatchingPoints()
{
    // // 只画出匹配的特征点
    glPointSize(2);
    glBegin(GL_POINTS);

	// 255 231 186	 	192 255 62

    double r_=192/0xFF;
    double g_=255/0xFF;
    double b_=62/0xFF;
    glColor3f(r_,g_,b_); 

	for(int i=0;i<gcFusion.gcSLAM.keyFrameCorrList.size();i=i+1)
    {
        MultiViewGeometry::FrameCorrespondence &temp=gcFusion.gcSLAM.keyFrameCorrList[i];

        for (int j = 0; j < temp.matches.size(); j++)
        {
            // 地图点的坐标
            Eigen::Vector3d pi = temp.frame_ref.local_points[temp.matches[j].queryIdx];
            Eigen::Vector3d pj = temp.frame_new.local_points[temp.matches[j].trainIdx];
        
            Eigen::Matrix3d R_i = temp.frame_ref.pose_sophus[0].matrix().block<3, 3>(0, 0);
            Eigen::Matrix3d R_j =temp.frame_new.pose_sophus[0].matrix().block<3, 3>(0, 0);

            Eigen::Vector3d P_i = temp.frame_ref.pose_sophus[0].matrix().block<3, 1>(0, 3);
            Eigen::Vector3d P_j = temp.frame_new.pose_sophus[0].matrix().block<3, 1>(0, 3);


            // 转换到世界坐标系
            Eigen::Vector3d pi_w=R_i*pi+P_i;
            Eigen::Vector3d pj_w=R_j*pj+P_j;        

            Eigen::Vector3d p_=(pi_w+pj_w)/2.0;
            glVertex3f(p_(0),p_(1),p_(2));
        }
        // cout<<i<<"  asdasd"<<endl;
    }
    glEnd();

}


void Draw_framecorres()
{
    // 通过连线画出 连接新的关键帧的所有帧对
    glPointSize(0.1);
    glBegin(GL_LINES);


    double r_=192/0xFF;
    double g_=255/0xFF;
    double b_=62/0xFF;

    glColor3f(r_,g_,b_); 


    // for(int i=gcFusion.gcSLAM.keyFrameCorrList.size()-1;i<gcFusion.gcSLAM.keyFrameCorrList.size();i=i+1)
    for(int i=0;i<gcFusion.gcSLAM.keyFrameCorrList.size();i=i+1)
    {
        MultiViewGeometry::FrameCorrespondence &temp=gcFusion.gcSLAM.keyFrameCorrList[i];

        if(temp.frame_new.frame_index!=gcFusion.gcSLAM.keyFrameCorrList.back().frame_new.frame_index)
        {
            continue;
        }

        for (int j = 0; j < temp.matches.size(); j=j+1)
        {
            // 地图点的坐标
            Eigen::Vector3d pi = temp.frame_ref.local_points[temp.matches[j].queryIdx];
            Eigen::Vector3d pj = temp.frame_new.local_points[temp.matches[j].trainIdx];
        
            Eigen::Matrix3d R_i = temp.frame_ref.pose_sophus[0].matrix().block<3, 3>(0, 0);
            Eigen::Matrix3d R_j =temp.frame_new.pose_sophus[0].matrix().block<3, 3>(0, 0);

            Eigen::Vector3d P_i = temp.frame_ref.pose_sophus[0].matrix().block<3, 1>(0, 3);
            Eigen::Vector3d P_j = temp.frame_new.pose_sophus[0].matrix().block<3, 1>(0, 3);

            // 转换到世界坐标系
            Eigen::Vector3d pi_w=R_i*pi+P_i;
            Eigen::Vector3d pj_w=R_j*pj+P_j;        

            Eigen::Vector3d p_=(pi_w+pj_w)/2.0;
            glVertex3f(p_(0),p_(1),p_(2));
            glVertex3f(P_i(0),P_i(1),P_i(2));
            glVertex3f(p_(0),p_(1),p_(2));
            glVertex3f(P_j(0),P_j(1),P_j(2));

        }
    }
    glEnd();
}




void DrawKeyFrames()
{
    float w;
    float h;
    float z;
    float color[3];

    pthread_mutex_lock(&mutex_show);
	for (int i=0;i<g_frame_pose.size();i++)
    {
        if(g_frame_pose[i].is_keyframe==1 && g_frame_pose[i].origin_index==0)  //跟踪成功，主序列关键帧
        {
            w = 0.05;
            h = w*0.75;
            z = w*0.6;
    	// 124 252 0 255 255 0
            color[0]=255/0xFF;
            color[1]=255/0xFF;
            color[2]=0/0xFF;

        }   
        else if(g_frame_pose[i].is_keyframe==1 && g_frame_pose[i].tracking_success==0)  //跟踪失败的关键帧
        {
            w = 0.07;
            h = w*0.75;
            z = w*0.6;
            color[0]=1.0;
            color[1]=0.0;
            color[2]=0.0;
        }
        else if(g_frame_pose[i].is_keyframe==0 && g_frame_pose[i].tracking_success==1)//跟踪成功，普通帧
        {
            w = 0.01; 
            h = w*0.75;
            z = w*0.6;
            color[0]=0.0;
            color[1]=1.0;
            color[2]=0.0;
        }
        else if(g_frame_pose[i].is_keyframe==0 && g_frame_pose[i].tracking_success==0) //跟踪失败的普通帧
        {
            w = 0.05; 
            h = w*0.75;
            z = w*0.6;
            color[0]=1.0;
            color[1]=0.0;
            color[2]=0.0;
        }
        else
        {
            
        }
        

        pangolin::OpenGlMatrix Twc;
        GetCurrentOpenGLCameraMatrix( g_frame_pose[i].pose,Twc);

        glPushMatrix();

        glMultMatrixd(Twc.m);

        glLineWidth(4);
        glColor3f(color[0],color[1],color[2]);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
        
    }
    pthread_mutex_unlock(&mutex_show);

    glEnd();
    
}

void DrawCurrentCamera()
{
    const float w = 0.01;
    const float h = w*0.75;
    const float z = w*0.6;

    pthread_mutex_lock(&mutex_show);
	for (int i=0;i<g_frame_pose.size();i++)
    {

        pangolin::OpenGlMatrix Twc;
        GetCurrentOpenGLCameraMatrix( g_frame_pose[i].pose,Twc);

        glPushMatrix();
        glMultMatrixd(Twc.m);

        glLineWidth(3);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();
        glPopMatrix();
    }
    pthread_mutex_unlock(&mutex_show);
}



void GetCurrentOpenGLCameraMatrix( Eigen::Matrix3d R_temp,Eigen::Vector3d t_temp,pangolin::OpenGlMatrix &M)
{
    M.m[0] = R_temp(0,0);
    M.m[1] = R_temp(1,0);
    M.m[2] = R_temp(2,0);
    M.m[3]  = 0.0;

    M.m[4] = R_temp(0,1);
    M.m[5] = R_temp(1,1);
    M.m[6] = R_temp(2,1);
    M.m[7]  = 0.0;

    M.m[8] = R_temp(0,2);
    M.m[9] = R_temp(1,2);
    M.m[10] = R_temp(2,2);
    M.m[11]  = 0.0;

    M.m[12] = t_temp(0);
    M.m[13] = t_temp(1);
    M.m[14] = t_temp(2);
    M.m[15]  = 1.0;
}


void GetCurrentOpenGLCameraMatrix(Sophus::SE3d pose,pangolin::OpenGlMatrix &M)
{
    Eigen::Matrix3d R_temp=pose.matrix().block<3, 3>(0, 0);
    Eigen::Vector3d t_temp=pose.matrix().block<3, 1>(0, 3);

    M.m[0] = R_temp(0,0);
    M.m[1] = R_temp(1,0);
    M.m[2] = R_temp(2,0);
    M.m[3]  = 0.0;

    M.m[4] = R_temp(0,1);
    M.m[5] = R_temp(1,1);
    M.m[6] = R_temp(2,1);
    M.m[7]  = 0.0;

    M.m[8] = R_temp(0,2);
    M.m[9] = R_temp(1,2);
    M.m[10] = R_temp(2,2);
    M.m[11]  = 0.0;

    M.m[12] = t_temp(0);
    M.m[13] = t_temp(1);
    M.m[14] = t_temp(2);
    M.m[15]  = 1.0;
}







