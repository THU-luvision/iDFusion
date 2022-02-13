#include "head.h"

#include<GL/gl.h>
#include<GL/glu.h>
#include<GL/glut.h>
#include <GLFW/glfw3.h>

MobileGUI gui(1);
MobileFusion gcFusion;
int end_flag=0;
extern void record_vector(double temp,int count_record,string name,int out_flag=0);

#ifdef SHOW_3D
    pangolin::GlTexture imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
#endif

// 当前帧的位姿等变量，如果直接使用frame vector而不加锁的显示明显会经常出问题
Sophus::SE3d g_pose_sophus_current;
Vector3d g_v_current;
double g_time_current;
int g_tracking_success_current=-1;
int g_origin_index_current=-1;

Vector3d g_bias_g_current(-1,-1,-1);
Vector3d g_bias_a_current(-1,-1,-1);

void *show_frame_pose_thread(void *ptr)
{
    // 等待第一帧
    while(1)
    {
        pthread_mutex_lock(&mutex_current);
        if(g_tracking_success_current!=-1)
        {
            pthread_mutex_unlock(&mutex_current);
            break;
        }
        pthread_mutex_unlock(&mutex_current);
        usleep(100000); //线程休眠100ms  
    }
    
    while(1)
    {

        pthread_mutex_lock (&mutex_current);

        Sophus::SE3d pose_sophus_current=g_pose_sophus_current;
        Vector3d v_current=g_v_current;

        double time_current=g_time_current;
        int tracking_success_current=g_tracking_success_current;
        int origin_index_current=g_origin_index_current;

        Vector3d bias_g_current=g_bias_g_current;
        Vector3d bias_a_current=g_bias_a_current;

        pthread_mutex_unlock(&mutex_current);

        if(gui.followPose->Get())
        {
            Eigen::Matrix4f currPose;

            if(G_parameter.imu_locality && G_parameter.on_line_ros)
            // if(0)
            {
                if(g_bias_g_current[0]==-1)
                {
                    currPose = pose_sophus_current.matrix().cast<float>();
                }
                else
                {
                    //预积分 获得IMUPreInt
                    IMUPreintegrator IMUPreInt;
                    
                    static int count_imu=0;       
                    // 刚好大于当前帧时间的imu数据序号
                    while(IMU_data_raw[count_imu].time_stamp<time_current)
                    {
                        count_imu++;
                        // cout<<count_imu<<"  ";
                    }
                    double last_imu_time=time_current;
                    int count_imu_temp=count_imu;
                    double delta_time=IMU_data_raw.back().time_stamp-time_current;

                    while(count_imu_temp<IMU_data_raw.size())
                    {
                        double dt = IMU_data_raw[count_imu_temp].time_stamp - last_imu_time;

                        Vector3d g_=(IMU_data_raw[count_imu_temp]._g +IMU_data_raw[count_imu_temp-1]._g )/2;
                        Vector3d a_=(IMU_data_raw[count_imu_temp]._a +IMU_data_raw[count_imu_temp-1]._a )/2;
                        IMUPreInt.update(g_ - bias_g_current, a_ - bias_a_current, dt);
                        
                        last_imu_time=IMU_data_raw[count_imu_temp].time_stamp;
                        count_imu_temp++;
                    }

                    Eigen::Matrix3d r1 =pose_sophus_current.matrix().block<3, 3>(0, 0);
                    Eigen::Vector3d t1 =pose_sophus_current.matrix().block<3, 1>(0, 3);

                    pthread_mutex_lock(&mutex_g_R_T);
                    //论文中33式 
                    Eigen::Matrix3d r2 =  r1*imu_to_cam_rota* IMUPreInt._delta_R*imu_to_cam_rota.transpose();
                    Eigen::Vector3d t2 =  r1*imu_to_cam_rota* IMUPreInt._delta_P+v_current*delta_time+0.5*rota_gravity*initial_gravity* delta_time*delta_time
                    +t1+r1*imu_to_cam_trans-r2*imu_to_cam_trans;
                    pthread_mutex_unlock(&mutex_g_R_T);
    
                    Sophus::SE3d SE3_Rt(r2, t2);
                    currPose= SE3_Rt.matrix().cast<float>();

                }
            }
            else
            {
                // currPose =pose_sophus_current.matrix().cast<float>();
                if(tracking_success_current && origin_index_current== 0)
                {
                    currPose =pose_sophus_current.matrix().cast<float>();
                }
            }
    
            gui.setModelView(currPose,0);  // camera.c_fy < 0

        }
        usleep(10000); //线程休眠10ms  100HZ
    }
}

int main(int argc, char *argv[])
{
    string basepath;

    BasicAPI::loadGlobalParameters(MultiViewGeometry::g_para,(char*)G_parameter.setting_route.c_str());
    
    MultiViewGeometry::CameraPara camera;
    BasicAPI::read_camera_param(camera);

    vector <string> rgb_files;
    vector <string> depth_files;
    Eigen::MatrixXd ground_truth;
    vector<double> time_stamp;

    int number_limit;
    if(G_parameter.on_line_ros)
    {
        pthread_t id;
        int ret = pthread_create(&id, NULL, BasicAPI::get_xtion_imu_data, NULL);  
        number_limit=100000;
    }
    else
    {
        basepath = (char*)G_parameter.dataset_route.c_str();
        BasicAPI::get_imu_data(basepath);
        BasicAPI::initOfflineData(basepath, rgb_files, depth_files, time_stamp, ground_truth);
        number_limit=rgb_files.size()-1;
    }
    
    if(G_parameter.on_line_ros && G_parameter.exit_thread)
    {
        pthread_t id1;
        int ret1 = pthread_create(&id1, NULL, BasicAPI::exit_control, NULL);  

    }
        

    gcFusion.initChiselMap(camera,G_parameter.vixel,MultiViewGeometry::g_para.far_plane_distance);
    gcFusion.initGCSLAM(20000,MultiViewGeometry::g_para,camera);

    if(G_parameter.show_trajectory)
    {
        pthread_t id;
        int ret = pthread_create(&id, NULL, Viewer_Run, NULL);  
    }
    if(G_parameter.show_cam_imu)
    {
        pthread_t id;
        int ret = pthread_create(&id, NULL, show_cam_imu, NULL);  
    }
#ifdef SHOW_3D
    if(G_parameter.pose_show_thread)
    {
        pthread_t id;
        int ret = pthread_create(&id, NULL, show_frame_pose_thread, NULL);  
    }
#endif   

    if(G_parameter.global_opti)boost::thread optimization_thread(boost::bind(&MobileFusion::global_optimization,&gcFusion));


#if MULTI_THREAD
    boost::thread map_thread(boost::bind(&MobileFusion::MapManagement,&gcFusion));
#endif

    vector<double> time_pic;

#ifdef SHOW_3D
    // while(!pangolin::ShouldQuit()&& gcFusion.gcSLAM.globalFrameList.size()<number_limit  )
    while(!pangolin::ShouldQuit() )
#else
    while(gcFusion.gcSLAM.globalFrameList.size()<number_limit)
#endif
    {
    double time_start= (double)cv::getTickCount();
    #ifdef SHOW_3D
        if((!gui.pause->Get() || pangolin::Pushed(*gui.step))
        &&gcFusion.gcSLAM.globalFrameList.size()<G_parameter.PIC_NUMBER &&gcFusion.gcSLAM.globalFrameList.size()<(number_limit-1) &&end_flag==0)
    #else
        if(gcFusion.gcSLAM.globalFrameList.size()<G_parameter.PIC_NUMBER && end_flag==0)
    #endif
        {

            Frame f;

            if(G_parameter.on_line_ros)
            {
                int exit_flag_temp=BasicAPI::get_ros_data(f,camera);
                if(exit_flag_temp==-1)continue;
            }
            else
                BasicAPI::LoadRawData(f,rgb_files,depth_files,time_stamp,camera);

            if(G_parameter.save_pic_time)
            {     
                time_pic.push_back(f.time_stamp);
            }

            BasicAPI::detectAndExtractFeatures(f,MultiViewGeometry::g_para.max_feature_num,camera);
            BasicAPI::extractNormalMapSIMD(f.refined_depth, f.normal_map,camera.c_fx,camera.c_fy,camera.c_cx, camera.c_cy);
          

            gcFusion.gcSLAM.update_frame(f);
          
            // 用于位姿显示线程中的显示
            pthread_mutex_lock(&mutex_current);
            int keyframe_length=gcFusion.gcSLAM.KeyframeDataList.size();
            if(keyframe_length>G_parameter.ini_window_length+10)
            {
                int keyframe_index=gcFusion.gcSLAM.KeyframeDataList[keyframe_length-3].keyFrameIndex;
                g_bias_g_current=gcFusion.gcSLAM.globalFrameList[keyframe_index]._BiasGyr+gcFusion.gcSLAM.globalFrameList[keyframe_index]._dBias_g;
                g_bias_a_current=gcFusion.gcSLAM.globalFrameList[keyframe_index]._BiasAcc+gcFusion.gcSLAM.globalFrameList[keyframe_index]._dBias_a;
            }
            g_time_current=gcFusion.gcSLAM.globalFrameList.back().time_stamp;
            g_v_current=gcFusion.gcSLAM.globalFrameList.back()._V;
            g_pose_sophus_current=gcFusion.gcSLAM.globalFrameList.back().pose_sophus[0];
            g_tracking_success_current=gcFusion.gcSLAM.globalFrameList.back().tracking_success;
            g_origin_index_current=gcFusion.gcSLAM.globalFrameList.back().origin_index;
            pthread_mutex_unlock(&mutex_current);

            Frame &frame_current = gcFusion.gcSLAM.globalFrameList.back();
            if(frame_current.tracking_success &&!frame_current.is_keyframe)
            {
                int keyframeIndex = gcFusion.gcSLAM.GetKeyframeDataList().back().keyFrameIndex;
                BasicAPI::refineKeyframesSIMD(gcFusion.gcSLAM.globalFrameList[keyframeIndex],frame_current,camera);
                BasicAPI::refineNewframesSIMD(gcFusion.gcSLAM.globalFrameList[keyframeIndex],frame_current,camera);
            }
            BasicAPI::refineDepthUseNormalSIMD((float *)frame_current.normal_map.data, (float *)frame_current.refined_depth.data,
                                                 camera.c_fx,camera.c_fy, camera.c_cx, camera.c_cy,camera.width, camera.height);

#ifdef SHOW_3D
            if(frame_current.is_keyframe )
            {
                BasicAPI::checkColorQuality(frame_current.normal_map,frame_current.colorValidFlag,camera.c_fx,camera.c_fy,camera.c_cx, camera.c_cy);
                gcFusion.clearRedudentFrameMemory(6);

    #if MULTI_THREAD
                gcFusion.updateGlobalMap(gcFusion.gcSLAM.globalFrameList.size(),gcFusion.gcSLAM.globalFrameList.size() - 1);
    #else

                gcFusion.tsdfFusion(gcFusion.gcSLAM.globalFrameList,
                                    gcFusion.gcSLAM.globalFrameList.size() - 1,
                                    gcFusion.gcSLAM.GetKeyframeDataList(),
                                    gcFusion.gcSLAM.GetKeyframeDataList().size() - 2);
    #endif  
            }

            imageTexture.Upload(gcFusion.gcSLAM.globalFrameList.back().rgb.data,GL_RGB,GL_UNSIGNED_BYTE);   
#endif
            cout<<"frame "<<gcFusion.gcSLAM.globalFrameList.back().frame_index;
            if(frame_current.is_keyframe)cout<<" is keyframe: "<<gcFusion.gcSLAM.KeyframeDataList.size();
            cout<<endl;
          
        }

        static int all_opti=0;
#ifdef SHOW_3D
        if( (gui.pause->Get()&&(all_opti==0))    || (end_flag==1 && all_opti==0)  ||  (all_opti==0&& G_parameter.on_line_ros==0 && gcFusion.gcSLAM.globalFrameList.size()==(number_limit-1) ))  
#else  
        if(  (end_flag==1 && all_opti==0)  ||  (all_opti==0&& G_parameter.on_line_ros==0 && gcFusion.gcSLAM.globalFrameList.size()==(number_limit-1) ))  
#endif 
        {

            if(G_parameter.save_pic_time)
            {   
                BasicAPI::write_pic_time_txt(time_pic,"associate-temp1.txt");
                
            }

            mutex_global_opti_condi.notify_one();
            usleep(1000000);
            // gcFusion.updateGlobalMap(gcFusion.gcSLAM.globalFrameList.size(),gcFusion.gcSLAM.globalFrameList.size() - 1);
            // usleep(3000000);
            // mutex_global_opti_condi.notify_one();
            // usleep(1000000);
            // gcFusion.updateGlobalMap(gcFusion.gcSLAM.globalFrameList.size(),gcFusion.gcSLAM.globalFrameList.size() - 1);
            // usleep(3000000);
            // mutex_global_opti_condi.notify_one();
            // usleep(1000000);

            // gcFusion.updateGlobalMap(gcFusion.gcSLAM.globalFrameList.size(),gcFusion.gcSLAM.globalFrameList.size() - 1);
            // usleep(3000000);

            cout <<"offline re-integrate all frames" << endl;
            gcFusion.inter_all(gcFusion.gcSLAM.globalFrameList,gcFusion.gcSLAM.GetKeyframeDataList());
            cout <<"offline re-integrate all frames done" << endl;

            all_opti=1;
           
            // for(int i = 0; i < gcFusion.gcSLAM.globalFrameList.size(); i++)
            // {
                // gcFusion.IntegrateFrame(gcFusion.gcSLAM.globalFrameList[i]);
            // }
           
        }
        


       
#ifdef SHOW_3D

        if(G_parameter.pose_show_thread==0)
        {
            if(gui.followPose->Get())
            {
                Eigen::Matrix4f currPose;
                // if(gcFusion.gcSLAM.globalFrameList.back().tracking_success && gcFusion.gcSLAM.globalFrameList.back().origin_index == 0)
                // {
                    currPose = gcFusion.gcSLAM.globalFrameList.back().pose_sophus[0].matrix().cast<float>();
                // }
                gui.setModelView(currPose, camera.c_fy < 0);

            }
        }



        gui.PreCall();
        if(gui.drawGlobalModel->Get())
        {
            gcFusion.MobileShow(gui.s_cam.GetProjectionModelViewMatrix(),
                                VERTEX_WEIGHT_THRESHOLD,
                                gui.drawUnstable->Get(),
                                gui.drawNormals->Get(),
                                gui.drawColors->Get(),
                                gui.drawPoints->Get(),
                                gui.drawWindow->Get(),
                                gui.drawTimes->Get(),
                                gcFusion.gcSLAM.globalFrameList.size(),
                                1,
                                gcFusion.gcSLAM.globalFrameList);
        }

        gui.DisplayImg(gui.RGB,&imageTexture);
        gui.PostCall();
       
#endif

        // double time2= (double)cv::getTickCount() - time_start;
        // time2=time2*1000/ cv::getTickFrequency();
        // // cout<<"local optimization duration time: "<<time2<<endl;
        // record_vector(time2,2480,"time");

    }
    
    // gui.~MobileGUI();
    basepath="/home/luvision/Documents/model";
    cout<<"path: "<<basepath<<endl;
    char fileName[2560];
    memset(fileName,0,2560);
    sprintf(fileName,"%s/trajectory.txt",basepath.c_str());
    //BasicAPI::saveTrajectoryFrameList(gcFusion.gcSLAM.globalFrameList,fileName);

    
    if(G_parameter.save_ply)
    {
        memset(fileName,0,2560);
        sprintf(fileName,"%s/OnlineModel_%dmm.ply",basepath.c_str(),(int)(1000 *(gcFusion.GetVoxelResolution())));
        cout << "saving online model to:    " << fileName << endl;
        gcFusion.chiselMap->SaveAllMeshesToPLY(fileName);
        gcFusion.chiselMap->Reset();

        cout <<"offline re-integrate all frames" << endl;
        TICK("Final::IntegrateAllFrames");
        for(int i = 0; i < gcFusion.gcSLAM.globalFrameList.size(); i++)
        {
            gcFusion.IntegrateFrame(gcFusion.gcSLAM.globalFrameList[i]);
        }
        TOCK("Final::IntegrateAllFrames");
        gcFusion.chiselMap->UpdateMeshes(gcFusion.cameraModel);
        memset(fileName,0,2560);
        sprintf(fileName,"%s/finalModelAllframes_%dmm.ply",basepath.c_str(),(int)(1000 *(gcFusion.GetVoxelResolution())));
        cout << "saving offline model to:    " << fileName << endl;
        gcFusion.chiselMap->SaveAllMeshesToPLY(fileName);
        gcFusion.chiselMap->Reset();
    }


    // for(int i = 0; i < gcFusion.gcSLAM.KeyframeDataList.size(); i++)
    // {
    //     MultiViewGeometry::KeyFrameDatabase kfd = gcFusion.gcSLAM.KeyframeDataList[i];
    //     Frame &f = gcFusion.gcSLAM.globalFrameList[kfd.keyFrameIndex];

    //     Eigen::MatrixXf transform = f.pose_sophus[0].matrix().cast<float>();
    //     transform = transform.inverse();
    //     Eigen::Matrix3f r = transform.block<3,3>(0,0);
    //     Eigen::MatrixXf t = transform.block<3,1>(0,3);

    //     memset(fileName,0,2560);
    //     sprintf(fileName,"%s/texture/%06d.cam",basepath.c_str(),i);
    //     FILE * fp = fopen(fileName,"w+");
    //     fprintf(fp,"%f %f %f %f %f %f %f %f %f %f %f %f\r\n",
    //             t(0),t(1),t(2),
    //             r(0,0),r(0,1),r(0,2),
    //             r(1,0),r(1,1),r(1,2),
    //             r(2,0),r(2,1),r(2,2));
    //     fprintf(fp,"%f %f %f %f %f %f",
    //             camera.c_fx / camera.width ,camera.d[0],camera.d[1],
    //             camera.c_fx / camera.c_fy, camera.c_cx / camera.width, camera.c_cy / camera.height);
    //     fclose(fp);
    //     memset(fileName,0,2560);
    //     sprintf(fileName,"%s/texture/%06d.png",basepath.c_str(),i);
    //     cv::imwrite(fileName,f.rgb);

    // }

    cout << "program finish" << endl;

#if MULTI_THREAD
    map_thread.join();
#endif
}



