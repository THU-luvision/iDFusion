#include "GCSLAM.h"
#include "../CHISEL/src/open_chisel/Stopwatch.h"
#include "PLANE/peac_plane_detect.h"
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <pthread.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include "../parameter.h"
#include "../sparce_show.h"

using namespace std;





//在imu重定位之后，优化后面的部分
extern pthread_mutex_t mutex_pic;
extern pthread_mutex_t mutex_imu;


namespace MultiViewGeometry
{
 float optimizeKeyFrameMapRobust_global(vector<FrameCorrespondence> &fCList, vector<Frame> &F,
                                std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,
                                int origin, float robust_u,
                                int flag_opti_count=0);
}



namespace PLANE
{ 
  int detect_plane(Frame &frame ,MultiViewGeometry::CameraPara camera);        //平面检测
}

void  vector_write_txt(vector<double> var,string name)
{
    fstream fout;
    char fileName[256];
    sprintf(fileName, "/home/computer/Documents/python_script/data_show/txt/%s.txt",name.c_str());
    fout.open(fileName, ios::out);

    int length=16;
    for(int i =0;i<var.size();i++)
    {
      fout <<setprecision(length) <<var[i]<<'\n';
    }
}

//返回元素第一次出现的位置
template <class T>
int get_position(vector<T> vector_temp,T temp)
{
    typename vector<T>::iterator iElement;
    iElement = find(vector_temp.begin(), vector_temp.end(),temp);
    int nPosition=-1;

    if( iElement != vector_temp.end() )
    {
        // nPosition =iElement-vector_temp.begin();
        nPosition = std::distance(vector_temp.begin(),iElement);
    }
    return nPosition;
}


// 根据name来判断是否是同一个记录; 如果某一个记录达到指定个数，那么输出到文件 也可以直接指定输出到文件
void record_vector(double temp,int count_record,string name,int out_flag)
{
    static vector<string> vector_name;
    static vector<vector<double>> var_record;

    int posi=get_position(vector_name,name);

    if(posi==-1)
    {
        vector_name.push_back(name);
        vector<double> temp1;
        var_record.push_back(temp1);
        posi=get_position(vector_name,name);
    }

    var_record[posi].push_back(temp);

    if(var_record[posi].size()==count_record || out_flag==1)
    {
        vector_write_txt(var_record[posi],name);
    }
}


void GCSLAM::select_closure_candidates(Frame &f, std::vector<int> &candidate_frame_index)
{
  std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
  MILD::LoopClosureDetector & lcd = mild;
  std::vector<Frame> & frame_list = globalFrameList;

  MILD::BayesianFilter spatial_filter;
  std::vector<float > similarity_score;
  lcd.query_database(f.descriptor, similarity_score);
  // select candidates
  candidate_frame_index.clear();
  std::vector<float> salient_score;
  std::vector<MILD::LCDCandidate> candidates;
  spatial_filter.calculateSalientScore(similarity_score, salient_score);

  TICK("GCSLAM::GlobalRegistration");

  //only select top 5, disgard the last frame
  for (int k = 0; k < kflist.size() - 1; k++)
  {
    if (salient_score[k] > salientScoreThreshold &&
        frame_list[kflist[k].keyFrameIndex].is_keyframe)
    {
      MILD::LCDCandidate candidate(salient_score[k],k);
      candidates.push_back(candidate);
    }
  }
  //把回环检测的得分进行排序
  std::sort(candidates.begin(), candidates.end(),greater<MILD::LCDCandidate>());
  //选择前几个  maxCandidateNum是最多选择的个数
  for (int k = 0; k < fmin(candidates.size(), maxCandidateNum); k++)
  {
//    cout << kflist[candidates[k].index].keyFrameIndex << " " << candidates[k].salient_score << endl;
    candidate_frame_index.push_back(candidates[k].index);
  }


  string candidates_str = "candidates: ";
  string candidates_score;
  for (int k = 0; k < candidate_frame_index.size(); k++)
  {
    candidates_str += std::to_string(kflist[candidate_frame_index[k]].keyFrameIndex) + " ";
  }

  //显示回环检测的照片，判断回环检测的正确性

//  cout << "running frame : " << f.frame_index << " " << candidates_str << endl;
}

void GCSLAM::update_keyframe(int newKeyFrameIndex,
                          MultiViewGeometry::FrameCorrespondence &key_frame_corr,
                          float average_disparity,
                          PoseSE3d relative_pose_from_key_to_new,
                          int registration_success)
{
  float scale_change_ratio;

  bool update_keyframe_from_dense_matching =0;
  int global_tracking_success = 0;


  std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrList_keyframes = keyFrameCorrList;
  std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
  MILD::LoopClosureDetector & lcd = mild;
  std::vector<Frame> & frame_list = globalFrameList;
  MILD::SparseMatcher sparseMatcher(FEATURE_TYPE_ORB, 32, 0, 50);
  sparseMatcher.train(frame_list[newKeyFrameIndex].descriptor);
  // loop closure detection
  std::vector<int> candidate_keyframes_;
  std::vector<int> candidate_keyframes;
  TICK("GCSLAM::MILD");
  select_closure_candidates(frame_list[newKeyFrameIndex],candidate_keyframes_);
  TOCK("GCSLAM::MILD");

  TICK("GCSLAM::GlobalRegistration");
  //*********************** add current keyframe to keyframe list
  MultiViewGeometry::KeyFrameDatabase kfd(newKeyFrameIndex);
  kflist.push_back(kfd);

  //*********************** select candidates

  std::vector<MultiViewGeometry::FrameCorrespondence> fCorrCandidate;


  for (size_t k = 0; k < candidate_keyframes_.size(); k++)
  {
    candidate_keyframes.push_back(candidate_keyframes_[k]);
  }


  for (size_t k = 0; k < candidate_keyframes.size(); k++)
  {
    int candidate_frame_index = kflist[candidate_keyframes[k]].keyFrameIndex;
    //根据回环检测的结果构建帧对
    MultiViewGeometry::FrameCorrespondence global_frame_corr(frame_list[candidate_frame_index], frame_list[newKeyFrameIndex]);
    global_frame_corr.long_corresponding=0;
    fCorrCandidate.push_back(global_frame_corr);
//    std::cout << "candidate key frame: " << kflist[candidate_keyframes[k]].keyFrameIndex << std::endl;
  }

  std::vector<float> average_disparity_list(candidate_keyframes.size());
  std::vector<int> registration_success_list(candidate_keyframes.size());
  PoseSE3dList relative_pose_from_ref_to_new_list(candidate_keyframes.size());


  for (size_t k = 0; k < candidate_keyframes.size(); k++)
  {
    int candidate_frame_index = kflist[candidate_keyframes[k]].keyFrameIndex;

    registration_success_list[k] = 1e8;
    average_disparity_list[k] = 1e8;
    registration_success_list[k] = MultiViewGeometry::FrameMatchingTwoViewRGB(fCorrCandidate[k],
                                                                              camera_para,
                                                                              sparseMatcher,
                                                                              relative_pose_from_ref_to_new_list[k],
                                                                              average_disparity_list[k],
                                                                              scale_change_ratio,
                                                                              update_keyframe_from_dense_matching,
                                                                              1,
                                                                              0.01);
    relative_pose_from_ref_to_new_list[k] = relative_pose_from_ref_to_new_list[k].inverse();

  }
  relative_pose_from_ref_to_new_list.push_back(relative_pose_from_key_to_new);
  registration_success_list.push_back(registration_success);
  average_disparity_list.push_back(average_disparity);
  fCorrCandidate.push_back(key_frame_corr); //当前帧和上一个关键帧的帧相关



  for(size_t k = 0; k < registration_success_list.size(); k++)
  {
      if(registration_success_list[k] )
      {
        // cout<<average_disparity_list[k]<<endl;
        kflist.back().corresponding_keyframes.push_back(fCorrCandidate[k].frame_ref.frame_index);
 //         kflist.back().relative_pose_from_key_to_current.push_back(relative_pose_from_ref_to_new_list[k]);
      }
  }
  //update camera pose based on previous results
  float min_average_disparity = 1e9;
  int min_index = 0;
//  std::cout << "average disparity / reprojection error: ";
  for (size_t k = 0; k < fCorrCandidate.size(); k++)
  {
//    std::cout << fCorrCandidate[k].frame_ref.frame_index << "	/"
//      << average_disparity_list[k] << "	/"
//      << registration_success_list[k] << std::endl;

      //除非跟踪失败，只有回环帧对，否则不通过回环帧来进行定位；  在有跟踪帧对的情况下，回环检测的结果只用于优化
      if((frame_list[newKeyFrameIndex].keyframe_index-fCorrCandidate[k].frame_ref.keyframe_index)>15)
      {
        average_disparity_list[k]=average_disparity_list[k]+1000;
      }


      if (min_average_disparity > average_disparity_list[k] && registration_success_list[k])
      {
        min_average_disparity = average_disparity_list[k];
        min_index = k;
        global_tracking_success = 1;
      }

      // cout<<"index: "<<fCorrCandidate[k].frame_ref.frame_index<<
      // "  "<<fCorrCandidate[k].frame_ref.keyframe_index<<"  "<<frame_list[newKeyFrameIndex].keyframe_index<<endl;
    // }
  }
  // cout<<"loop index: "<<fCorrCandidate[min_index].frame_ref.frame_index<<endl;

  int current_map_origin = 0;
  if (global_tracking_success == 1)
  {
    //如果回环检测成功，那么根据最小重投影误差的关键帧来进行当前关键帧的定位
    frame_list[newKeyFrameIndex].tracking_success = 1;
    frame_list[newKeyFrameIndex].pose_sophus[0] = frame_list[fCorrCandidate[min_index].frame_ref.frame_index].pose_sophus[0]
            * relative_pose_from_ref_to_new_list[min_index] ;
  }


  if(!global_tracking_success)
  {
    //如果全局定位不成功
    current_map_origin = newKeyFrameIndex;
//      std::cout << "update anchor keyframe index! " << std::endl;
  }
  else
  {
    std::vector<int> matched_frames;
    for (size_t k = 0; k < fCorrCandidate.size(); k++)
    {
      if (registration_success_list[k] )
      {
        matched_frames.push_back(fCorrCandidate[k].frame_ref.origin_index);
      }
    }
    current_map_origin = *max_element(matched_frames.begin(), matched_frames.end());
  }
//  std::cout << "add new keyframe!" << std::endl;
  frame_list[newKeyFrameIndex].is_keyframe = 1;
  frame_list[newKeyFrameIndex].origin_index = current_map_origin; //这里取最大
  int reg_success_cnt = 0;
  for (size_t k = 0; k < fCorrCandidate.size(); k++)
  {
    if (registration_success_list[k] )
    {
        reg_success_cnt ++;
    }
  }
  if(reg_success_cnt < 4)
  {
      lcd.construct_database(frame_list[newKeyFrameIndex].descriptor);
  }
  else
  {
      cv::Mat descriptor;
      descriptor.release();
      lcd.construct_database(descriptor);
  }

  if(G_parameter.show_loop&&frame_list.size()>G_parameter.show_loop_number)
  {
    char name[20];
    sprintf(name,"%d ",fCorrCandidate[0].frame_new.frame_index);
    PLANE::show_by_order(name,fCorrCandidate[0].frame_new.rgb,0);
  }

  for (size_t k = 0; k < fCorrCandidate.size(); k++)
  {
    //在这里记录当前回环检测的结果，如果imu判别回环检测不成立，那么取消当前的回环检测，并直接把当前帧设置为跟踪失败就可以了
    if (registration_success_list[k] )
    {
      if(G_parameter.show_loop&&frame_list.size()>G_parameter.show_loop_number)
      {
        //显示回环检测得到rgb图片
        char name[20];
        sprintf(name,"%d %d",fCorrCandidate[k].frame_new.frame_index,fCorrCandidate[k].frame_ref.frame_index);
        PLANE::show_by_order(name,fCorrCandidate[k].frame_ref.rgb,k+1);
      }

      //帧对只在这里加入
      fCorrList_keyframes.push_back(fCorrCandidate[k]);
    }
  }

  if(G_parameter.show_loop&& frame_list.size()>G_parameter.show_loop_number)
  {
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  
  updateMapOrigin(fCorrCandidate, registration_success_list,newKeyFrameIndex); //这里取最小
  TOCK("GCSLAM::GlobalRegistration");
 }

//更新origin_index
void GCSLAM::updateMapOrigin(std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrCandidate,
                             std::vector<int> &registration_success_list,
                             int newKeyFrameIndex)
{

    std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
    std::vector<Frame> &frame_list = globalFrameList;

    //这也是以空间换时间
    std::vector<int> keyFrameIndex(frame_list.size());
    for (size_t k = 0; k < frame_list.size(); k++)
    {
      keyFrameIndex[k] = -1;
    }
    for (size_t k = 0; k < kflist.size(); k++)
    {
      keyFrameIndex[kflist[k].keyFrameIndex] = k;
    }
    std::vector<int >tracked_frame_index;
    for (size_t k = 0; k < fCorrCandidate.size(); k++)
    {
      if (registration_success_list[k])
      {
        int ref_frame_index = fCorrCandidate[k].frame_ref.frame_index;
        if (keyFrameIndex[ref_frame_index] < 0)
        {
          std::cout << "warning! ref frame is not keyframe" << std::endl;
        }
        int ref_origin = frame_list[ref_frame_index].origin_index;
        int current_origin = frame_list[newKeyFrameIndex].origin_index;
        frame_list[newKeyFrameIndex].origin_index = std::min(ref_origin,current_origin);

        if(0)
        {
          if (ref_origin < current_origin)
          {
            for (int keyframeCnt = 0; keyframeCnt < kflist.size(); keyframeCnt++)
            {
              if (frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index == current_origin)
              {
                frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index = ref_origin;
                frame_list[kflist[keyframeCnt].keyFrameIndex].is_keyframe = 0;
                for (int localframeCnt = 0; localframeCnt < kflist[keyframeCnt].corresponding_frames.size(); localframeCnt++)
                {
                  frame_list[kflist[keyframeCnt].corresponding_frames[localframeCnt]].origin_index = ref_origin;
                }
              }
            }
          }
          if (current_origin < ref_origin)
          {
            for (int keyframeCnt = 0; keyframeCnt < kflist.size() - 1; keyframeCnt++)
            {
              if (frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index == ref_origin)
              {
                frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index = current_origin;
                frame_list[kflist[keyframeCnt].keyFrameIndex].is_keyframe = 0;
                for (int localframeCnt = 0; localframeCnt < kflist[keyframeCnt].corresponding_frames.size(); localframeCnt++)
                {
                  frame_list[kflist[keyframeCnt].corresponding_frames[localframeCnt]].origin_index = current_origin;
                }
              }
            }
          }
        }

      }
    }
}


void GCSLAM::update_frame(Frame &frame_input)
{
    if(GCSLAM_INIT_FLAG == 0)
    {
        std::cout << "error ! gcSLAM not initialized! " << std::endl;
        exit(1);
    }
    std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrList_keyframes = keyFrameCorrList;
    std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
    std::vector<Frame> &frame_list = globalFrameList;
    MILD::LoopClosureDetector & lcd = mild;


    //*********************** add current frame to database
    frame_list.push_back(frame_input);
    Frame &f = frame_list.back();

    //*********************** init keyframe database
    if (kflist.size() == 0)
    {
      MultiViewGeometry::KeyFrameDatabase kfd(f.frame_index);
      kflist.push_back(kfd);
      lcd.construct_database(f.descriptor);
      f.origin_index = f.frame_index;
      f.tracking_success = 1;
      f.is_fixed_frame = 1;
      f.is_keyframe = 1;
      f.keyframe_index=0;
      
      int count_t1=0;
      pthread_mutex_lock(&mutex_imu);
      while(IMU_data_raw[count_t1].time_stamp<f.time_stamp)
      {
        count_t1++;
      }
      cout<<"使用的第一个imu数据的序号："<<count_t1<<endl;
      Vector3d g_temp;
      g_temp.setZero();
      int length=10;  //取10个的平均值作为重力
      if(count_t1<length/2+1)
      {
        for(int i=0;i<length;i++)
        {
          g_temp=g_temp+IMU_data_raw[count_t1+i]._a;
          // cout<<IMU_data_raw[count_t1+i]._a.transpose()<<endl;
        }
      }
      else
      {
        for(int i=0;i<length;i++)
        {
          g_temp=g_temp+IMU_data_raw[count_t1-length/2+i]._a;
          // cout<<IMU_data_raw[count_t1-length/2+i]._a.transpose()<<endl;
        }
      }
      pthread_mutex_unlock(&mutex_imu);

      g_temp=g_temp/length;
      initial_gravity=-g_temp;                   //注意负号 g_temp是重力加速度 gravity是重力值

      // initial_gravity<< 0.3332, -8.575,-4.7432;
      // initial_gravity<<1, -7,-5;

      initial_gravity=G_parameter.gravity_norm*initial_gravity.normalized();

      initial_gravity=imu_to_cam_rota*initial_gravity;

      cout<<"gravity now is:"<<initial_gravity.transpose()<<endl<<endl;   //自动设置重力的结果
      return;
    }

    int add_new_key_frame_flag = 0;
    bool registration_success = 0;
    float average_disparity = 1e8;
    float scale_change_ratio = 0;


    //*********************** SparseMatcher is used for efficient binary feature matching
    MILD::SparseMatcher sparseMatcher(FEATURE_TYPE_ORB, 32, 0, 50);
    sparseMatcher.train(f.descriptor);

    int last_keyframe_index = kflist.back().keyFrameIndex;
    int anchor_frame = f.frame_index - 1;

    
    PoseSE3d relative_transform_from_key_to_new;

    pthread_mutex_lock(&mutex_pose);
    if(anchor_frame >= 0 && frame_list[anchor_frame].tracking_success)
    {
      relative_transform_from_key_to_new = frame_list[anchor_frame].pose_sophus[0].inverse()*
              frame_list[last_keyframe_index].pose_sophus[0]; 
    }
    pthread_mutex_unlock(&mutex_pose);

    //当前帧和上一个关键帧的帧相关
    MultiViewGeometry::FrameCorrespondence key_frame_corr(frame_list[last_keyframe_index], f);
    bool update_keyframe_from_dense_matching = 0;


    //*********************** Match two frames based on RGBD features
    registration_success = MultiViewGeometry::FrameMatchingTwoViewRGB(key_frame_corr,
                                                                      camera_para,
                                                                      sparseMatcher,
                                                                      relative_transform_from_key_to_new,
                                                                      average_disparity,
                                                                      scale_change_ratio,
                                                                      update_keyframe_from_dense_matching,
                                                                      1,
                                                                      0.01); 


    int update_keyframe_flag = 0;
   
    // if(f.frame_index>=647&&f.frame_index<=655)
    // {
    //   cout<<registration_success<<" "
    //   <<scale_change_ratio<<"  "<<average_disparity<<endl;
    //   registration_success=0;
    // }

    if((average_disparity > minimumDisparity  || (scale_change_ratio > 0.4)) && registration_success)
    {
      // cout<<"update keyframe"<<endl;
      // cout<<average_disparity<<endl;
      // cout<<minimumDisparity<<endl;
      // cout<<scale_change_ratio<<endl;
      update_keyframe_flag = 1;
    }

    static int local_tracking_cnt = 0;

    // double time1= (double)cv::getTickCount();
    //跟踪失败的帧,通过imu设置位姿 所有的帧通过imu来计算定位依靠的关键帧速度
    if(G_parameter.imu_locality&&KeyframeDataList.size()>G_parameter.ini_window_length)
    {
      int k=1;
      while(1)
      {
        //取出前一个跟踪成功并且在主序列的关键帧,以此进行帧位姿的初始化
        int last_keyframe_index =kflist[kflist.size()-k].keyFrameIndex;
        if(frame_list[last_keyframe_index].tracking_success==1&&frame_list[last_keyframe_index].origin_index==0)
        {
          if(!registration_success)
          {
            cout<<"跟踪失败, 定位依靠的关键帧: "<<last_keyframe_index<<"  =  "<<frame_list[last_keyframe_index].frame_index<<endl;
          }
          break;
        }
        else
        {
          k++;
        }
      }

      //根据imu数据初始化  只要注册失败，就根据imu数据进行位姿初始化
      Vector3d _bg=frame_list[last_keyframe_index]._BiasGyr+frame_list[last_keyframe_index]._dBias_g;
      Vector3d _ba=frame_list[last_keyframe_index]._BiasAcc+frame_list[last_keyframe_index]._dBias_a;
      double lasttime=frame_list[last_keyframe_index].time_stamp;
      double delta_time=f.time_stamp-lasttime;

      //预积分 获得IMUPreInt
      IMUPreintegrator IMUPreInt;
      
      static int count_imu=0;       
      // imu变量在下面只进行了读，在其他的地方也没有对raw data的写操作，所以不需要加锁
      // pthread_mutex_lock (&mutex_imu);
      while(IMU_data_raw[count_imu].time_stamp<lasttime)
      {
          count_imu++;
          // cout<<count_imu<<"  ";
      }
      double last_imu_time=lasttime;

      int count_imu_temp=count_imu;

      while(IMU_data_raw[count_imu_temp].time_stamp<f.time_stamp)
      {
          double dt = IMU_data_raw[count_imu_temp].time_stamp - last_imu_time;

          Vector3d g_=(IMU_data_raw[count_imu_temp]._g +IMU_data_raw[count_imu_temp-1]._g )/2;
          Vector3d a_=(IMU_data_raw[count_imu_temp]._a +IMU_data_raw[count_imu_temp-1]._a )/2;
          IMUPreInt.update(g_ - _bg, a_ - _ba, dt);
          
          last_imu_time=IMU_data_raw[count_imu_temp].time_stamp;
          count_imu_temp++;
      }
      Vector3d g_=(IMU_data_raw[count_imu_temp]._g +IMU_data_raw[count_imu_temp-1]._g )/2;
      Vector3d a_=(IMU_data_raw[count_imu_temp]._a +IMU_data_raw[count_imu_temp-1]._a )/2;
      IMUPreInt.update(g_ - _bg, a_ - _ba, f.time_stamp-IMU_data_raw[count_imu_temp-1].time_stamp);
      // pthread_mutex_unlock(&mutex_imu); 

      // cout<<"imu localization integration data amount:"<<count_imu_temp-count_imu<<endl;
      if(IMUPreInt._delta_time!=delta_time)
      {
        cout<<"imu localization integration error"<<endl;
        exit(1);
      }

      Eigen::Matrix3d r1 =frame_list[last_keyframe_index].pose_sophus[0].matrix().block<3, 3>(0, 0);
      Eigen::Vector3d t1 =frame_list[last_keyframe_index].pose_sophus[0].matrix().block<3, 1>(0, 3);
      Eigen::Matrix3d r2;
      Eigen::Vector3d t2;
      Eigen::Vector3d v2;
          
      pthread_mutex_lock (&mutex_g_R_T);
      //论文中33式 
      r2 =  r1*imu_to_cam_rota* IMUPreInt._delta_R*imu_to_cam_rota.transpose();
      v2 =  r1*imu_to_cam_rota* IMUPreInt._delta_V +frame_list[last_keyframe_index]._V+rota_gravity*initial_gravity* delta_time;
      t2 =  r1*imu_to_cam_rota* IMUPreInt._delta_P+frame_list[last_keyframe_index]._V*delta_time+0.5*rota_gravity*initial_gravity* delta_time*delta_time
        +t1+r1*imu_to_cam_trans-r2*imu_to_cam_trans;
      pthread_mutex_unlock(&mutex_g_R_T);

      f._V=v2;  // 所有的帧都有速度，用于当前的位姿显示

      // 只有视觉跟踪失败的帧才通过imu计算位姿
      if(!registration_success)
      {
        Sophus::SE3d SE3_Rt(r2, t2);
        f.pose_sophus[0] = SE3_Rt;
      }
      // 普通帧不需要零偏
      // f._BiasGyr=_bg;
      // f._BiasAcc=_ba;
    }
    // double time2= (double)cv::getTickCount() - time1;
    // time2=time2*1000/ cv::getTickFrequency();
    // cout<<"integration time(ms): "<<time2<<endl;


    if(!registration_success)
    {
      local_tracking_cnt++;
    }

    if(local_tracking_cnt > 3)
    {
      cout<<endl<<"frame跟踪失败"<<endl;
      update_keyframe_flag = 1;
    }

    PoseSE3d relative_pose_from_key_to_new = relative_transform_from_key_to_new;
    relative_pose_from_key_to_new = relative_pose_from_key_to_new.inverse();
    if (registration_success && !update_keyframe_flag)
    {
        local_tracking_cnt = 0;
        f.tracking_success = 1;
        f.pose_sophus[0] = frame_list[last_keyframe_index].pose_sophus[0] * relative_pose_from_key_to_new;
        f.origin_index = frame_list[last_keyframe_index].origin_index;

        kflist.back().corresponding_frames.push_back(f.frame_index);

        kflist.back().localFrameCorrList.push_back(key_frame_corr);  //当前帧和上一个关键帧的帧相关
        kflist.back().relative_pose_from_key_to_current.push_back(relative_pose_from_key_to_new);
    }

   

    //*********************** update keyframe
    if (update_keyframe_flag)
    { 
      local_tracking_cnt = 0;

      // double time11= (double)cv::getTickCount();

      static int key_index=0;
      key_index++;
      f.keyframe_index=key_index;

      update_keyframe(f.frame_index,
                      key_frame_corr,
                      average_disparity,
                      relative_pose_from_key_to_new,
                      registration_success);
      f.is_keyframe = 1;

      // double time22= (double)cv::getTickCount() - time11;
      // time22=time22*1000/ cv::getTickFrequency();
      // cout<<"update keyframe time: "<<time22<<endl;

      // PLANE::detect_plane(f,camera_para);

      // if (f.is_keyframe  )    //origin = 0 origin_index=0表示没有track失败
      // {
      //   cout<<"current keyframe frame_index tracking_success origin_index"<<endl
      //   <<"                 "<<f.frame_index<<"     "<<f.tracking_success<<"     "<<f.origin_index<<endl;
      // }

      static int count_imu_1=0;       
      while(IMU_data_raw[count_imu_1].time_stamp<f.time_stamp)
      {
          count_imu_1++;
      }
      f.angular_V=IMU_data_raw[count_imu_1]._g;
      cout<<"角速度:  "<<f.frame_index<<"  "<<f.angular_V.norm()<<endl;
      // cout<<"g_ norm: "<<IMU_data_raw[count_imu_1]._g.norm()<<endl;
      // 通过角速度来判断，如果当前角速度太大，那么不更新关键帧


      if(G_parameter.imu_locality&&KeyframeDataList.size()>G_parameter.ini_window_length)
      {
        //-----跟踪失败的时候三种imu定位的方式： (显示的位姿是一直有的，这里的定位是用于优化的关键帧的定位)
        //-----1 只要跟踪失败，就通过imu定位
        //-----2 跟踪失败的时候，额外判断角速度；当前帧对应的角速度小于阈值的时候才进行imu定位
        //-----3 只有跟踪成功且不在主序列的时候通过imu定位，跟踪失败的时候不通过imu定位

        //-------------------------------------------------------1
        // // 只要是关键帧跟踪失败，就通过imu定位
        // if (f.is_keyframe && (f.tracking_success == 0 || f.origin_index>0))
        // {
        //   // 这个参数的作用在于指明通过imu定位的帧不进行建图
        //   f.imu_locality=1;

        //   cout<<"通过imu重定位"<<endl;
        //   f.tracking_success=1;          
        //   f.origin_index=0;
        // }

        //-------------------------------------------------------2
        //找到当前帧对应的imu数据
        // static int count_imu1=0;       
        // while(IMU_data_raw[count_imu1].time_stamp<f.time_stamp)
        // {
        //     count_imu1++;
        // }
        // cout<<"g_ norm: "<<IMU_data_raw[count_imu1]._g.norm()<<endl;
        // // 通过角速度来判断，如果当前角速度太大，那么不更新关键帧
        // if (f.is_keyframe && (f.tracking_success == 0 || f.origin_index>0))
        // {
        //   if(IMU_data_raw[count_imu1]._g.norm()<2)
        //   {
        //     // 这个参数的作用在于指明通过imu定位的帧不进行建图
        //     f.imu_locality=1;

        //     cout<<"通过imu重定位"<<endl;
        //     f.tracking_success=1;          
        //     f.origin_index=0;
        //   }
        // }

        //-------------------------------------------------------3
        // // 如果跟踪成功而且不在主序列,才进行imu的重定位
        if (f.is_keyframe && f.tracking_success == 1 && f.origin_index>0)
        {
          // 这个参数的作用在于指明通过imu定位的帧不进行建图
          f.imu_locality=1;

          cout<<"通过imu重定位"<<endl;
          f.tracking_success=1;          
          f.origin_index=0;
        }

      }

      if (f.is_keyframe && f.tracking_success == 1 && f.origin_index==0)   
      {

        preintegration(&f);
        //*********************** fastBA for globally consistent pose estimation
        double time1= (double)cv::getTickCount();
      
        cout<<"lcoal optimization start"<<endl;
        MultiViewGeometry::optimizeKeyFrameMap(fCorrList_keyframes, frame_list, kflist,0);

        double time2= (double)cv::getTickCount() - time1;
        time2=time2*1000/ cv::getTickFrequency();
        cout<<"local optimization duration time: "<<time2<<endl;
        
        // 线程中的全局优化
        if(g_global_start)
        {
          mutex_global_opti_condi.notify_one();  //唤醒全局优化，如果此时全局优化没有结束，那么这个通知会被丢弃

          double time_start= (double)cv::getTickCount();
          // //  进程中的全局优化
          //   count_global_opti=frame_list.size();
          //   cout<<"进行全局优化"<<endl;
          // MultiViewGeometry::optimizeKeyFrameMapRobust_global( fCorrList_keyframes, frame_list,kflist,0, 0.5);  

          // time2= (double)cv::getTickCount() - time_start;
          // time2=time2*1000/ cv::getTickFrequency();
          
          // static int flag1=0;
          // if(f.frame_index<2450)
          // {
          //   record_vector(time2,2480,"time2",0);
          //   record_vector(f.frame_index,2480,"frame_index",0);
          // }
          // else if(flag1==0)
          // {
          //   flag1=1;
          //   record_vector(time2,2480,"time2",1);
          //   record_vector(f.frame_index,2480,"frame_index",1);
          // }
          

        }
        // else
        // {
        //   record_vector(time2,2480,"time2",0);
        //   record_vector(f.frame_index,2480,"frame_index",0);
        // }
        
        
      }
      else
      {
        cout<<"keyframe跟踪失败,关键帧帧序号为:"<<f.frame_index<<endl<<endl;
      }

      if(G_parameter.show_trajectory)
      {
        pthread_mutex_lock (&mutex_show);
        // pthread_mutex_lock (&mutex_pose);

        // 这里不加的话可以看出来位姿的变化，所有历史的位姿都会显示出来
        g_frame_pose.clear();

        for (int i = 0; i < frame_list.size(); i++)
        {
          Pose_flag pose_temp;
          pose_temp.pose=frame_list[i].pose_sophus[0];
          pose_temp.local_points=frame_list[i].local_points;
          
          pose_temp.tracking_success=frame_list[i].tracking_success;
          pose_temp.origin_index= frame_list[i].origin_index;
          pose_temp.is_keyframe=frame_list[i].is_keyframe;

          g_frame_pose.push_back(pose_temp);
        } 
        // pthread_mutex_unlock(&mutex_pose);
        pthread_mutex_unlock(&mutex_show);
   
      }

    }
}

int  GCSLAM::preintegration(Frame  *newkeyframe)
{
    Frame* lastframe;
    int k=2;
    while(1)
    {
      //取出前一个跟踪成功的关键帧
      lastframe =&globalFrameList[KeyframeDataList[KeyframeDataList.size()-k].keyFrameIndex];
      if(lastframe->tracking_success==1 && lastframe->origin_index==0)   //上一个关键帧满足跟踪成功  回环检测成功,在主序列中
      // if(lastframe->tracking_success==0 )   //上一个关键帧满足跟踪成功才可以
      {
        break;
      }
      else
      {
        k++;
      }
    }

    // cout<<"上一关键帧序号:  "<<lastframe->frame_index<<endl;
    double lasttime = lastframe->time_stamp;

    // 不能用下面的初始化方法，因为update中计算了零偏的一阶导，这个一阶导的计算是在_bg，_ba
    // 的点展开的。后面_dBias_g又通过这个一阶导来计算，那么展开点应该是_BiasGyr
    // Vector3d _bg= lastframe->_BiasGyr+lastframe->_dBias_g;
    // Vector3d _ba= lastframe->_BiasAcc+lastframe->_dBias_a;

    // Vector3d _bg= lastframe->_BiasGyr;
    // Vector3d _ba= lastframe->_BiasAcc;

    Vector3d _bg;
    Vector3d _ba;
    _bg.setZero();
    _ba.setZero();

    // Vector3d _bg1(0.0,0,0.01);
    // Vector3d _ba1(0.0,0.4,0.0);

    // 这里可以采用更合理的方式 
    // Vector3d _bg= lastframe->_BiasGyr+lastframe->_dBias_g;
    // Vector3d _ba= lastframe->_BiasAcc+lastframe->_dBias_a;
    // lastframe->_BiasGyr=lastframe->_BiasGyr+lastframe->_dBias_g;
    // lastframe->_BiasAcc=lastframe->_BiasAcc+lastframe->_dBias_a;
    // lastframe->_dBias_g.setZero();
    // lastframe->_dBias_a.setZero();
    // // 零偏初始化
    // if(KeyframeDataList.size()>3)
    // {
    //   Frame* last_3_frame =&globalFrameList[KeyframeDataList[KeyframeDataList.size()-3].keyFrameIndex];
    //   newkeyframe->_BiasGyr=last_3_frame->_BiasGyr+last_3_frame->_dBias_g;
    //   newkeyframe->_BiasAcc=last_3_frame->_BiasAcc+last_3_frame->_dBias_a;
    // }

    IMUPreintegrator IMUPreInt; 
    // IMUPreintegrator IMUPreInt_temp; 
    IMUPreInt.frame_index_qian=lastframe->frame_index;
    IMUPreInt.frame_index_hou=newkeyframe->frame_index;

    static int count_imu11=0;
    double last_imu_time=lasttime;
    pthread_mutex_lock (&mutex_imu);
    while(IMU_data_raw[count_imu11].time_stamp<lasttime)
    {
        count_imu11++;
      // cout<<count_imu11<<"  ";
    }
    IMUPreInt.imu_index_qian=count_imu11; //刚好大于前一个关键帧的时间
    int count11=0;


    while(IMU_data_raw[count_imu11].time_stamp<newkeyframe->time_stamp)
    {
        double dt = IMU_data_raw[count_imu11].time_stamp - last_imu_time;

        Vector3d g_;
        Vector3d a_;
        if(count_imu11>1)
        {
          g_=(IMU_data_raw[count_imu11]._g +IMU_data_raw[count_imu11-1]._g )/2;
          a_=(IMU_data_raw[count_imu11]._a +IMU_data_raw[count_imu11-1]._a )/2;
        }
        else
        {
          g_=IMU_data_raw[count_imu11]._g;
          a_=IMU_data_raw[count_imu11]._a;
        }

        IMUPreInt.update(g_ - _bg, a_ - _ba, dt);

        // //数值验证bias的一阶导准不准
        // IMUPreInt_temp.update(g_ - _bg1, a_ - _ba1, dt);

        last_imu_time=IMU_data_raw[count_imu11].time_stamp;
        count_imu11++;
        count11++;
    }
 
    IMUPreInt.imu_index_hou=count_imu11;//刚好大于后一个关键帧的时间
    Vector3d g_=(IMU_data_raw[count_imu11]._g +IMU_data_raw[count_imu11-1]._g )/2;
    Vector3d a_=(IMU_data_raw[count_imu11]._a +IMU_data_raw[count_imu11-1]._a )/2;
    IMUPreInt.update(g_ - _bg,a_ - _ba, newkeyframe->time_stamp-IMU_data_raw[count_imu11-1].time_stamp);
    // IMUPreInt_temp.update(g_ - _bg1,a_ - _bg1, newkeyframe->time_stamp-IMU_data_raw[count_imu11-1].time_stamp);
    pthread_mutex_unlock(&mutex_imu);
    
    // //数值验证
    // Matrix3d temp_r=IMUPreInt._delta_R *Sophus::SO3d::exp(IMUPreInt._J_R_Biasg *_bg1).matrix();
    // Vector3d temp_v=IMUPreInt._delta_V+IMUPreInt._J_V_Biasg * _bg1 + IMUPreInt._J_V_Biasa * _ba1;
    // Vector3d temp_p=IMUPreInt._delta_P+IMUPreInt._J_P_Biasg *_bg1 + IMUPreInt._J_P_Biasa * _ba1;

    // cout<<endl<<"numerical bias verify:"<<endl;
    // // cout<<  IMUPreInt._delta_V<<endl<<endl;
    // // cout<<  IMUPreInt._delta_P<<endl<<endl;
    // // cout<<  IMUPreInt_temp._delta_V<<endl<<endl;
    // // cout<<  IMUPreInt_temp._delta_P<<endl<<endl;  
    // // cout<<  temp_v<<endl<<endl;
    // // cout<<  temp_p<<endl<<endl;  
    // temp_r=   temp_r-IMUPreInt_temp._delta_R;
    // temp_v=  temp_v-IMUPreInt_temp._delta_V;
    // temp_p=  temp_p-IMUPreInt_temp._delta_P;
    // cout<<temp_r.norm()<<" "<<temp_v.norm()<<" "<<temp_p.norm()<<endl<<endl;

    lastframe->imu_res=IMUPreInt;
    Eigen::Matrix3d r1=lastframe->pose_sophus[0].matrix().block<3,3>(0,0);
    pthread_mutex_lock(&mutex_g_R_T);
    newkeyframe->_V =  r1 *imu_to_cam_rota* IMUPreInt._delta_V +lastframe->_V+rota_gravity*initial_gravity* (newkeyframe->time_stamp-lastframe->time_stamp);
    pthread_mutex_unlock(&mutex_g_R_T);
    return 0;
}







