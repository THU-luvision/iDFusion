
#include "MultiViewGeometry.h"
#include "ORBSLAM/ORBextractor.h"
#include <xmmintrin.h>


#include <smmintrin.h>
#include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "matrix_youhua.h"
#include "PLANE/peac_plane_detect.h"
#include "IMU/imudata.h"
#include "IMU/so3_cal.h"
// #include <Eigen/CholmodSupport>
#include "../parameter.h"
#include "../sparce_show.h"

using namespace std;
using namespace cv;
using namespace Eigen::internal;  
using namespace Eigen::Architecture;  


#define REPROJECTION_TH 0.01


extern void vector_write_txt(vector<double> var,string name);
extern void record_vector(double temp,int count_record,string name,int out_flag=0);



inline float GetPoseDifference_(const Eigen::Matrix4f &prePose, const Eigen::Matrix4f &curPose)
{
    Eigen::Matrix4f diffTransformation = prePose.inverse() * curPose;
    Eigen::MatrixXf diff(6,1);
    diff.block<3,1>(0,0) = diffTransformation.block<3,3>(0,0).eulerAngles(2,1,0);
    for(int k = 0; k < 3; k++)
    {
        float a = fabs(diff(k,0) - 3.1415926);
        float b = fabs(diff(k,0) + 3.1415926);
        float c = fabs(diff(k,0));
        diff(k,0) = fmin(a,b);
        diff(k,0) = fmin(diff(k,0),c);
    }
    diff.block<3,1>(3,0) = diffTransformation.block<3,1>(0,3);


    float cost = pow(diff(0,0),2)*9 + pow(diff(1,0),2)*9  + pow(diff(2,0),2)*9 + pow(diff(3,0),2)
            + pow(diff(4,0),2)  + pow(diff(5,0),2);
    return cost;
}



class mystreambuf: public std::streambuf
{
};
mystreambuf nostreambuf;
std::ostream nocout(&nostreambuf);

#define COUT_THRESHOLD 3
#define LOG_INFO(x) ((x >= COUT_THRESHOLD)? std::cout : nocout)


namespace MultiViewGeometry
{
    //以 o_x行 o_y列 为原点， 取出长度为 x_length y_length 的矩阵块，赋值给temp 
    void fuzhi_matrix_v2(Eigen::MatrixXd &temp ,Eigen::MatrixXd original,int o_x,int o_y,int flag=1)
    {
        if(flag==0)
        {
            int x_length=temp.rows();
            int y_length=temp.cols();

            for(int i=0;i<x_length;i++)
            {
                for(int j=0;j<y_length;j++)
                {
                    temp(i,j)=original(o_x+i,o_y+j);
                }
            }
        }
        else if(flag==1)
        {
            int x_length=original.rows();
            int y_length=original.cols();

            for(int i=0;i<x_length;i++)
            {
                for(int j=0;j<y_length;j++)
                {
                    temp(o_x+i,o_y+j)=original(i,j);
                }
            }
        }
    }

    void matrix_sparse_to_dense(Eigen::MatrixXd &temp ,Eigen::SparseMatrix<double> s_matrix)
    {
        for (int k = 0; k < s_matrix.outerSize(); ++k) 
        {
          for (Eigen::SparseMatrix<double>::InnerIterator it(s_matrix, k); it; ++it)
          {
            temp(it.row(), it.col())=it.value();  
          }
        }
    }

    void matrix_dense_to_sparse(Eigen::SparseMatrix<double> &s_matrix,Eigen::MatrixXd temp)
    {
      for(int i=0;i<temp.rows();i++)
      {
        for(int j=0;j<temp.cols();j++)
        {
          s_matrix.insert(i, j) = temp(i,j);  
        }        
      }        
    }



    //输出带索引的matrix
    //accuracy 输出精度 length 输出长度  flag_sci 是否采用科学计数法  for_copy 如果等于1，那么输出结果可以直接用于在代码中给矩阵赋值
    void out_matrix(Eigen::MatrixXd temp,int accuracy=1,int length=8,int flag_sci=1,int for_copy=0)
    {
        if(for_copy==0)
        {
            if(flag_sci==1)
            {
                length=accuracy+7;
            }

            for(int j=0;j<temp.cols();j++)
            {
                cout<<setw(length);
                cout<<j<<":";

            }
            cout<<endl;

            for(int i=0;i<temp.rows();i++)
            {
                cout<<setw(3);
                cout<<i<<": ";
                for(int j=0;j<temp.cols();j++)
                {
                    if(flag_sci==1)
                    {
                        cout<<setiosflags(ios::scientific);
                    }
                    cout<<setprecision(accuracy);
                    cout<<setw(length);
                    cout<<temp(i,j)<<" ";
                }
                cout<<endl; 
            }
             cout<<endl<<endl;
        }
        else
        {
            cout<<endl<<endl;
            for(int i=0;i<temp.rows();i++)
            {
                for(int j=0;j<temp.cols();j++)
                {
                    cout<<temp(i,j)<<",";
                }
                cout<<endl; 
            }
            cout<<endl<<endl;
        }
        // setw(设置最小输出宽度)
        // setprecision(length) 设置有效数字位数
        // ios::scientific指数表示
    }

  GlobalParameters g_para;

  void optimize_3d_to_3d_huber_filter(Frame &frame_ref,
                                            Frame& frame_new,
                                            std::vector< cv::DMatch > &ransac_inlier_matches,
                                            PoseSE3d &relative_pose_from_ref_to_new,
                                            std::vector<float> &weight_per_point,
                                            float outlier_threshold = 0.015,
                                            int max_iter_num = 4,
                                            float huber_threshold = 0.008)
    {

    weight_per_point.clear();
    clock_t  start,end;
        int valid_3d_cnt = ransac_inlier_matches.size();

    Eigen::MatrixXf Je(6, 1), JTJ(6, 6);
    Eigen::Vector3f err;
    Eigen::VectorXf delta(6);
    Point3dList p_ref, p_new;
        p_ref.clear();
        p_new.clear();
        p_ref.reserve(valid_3d_cnt);
        p_new.reserve(valid_3d_cnt);

    std::vector<__m128> ref_points_vectorized(valid_3d_cnt);
    std::vector<__m128> new_points_vectorized(valid_3d_cnt);
    std::vector<float> depth(valid_3d_cnt);
        for (size_t i = 0; i < valid_3d_cnt; i++)
        {

            Eigen::Vector3d pt_ref(frame_ref.local_points[ransac_inlier_matches[i].queryIdx]);
            Eigen::Vector3d pt_new(frame_new.local_points[ransac_inlier_matches[i].trainIdx]);
            p_ref.push_back(pt_ref);
            p_new.push_back(pt_new);

          ref_points_vectorized[i] = _mm_setr_ps(pt_ref.x(),pt_ref.y(),pt_ref.z(),1);
          new_points_vectorized[i] = _mm_setr_ps(pt_new.x(),pt_new.y(),pt_new.z(),0);
        depth[i] = pt_ref.z();
        }
//		float init_error = reprojection_error_3Dto3D(p_ref, p_new, relative_pose_from_ref_to_new, 1);
        for (int iter_cnt = 0; iter_cnt < max_iter_num; iter_cnt++)
        {

            Je.setZero();
            JTJ.setZero();


            Eigen::MatrixXd R_ref = relative_pose_from_ref_to_new.rotationMatrix();
            Eigen::Vector3d t_ref = relative_pose_from_ref_to_new.translation();
            __m128 T_ref[3];

            T_ref[0] = _mm_setr_ps(R_ref(0,0),R_ref(0,1),R_ref(0,2),t_ref(0));
            T_ref[1] = _mm_setr_ps(R_ref(1,0),R_ref(1,1),R_ref(1,2),t_ref(1));
            T_ref[2] = _mm_setr_ps(R_ref(2,0),R_ref(2,1),R_ref(2,2),t_ref(2));

            Eigen::MatrixXf J_i_sse(3,6);

            J_i_sse.setZero();
            J_i_sse(0, 0) = 1;
            J_i_sse(1, 1) = 1;
            J_i_sse(2, 2) = 1;
            __m128 res, reprojection_error_vec;
            res[3] = 0;
            reprojection_error_vec[3] = 0;
            for (int i = 0; i < valid_3d_cnt; i++)
            {

                res = _mm_add_ps(_mm_dp_ps(T_ref[1], ref_points_vectorized[i], 0xf2),
                    _mm_dp_ps(T_ref[0], ref_points_vectorized[i], 0xf1));
                res = _mm_add_ps(res, _mm_dp_ps(T_ref[2], ref_points_vectorized[i], 0xf4));
                reprojection_error_vec = _mm_sub_ps(res, new_points_vectorized[i]);

                float error_sse = sqrt(_mm_cvtss_f32(_mm_dp_ps(res, res, 0x71))) / depth[i];

                float weight_huber = 1;
                if (error_sse > huber_threshold)
                {
                  weight_huber = huber_threshold / error_sse;
                }
                float weight = weight_huber / (depth[i]);


                const __m128 scalar = _mm_set1_ps(weight);
                reprojection_error_vec = _mm_mul_ps(reprojection_error_vec, scalar);
                __m128 cross_value = CrossProduct(res,reprojection_error_vec);
                J_i_sse(0, 4) = res[2];
                J_i_sse(0, 5) = -res[1];
                J_i_sse(1, 3) = -res[2];
                J_i_sse(1, 5) = res[0];
                J_i_sse(2, 3) = res[1];
                J_i_sse(2, 4) = -res[0];
                Je(0,0) += reprojection_error_vec[0];
                Je(1,0) += reprojection_error_vec[1];
                Je(2,0) += reprojection_error_vec[2];
                Je(3,0) += cross_value[0];
                Je(4,0) += cross_value[1];
                Je(5,0) += cross_value[2];

                JTJ += J_i_sse.transpose() * J_i_sse * weight;

            }
            delta = JTJ.inverse() * Je;
            Eigen::VectorXd delta_double = delta.cast<double>();


            relative_pose_from_ref_to_new = Sophus::SE3d::exp(delta_double).inverse() * relative_pose_from_ref_to_new;
        }
        std::vector< cv::DMatch > matches_refined;
        matches_refined.reserve(valid_3d_cnt);
        for (int i = 0; i < p_ref.size(); i++)
        {
            Eigen::Vector3d reprojection_error = applyPose(relative_pose_from_ref_to_new, p_ref[i]) - (p_new[i]);



            if (reprojection_error.norm() / p_ref[i].z() < outlier_threshold)
            {

                float weight_huber = 1;
                float error = reprojection_error.norm() / p_ref[i].z();
                if (error > huber_threshold)
                {
                  weight_huber = huber_threshold / error;
                }
                matches_refined.push_back(ransac_inlier_matches[i]);
                weight_per_point.push_back(weight_huber);
            }
        }
        ransac_inlier_matches = matches_refined;
    }


  void estimateRigid3DTransformation(Frame &frame_ref,
      Frame &frame_new,
      std::vector< DMatch > &init_matches,
      Eigen::Matrix3d &R, Eigen::Vector3d &t,
      float reprojection_error_threshold,
      int max_iter_num)
    {

      std::vector<DMatch> matches_before_filtering = init_matches;
      // random 100 times test
      int N_predict_inliers = init_matches.size();
      int N_total = matches_before_filtering.size();
      Eigen::Vector3d ref_points[4], mean_ref_points;
      Eigen::Vector3d new_points[4], mean_new_points;
      int candidate_seed;
      Eigen::Matrix3d H, UT, V;
      Eigen::Matrix3d temp_R;
      Eigen::Vector3d temp_t;
      int count;
      int best_results = 0;
      float reprojection_error_threshold_square = reprojection_error_threshold *reprojection_error_threshold;
          //	[R t] * [x,1]  3x4 * 4xN
      std::vector<__m128> ref_points_vectorized(N_total);
      std::vector<__m128> new_points_vectorized(N_total);
      std::vector<float> depth_square(N_total);
      for (size_t i = 0; i < N_total; i++)
      {
        Eigen::Vector3d ref_point = frame_ref.local_points[matches_before_filtering[i].queryIdx];
        Eigen::Vector3d new_point = frame_new.local_points[matches_before_filtering[i].trainIdx];
        ref_points_vectorized[i] = _mm_setr_ps((float)ref_point[0],(float)ref_point[1],(float)ref_point[2],1);
        new_points_vectorized[i] = _mm_setr_ps((float)new_point[0],(float)new_point[1],(float)new_point[2],1);
        depth_square[i] = ref_point[2] * ref_point[2];
      }
      Eigen::MatrixXf rigid_transform(4, 3);
      for (int cnt = 0; cnt < max_iter_num; cnt++)
      {
        H.setZero();
        mean_ref_points.setZero();
        mean_new_points.setZero();
        for (int i = 0; i < 4; i++)
        {
          candidate_seed = rand() % N_predict_inliers;
          ref_points[i] = frame_ref.local_points[init_matches[candidate_seed].queryIdx];
          new_points[i] = frame_new.local_points[init_matches[candidate_seed].trainIdx];
          mean_ref_points += ref_points[i];
          mean_new_points += new_points[i];
        }
        mean_ref_points /= 4;
        mean_new_points /= 4;
        for (int i = 0; i < 4; i++)
        {
          H += (ref_points[i] - mean_ref_points) * (new_points[i] - mean_new_points).transpose();
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        UT = svd.matrixU().transpose();
        V = svd.matrixV();
        temp_R = V * UT;
        if (temp_R.determinant() < 0)
        {
          V(0, 2) = -V(0, 2);
          V(1, 2) = -V(1, 2);
          V(2, 2) = -V(2, 2);
          temp_R = V * UT;
        }
        temp_t = mean_new_points - temp_R * mean_ref_points;
        Eigen::MatrixXd transformation(3,4);

        unsigned char mask0 = 0xf1;
        unsigned char mask1 = 0xf2;
        unsigned char mask2 = 0xf4;
        unsigned char mask3 = 0xf8;
        __m128 transform_vectorized[4];
        for (int i = 0; i < 3; i++)
        {
          transform_vectorized[i] = _mm_setr_ps( (float)temp_R(i, 0), (float)temp_R(i, 1), (float)temp_R(i, 2), (float)temp_t(i));
        }
        count = 0;
        for (size_t i = 0; i < N_total; i++)
        {
          __m128 res;
          res = _mm_add_ps(_mm_dp_ps(transform_vectorized[1], ref_points_vectorized[i], 0xf2),
              _mm_dp_ps(transform_vectorized[0], ref_points_vectorized[i], 0xf1));
          res = _mm_add_ps(res, _mm_dp_ps(transform_vectorized[2], ref_points_vectorized[i], 0xf4));
          res = _mm_sub_ps(res, new_points_vectorized[i]);
          count += (_mm_cvtss_f32(_mm_dp_ps(res, res, 0x71)) / depth_square[i] < reprojection_error_threshold_square);
        }
        if (count > best_results)
        {
          best_results = count;
          R = temp_R;
          t = temp_t;
        }

      }
    }



  float ransac3D3D(Frame &frame_ref, Frame &frame_new, std::vector< DMatch > &init_matches, std::vector< DMatch > &matches_before_filtering,
      float reprojectionThreshold, int max_iter_num, FrameCorrespondence &fCorr,
      PoseSE3d &relative_pose_from_ref_to_new, CameraPara para)
    {

      if (init_matches.size() < minimum_3d_correspondence)
      {
        return 1e7;
      }
      clock_t start, end;
      clock_t start_total, end_total;

      start_total = clock();
      start = clock();
      Eigen::Matrix3d direct_R;
      Eigen::Vector3d direct_t;
      estimateRigid3DTransformation(frame_ref, frame_new, init_matches, direct_R, direct_t, reprojectionThreshold * 2, max_iter_num);
      end = clock();
      float time_ransac = end - start;
      start = clock();
      std::vector< DMatch > ransac_inlier_matches;
      std::vector< DMatch > ransac_2d_inlier_matches;
      ransac_inlier_matches.clear();
      ransac_inlier_matches.reserve(matches_before_filtering.size());
      int previous_query_index = -1;
      for (size_t i = 0; i < matches_before_filtering.size(); i++)
      {
        Eigen::Vector3d ref_point = frame_ref.local_points[matches_before_filtering[i].queryIdx];
        Eigen::Vector3d new_point = frame_new.local_points[matches_before_filtering[i].trainIdx];

        Eigen::Vector3d estimate_new_local_points = direct_R * ref_point + direct_t;
        Eigen::Vector3d normalized_predict_points = estimate_new_local_points / estimate_new_local_points(2);
        cv::KeyPoint predict_new_2D = frame_ref.keypoints[matches_before_filtering[i].queryIdx];
        cv::KeyPoint new_2D = frame_new.keypoints[matches_before_filtering[i].trainIdx];
        predict_new_2D.pt.x = normalized_predict_points(0) * para.c_fx + para.c_cx;
        predict_new_2D.pt.y = normalized_predict_points(1) * para.c_fy + para.c_cy;
        float delta_x = predict_new_2D.pt.x - new_2D.pt.x;
        float delta_y = predict_new_2D.pt.y - new_2D.pt.y;
        float average_reprojection_error_2d = sqrt(delta_x * delta_x + delta_y * delta_y);
        float reprojection_error = (new_point - direct_R * ref_point - direct_t).norm() / ref_point(2);
        if (average_reprojection_error_2d < 2)
        {
          ransac_2d_inlier_matches.push_back(matches_before_filtering[i]);
        }
        if (reprojection_error < reprojectionThreshold * 2)
        {
          if (g_para.runTestFlag)
          {
            cout << i << " " << average_reprojection_error_2d << " " << reprojection_error << endl;
          }
          // only add the first match
          if (previous_query_index != matches_before_filtering[i].queryIdx)
          {
            previous_query_index = matches_before_filtering[i].queryIdx;
            ransac_inlier_matches.push_back(matches_before_filtering[i]);
          }
        }
      }

      if (ransac_inlier_matches.size() < minimum_3d_correspondence)
      {
        return 1e7;
      }
      end = clock();
      float time_select_init_inliers = end - start;
      if (g_para.runTestFlag)
      {
        Mat img_matches;
        std::vector< DMatch > no_matches;
        no_matches.clear();
#if 0
        cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
          ransac_inlier_matches, img_matches);
        char frameMatchName[256];
        memset(frameMatchName, '\0', 256);
        sprintf(frameMatchName, "frame matching after ransac");
        cv::imshow(frameMatchName, img_matches);
        cvWaitKey();
#endif
      }

      relative_pose_from_ref_to_new = Sophus::SE3d(direct_R,direct_t);

      start = clock();
      //refine match
      std::vector<float> weight_per_feature;
      optimize_3d_to_3d_huber_filter(frame_ref, frame_new,
                                     ransac_inlier_matches,
                                     relative_pose_from_ref_to_new,
                                     weight_per_feature,
                                     reprojectionThreshold, 6,0.005);
      end = clock();
      float time_nonlinear_opt = end - start;


#if 1
      start = clock();
      init_matches.clear();
      weight_per_feature.clear();
      weight_per_feature.reserve(matches_before_filtering.size());
      init_matches.reserve(matches_before_filtering.size());
      Eigen::MatrixXd H = relative_pose_from_ref_to_new.matrix3x4();
      Eigen::MatrixXd invH = relative_pose_from_ref_to_new.matrix().inverse().block<3, 4>(0, 0);
      float average_reprojection_error_2D = 0;
      for (size_t i = 0; i < matches_before_filtering.size(); i++)
      {
        Eigen::Vector3d ref_point = frame_ref.local_points[matches_before_filtering[i].queryIdx];
        Eigen::Vector3d new_point = frame_new.local_points[matches_before_filtering[i].trainIdx];
        cv::KeyPoint predict_new_2D = frame_ref.keypoints[matches_before_filtering[i].queryIdx];
        cv::KeyPoint predict_ref_2D = frame_new.keypoints[matches_before_filtering[i].trainIdx];
        cv::KeyPoint ref_2D = frame_ref.keypoints[matches_before_filtering[i].queryIdx];
        cv::KeyPoint new_2D = frame_new.keypoints[matches_before_filtering[i].trainIdx];
        Eigen::Vector4d homo_ref_points,homo_new_points;

        homo_ref_points << ref_point(0), ref_point(1), ref_point(2), 1;
        homo_new_points << new_point(0), new_point(1), new_point(2), 1;
        Eigen::Vector3d estimate_new_local_points = H * homo_ref_points;
        Eigen::Vector3d estimate_ref_local_points = invH * homo_new_points;


        predict_new_2D.pt.x = estimate_new_local_points(0) / estimate_new_local_points(2) * para.c_fx + para.c_cx;
        predict_new_2D.pt.y = estimate_new_local_points(1) / estimate_new_local_points(2)* para.c_fy + para.c_cy;
        predict_ref_2D.pt.x = estimate_ref_local_points(0) / estimate_ref_local_points(2) * para.c_fx + para.c_cx;
        predict_ref_2D.pt.y = estimate_ref_local_points(1) / estimate_ref_local_points(2) * para.c_fy + para.c_cy;
        float delta_x = predict_new_2D.pt.x - new_2D.pt.x;
        float delta_y = predict_new_2D.pt.y - new_2D.pt.y;
        float average_reprojection_error_2d_ref = sqrt(delta_x * delta_x + delta_y * delta_y);
        delta_x = predict_ref_2D.pt.x - ref_2D.pt.x;
        delta_y = predict_ref_2D.pt.y - ref_2D.pt.y;
        float average_reprojection_error_2d_new = sqrt(delta_x * delta_x + delta_y * delta_y);
        float reprojection_error = (new_point - estimate_new_local_points).norm() / ref_point(2);
        double reprojection_error_2d = average_reprojection_error_2d_ref;// max(average_reprojection_error_2d_ref, average_reprojection_error_2d_new);

        if (reprojection_error < reprojectionThreshold  && reprojection_error_2d < g_para.reprojection_error_2d_threshold)
        {
          average_reprojection_error_2D += reprojection_error_2d;
          if (g_para.runTestFlag)
          {
            cout << i << " " << average_reprojection_error_2d_ref << " " << average_reprojection_error_2d_new << " " << reprojection_error << endl;
          }


          float huber_threshold = 0.008;
          float weight_huber = 1;
          if (reprojection_error > huber_threshold)
          {
            weight_huber = huber_threshold / reprojection_error;
          }

          init_matches.push_back(matches_before_filtering[i]);
          weight_per_feature.push_back(weight_huber);
        }
      }
      if (g_para.runTestFlag)
      {
        cout << "2D/3D inlier num: " << ransac_2d_inlier_matches.size() << " " << init_matches.size() << endl;
        if (ransac_2d_inlier_matches.size() > 0)
        {
#if 0
          Mat img_matches;
          cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
            ransac_2d_inlier_matches, img_matches);
          cv::imshow("2D match inliers", img_matches);
#endif
        }
      }
      ransac_inlier_matches = init_matches;
#endif
      end = clock();
      float time_select_final_inlier = end - start;

#if 0
      start = clock();
      optimize_3d_to_3d_huber_filter(frame_ref, frame_new,
                                     ransac_inlier_matches,
                                     relative_pose_from_ref_to_new,
                                     weight_lp,
                                     reprojectionThreshold, 6,0.005);
      end = clock();
#endif

      float time_last_non_linear = end - start;



//      float reprojection_error = reprojection_error_3Dto3D(frame_ref, frame_new, ransac_inlier_matches, (relative_pose_from_ref_to_new), 0);
      init_matches = ransac_inlier_matches;
      // make sure there is no outliers
  //		RefineByRotation(frame_ref, frame_new, init_matches);
  //		outlierFiltering(frame_ref, frame_new, init_matches, 5,0.01);
  //		outlierFiltering(frame_ref, frame_new, init_matches, 5,0.01);
  //    cout << "reprojection error after lp optimzation: " << reprojection_error << endl;


      end_total = clock();
      float time_total = end_total - start_total ;
#if 0
      LOG_INFO(1) << "ransac 3D to 3D total time: " << time_total << "    " << "time ransac/init/nonlinear/final/nonlinear: "
               << time_ransac << "/"
               << time_select_init_inliers << "/"
               << time_nonlinear_opt << "/"
               << time_select_final_inlier << "/"
               << time_last_non_linear <<endl;
#endif
      if (init_matches.size() > minimum_3d_correspondence)
      {
        fCorr.matches = init_matches;
        fCorr.weight_per_feature = weight_per_feature;
        fCorr.preIntegrate();
        float reprojection_error = reprojection_error_3Dto3D(fCorr,(relative_pose_from_ref_to_new));
        return reprojection_error;

      }
      return 1e6;
    }



  void outlierFiltering(Frame &frame_ref, Frame &frame_new, std::vector< cv::DMatch > &init_matches)
    {
        int candidate_num = 8;
        float distance_threshold = 0.015;
        int N = init_matches.size();
        std::vector< cv::DMatch > filtered_matches;
        filtered_matches.reserve(N);
        for (size_t i = 0; i < N; i++)
        {
            Eigen::Vector3d ref_point = frame_ref.local_points[init_matches[i].queryIdx];
            Eigen::Vector3d new_point = frame_new.local_points[init_matches[i].trainIdx];

            int distance_preserve_flag = 0;
            for (size_t j = 0; j < candidate_num; j++)
            {
                int rand_choice = rand() % N;
                Eigen::Vector3d ref_point_p = frame_ref.local_points[init_matches[rand_choice].queryIdx];
                Eigen::Vector3d new_point_p = frame_new.local_points[init_matches[rand_choice].trainIdx];
                double d1 = (ref_point_p - ref_point).norm();
                double d2 = (new_point_p - new_point).norm();
                if (fabs(d1 - d2) / ref_point(2) < distance_threshold)
                {
                    distance_preserve_flag = 1;
                    break;
                }
            }
            if (distance_preserve_flag)
            {
                filtered_matches.push_back(init_matches[i]);
            }
        }
        init_matches = filtered_matches;
    }

  bool FrameMatchingTwoViewRGB(FrameCorrespondence &fCorr,
                               MultiViewGeometry::CameraPara camera_para,
                               MILD::SparseMatcher frame_new_matcher,
                               PoseSE3d &relative_pose_from_ref_to_new,
                               float &average_disparity,
                               float &scale_change_ratio,
                               bool &update_keyframe_from_dense_matching,
                               bool use_initial_guess,
                               double threshold)
  {

    float time_feature_matching,time_ransac,time_filter,time_refine,time_rotation_filter ;
    update_keyframe_from_dense_matching = 0;

    float reprojection_error_feature_based;
    float reprojection_error_dense_based;

    PoseSE3d init_guess_relative_pose_from_ref_to_new = relative_pose_from_ref_to_new;
    Frame &frame_ref = fCorr.frame_ref;
    Frame &frame_new = fCorr.frame_new;
    bool matching_success = 0;
    bool dense_success = 0, sparse_success = 0;

    LOG_INFO(1) << "************frame registration: " << frame_ref.frame_index << " vs "<< frame_new.frame_index << "************" << endl;

    vector<vector<DMatch>> matches;
    std::vector< DMatch > init_matches;
    init_matches.clear();
    matches.clear();
    clock_t start, end;
    clock_t start_total, end_total;
    double duration;
    start_total = clock();

    // feature matching based on hamming distance    orb特征提取提供了特征点，特征点匹配是在mild中

    frame_new_matcher.search_8(frame_ref.descriptor, init_matches, g_para.hamming_distance_threshold);
    int matched_feature_pairs = init_matches.size();
    int rotation_inliers = 0;

    if (0)
    {
      Mat img_matches;
      std::vector< DMatch > no_matches;
      no_matches.clear();
      cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
        init_matches, img_matches);
      char frameMatchName[256];
      memset(frameMatchName, '\0', 256);
      //	sprintf(frameMatchName, "match_%04d_VS_%04d.jpg", frame_ref.frame_index, frame_new.frame_index);
      sprintf(frameMatchName, "frame matching");
      cv::imwrite("frame_match.jpg", img_matches);
      cvWaitKey(1);
    }
#if 1


    RefineByRotation(frame_ref, frame_new, init_matches);
    rotation_inliers = init_matches.size();
#endif

    // use ransac to remove outliers
    int candidate_num = 8;
    float min_distance_threshold = 0.015;
    int inliers_num_first, inliers_num_second;			// make sure 90% are inliers
    inliers_num_first = matched_feature_pairs;
    std::vector< DMatch > matches_before_filtering = init_matches;
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);



    int ransac_input_num = init_matches.size();

    double reprojection_error;


    reprojection_error = ransac3D3D(frame_ref,
                                    frame_new,
                                    init_matches,
                                    matches_before_filtering,
                                    g_para.reprojection_error_3d_threshold,
                                    g_para.ransac_maximum_iterations,
                                    fCorr,
                                    relative_pose_from_ref_to_new,
                                    camera_para);

    start = clock();

    /********************** fine search **********************/
    {
      // refine binary feature search results
      Eigen::MatrixXd H = relative_pose_from_ref_to_new.matrix3x4();
      std::vector< DMatch > predict_matches;
      std::vector<cv::KeyPoint> predict_ref_points(frame_ref.local_points.size());
      for (int i = 0; i < frame_ref.local_points.size(); i++)
      {
        Eigen::Vector4d homo_points;
        homo_points << frame_ref.local_points[i](0), frame_ref.local_points[i](1), frame_ref.local_points[i](2), 1;
        Eigen::Vector3d predict_points = H*homo_points;
        predict_points = predict_points / predict_points(2);
        cv::KeyPoint predict_ref = frame_ref.keypoints[i];
        predict_ref.pt.x = predict_points(0) * camera_para.c_fx + camera_para.c_cx;
        predict_ref.pt.y = predict_points(1) * camera_para.c_fy + camera_para.c_cy;
        predict_ref_points[i] = predict_ref;
        cv::DMatch m;
        m.queryIdx = i;
        m.trainIdx = i;
        if (i % 20 == 0)
        {
          predict_matches.push_back(m);

        }
      }

      if (g_para.runTestFlag)
      {
#if 0
        Mat img_matches;
        cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, predict_ref_points,
          predict_matches, img_matches);
        char frameMatchName[256];
        memset(frameMatchName, '\0', 256);
        //	sprintf(frameMatchName, "match_%04d_VS_%04d.jpg", frame_ref.frame_index, frame_new.frame_index);
        sprintf(frameMatchName, "frame matching predict");
        cv::imshow(frameMatchName, img_matches);
        cvWaitKey(1);
#endif

      }
      init_matches.clear();
      frame_new_matcher.search_8_with_range(frame_ref.descriptor, init_matches, frame_new.keypoints, predict_ref_points, 30,
                                              g_para.hamming_distance_threshold * 1.5);
      //RefineByRotation(frame_ref, frame_new, init_matches);
      std::vector< DMatch > complete_matches = init_matches;
      outlierFiltering(frame_ref, frame_new, init_matches);
      outlierFiltering(frame_ref, frame_new, init_matches);
      outlierFiltering(frame_ref, frame_new, init_matches);
      outlierFiltering(frame_ref, frame_new, init_matches);
      reprojection_error = ransac3D3D(frame_ref, frame_new, init_matches, complete_matches, g_para.reprojection_error_3d_threshold,
        g_para.ransac_maximum_iterations, fCorr, relative_pose_from_ref_to_new,camera_para);
    //  cout << "reprojection error: " << reprojection_error << endl;
    }

    end = clock();
    time_refine = end - start ;

    start = clock();


    std::vector<float> feature_weight_lp = fCorr.weight_per_feature;

    reprojection_error_feature_based = reprojection_error;
    float scale_increase = 0, scale_decrease = 0;
    for (int i = 0; i < init_matches.size(); i++)
    {
      cv::KeyPoint predict_new_2D = frame_ref.keypoints[init_matches[i].queryIdx];
      cv::KeyPoint predict_ref_2D = frame_new.keypoints[init_matches[i].trainIdx];
      if (predict_new_2D.octave >  predict_ref_2D.octave)
      {
        scale_increase++;
      }
      if (predict_new_2D.octave < predict_ref_2D.octave)
      {
        scale_decrease++;
      }
    }
    scale_change_ratio = fmax(scale_decrease, scale_increase) / (init_matches.size()+1);


    if(threshold != -1)
    {
      if(reprojection_error < threshold)
      {
        fCorr.preIntegrate();
        average_disparity = fCorr.calculate_average_disparity(camera_para);
  //        fCorr.clear_memory();
        sparse_success = 1;
        matching_success = 1;
      }
    }
    else
    {
      if(reprojection_error < REPROJECTION_TH)
      {
        sparse_success = 1;
        fCorr.preIntegrate();
        average_disparity = fCorr.calculate_average_disparity(camera_para);
  //        fCorr.clear_memory();
        matching_success = 1;
      }
    }


    end = clock();
    float time_finishing = end - start;

    if(0)
    {
        LOG_INFO(1) << "average reprojection error : " << reprojection_error_3Dto3D(fCorr,(relative_pose_from_ref_to_new)) << "    "
                 << "average disparity: "<< average_disparity << "    "
                 << "scale change ratio: " << scale_change_ratio << endl;
        LOG_INFO(1) << "sparse match: " << sparse_success << " " << init_matches.size() << " " << reprojection_error  << "    " << endl;
        LOG_INFO(1) << "run time: featureMatching/rotationFilter/filter/ransac/refine/finishing: "
                 << time_feature_matching << "/"
                 << time_rotation_filter << "/"
                 << time_filter << "/"
                 << time_ransac << "/"
                 << time_refine << "/"
                 << time_finishing << endl;
    }


    return matching_success;
  }

  void ComputeJacobianInfo(FrameCorrespondence &fC,
    Eigen::MatrixXd &Pre_JiTr,
    Eigen::MatrixXd &Pre_JjTr,
    Eigen::MatrixXd &Pre_JiTJi,
    Eigen::MatrixXd &Pre_JiTJj,
    Eigen::MatrixXd &Pre_JjTJi,
    Eigen::MatrixXd &Pre_JjTJj)
  {
    int valid_3d_cnt = fC.sparse_feature_cnt +  fC.dense_feature_cnt;
    // construct the four matrix based on pre-integrated points
    Pre_JiTr.setZero();
    Pre_JjTr.setZero();
    Pre_JiTJi.setZero();
    Pre_JiTJj.setZero();
    Pre_JjTJi.setZero();
    Pre_JjTJj.setZero();
    if (valid_3d_cnt < minimum_3d_correspondence)
    {
      return;
    }
    //prepare data
    pthread_mutex_lock (&mutex_pose);
    Eigen::Matrix3d R_ref = fC.frame_ref.pose_sophus[0].rotationMatrix();
    Eigen::Vector3d t_ref = fC.frame_ref.pose_sophus[0].translation();
    Eigen::Matrix3d R_new = fC.frame_new.pose_sophus[0].rotationMatrix();
    Eigen::Vector3d t_new = fC.frame_new.pose_sophus[0].translation();
    pthread_mutex_unlock(&mutex_pose);
    
    Eigen::Matrix3d Eye3x3;

    Eye3x3.setIdentity();
    Eigen::Matrix3d riWrj, riWri, rjWrj;
    riWrj = R_ref * fC.sum_p_ref_new * R_new.transpose();
    riWri = R_ref * fC.sum_p_ref_ref * R_ref.transpose();
    rjWrj = R_new * fC.sum_p_new_new * R_new.transpose();

    Eigen::Vector3d R_ref_sum_p_ref = R_ref * fC.sum_p_ref;
    Eigen::Vector3d R_new_sum_p_new = R_new * fC.sum_p_new;
    Eigen::Vector3d residual = R_ref_sum_p_ref + fC.sum_weight * (t_ref - t_new) - R_new_sum_p_new;
    //calculating JTr, see ProblemFormulation.pdf
    Pre_JiTr.block<3, 1>(0, 0) = residual;
    Pre_JiTr.block<3, 1>(3, 0) = Eigen::Vector3d(riWrj(2, 1) - riWrj(1, 2), -riWrj(2, 0) + riWrj(0, 2), riWrj(1, 0) - riWrj(0, 1))
      + R_ref_sum_p_ref.cross(t_ref - t_new) + t_ref.cross(residual);

    Pre_JjTr.block<3, 1>(0, 0) = residual;
    Pre_JjTr.block<3, 1>(3, 0) = Eigen::Vector3d(riWrj(2, 1) - riWrj(1, 2), -riWrj(2, 0) + riWrj(0, 2), riWrj(1, 0) - riWrj(0, 1))
      + R_new_sum_p_new.cross(t_ref - t_new) + t_new.cross(residual);
    Pre_JjTr = -Pre_JjTr;

    //calculating JTJ
    Pre_JiTJi.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
    Pre_JiTJi.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_ref_sum_p_ref + fC.sum_weight * t_ref);
    Pre_JiTJi.block<3, 3>(3, 0) = -Pre_JiTJi.block<3, 3>(0, 3);
    Pre_JiTJi(3, 3) = riWri(2, 2) + riWri(1, 1);	Pre_JiTJi(3, 4) = -riWri(1, 0);					Pre_JiTJi(3, 5) = -riWri(2, 0);
    Pre_JiTJi(4, 3) = -riWri(0, 1);					Pre_JiTJi(4, 4) = riWri(0, 0) + riWri(2, 2);	Pre_JiTJi(4, 5) = -riWri(2, 1);
    Pre_JiTJi(5, 3) = -riWri(0, 2);					Pre_JiTJi(5, 4) = -riWri(1, 2);					Pre_JiTJi(5, 5) = riWri(0, 0) + riWri(1, 1);
    Pre_JiTJi.block<3, 3>(3, 3) += -skewMatrixProduct(t_ref, R_ref_sum_p_ref) - skewMatrixProduct(R_ref_sum_p_ref, t_ref) - fC.sum_weight * 1 * skewMatrixProduct(t_ref, t_ref);

    Pre_JjTJj.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
    Pre_JjTJj.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_new_sum_p_new + fC.sum_weight * t_new);
    Pre_JjTJj.block<3, 3>(3, 0) = -Pre_JjTJj.block<3, 3>(0, 3);
    Pre_JjTJj(3, 3) = rjWrj(2, 2) + rjWrj(1, 1);	Pre_JjTJj(3, 4) = -rjWrj(1, 0);					Pre_JjTJj(3, 5) = -rjWrj(2, 0);
    Pre_JjTJj(4, 3) = -rjWrj(0, 1);					Pre_JjTJj(4, 4) = rjWrj(0, 0) + rjWrj(2, 2);	Pre_JjTJj(4, 5) = -rjWrj(2, 1);
    Pre_JjTJj(5, 3) = -rjWrj(0, 2);					Pre_JjTJj(5, 4) = -rjWrj(1, 2);					Pre_JjTJj(5, 5) = rjWrj(0, 0) + rjWrj(1, 1);
    Pre_JjTJj.block<3, 3>(3, 3) += -skewMatrixProduct(t_new, R_new_sum_p_new) - skewMatrixProduct(R_new_sum_p_new, t_new) - fC.sum_weight * 1 * skewMatrixProduct(t_new, t_new);


    Pre_JiTJj.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
    Pre_JiTJj.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_new_sum_p_new + fC.sum_weight * t_new);
    Pre_JiTJj.block<3, 3>(3, 0) = -getSkewSymmetricMatrix(R_ref_sum_p_ref + fC.sum_weight * t_ref).transpose();
    Pre_JiTJj(3, 3) = riWrj(2, 2) + riWrj(1, 1);	Pre_JiTJj(3, 4) = -riWrj(1, 0);	Pre_JiTJj(3, 5) = -riWrj(2, 0);
    Pre_JiTJj(4, 3) = -riWrj(0, 1);	Pre_JiTJj(4, 4) = riWrj(0, 0) + riWrj(2, 2);	Pre_JiTJj(4, 5) = -riWrj(2, 1);
    Pre_JiTJj(5, 3) = -riWrj(0, 2);	Pre_JiTJj(5, 4) = -riWrj(1, 2);		Pre_JiTJj(5, 5) = riWrj(0, 0) + riWrj(1, 1);
    Pre_JiTJj.block<3, 3>(3, 3) += -skewMatrixProduct(t_ref, R_new_sum_p_new) - skewMatrixProduct(R_ref_sum_p_ref, t_new) - fC.sum_weight * 1 * skewMatrixProduct(t_ref, t_new);
    Pre_JiTJj = -Pre_JiTJj;
    Pre_JjTJi = Pre_JiTJj.transpose();

  #if 0
    cout << "precomputing jacobian matrics: " << fC.frame_new.frame_index << " " << fC.frame_ref.frame_index << endl;
    cout << "JiTr:" << Pre_JiTr.transpose() << endl;
    cout << "JjTr:" << Pre_JjTr.transpose() << endl;
    cout << "JiTJi: " << endl << Pre_JiTJi << endl;
    cout << "JiTJj: " << endl << Pre_JiTJj << endl;
    cout << "JjTJi: " << endl << Pre_JjTJi << endl;
    cout << "JjTJj: " << endl << Pre_JjTJj << endl;
  #endif
  }


void ComputeJacobianInfo_simplify(FrameCorrespondence &fC,
                                  Eigen::MatrixXd &Pre_JiTr,
                                  Eigen::MatrixXd &Pre_JjTr,
                                  Eigen::MatrixXd &Pre_JiTJi,
                                  Eigen::MatrixXd &Pre_JiTJj,
                                  Eigen::MatrixXd &Pre_JjTJi,
                                  Eigen::MatrixXd &Pre_JjTJj)
    {
        int valid_3d_cnt = fC.sparse_feature_cnt +  fC.dense_feature_cnt;
        // construct the four matrix based on pre-integrated points
        Pre_JiTr.setZero();
        Pre_JjTr.setZero();
        Pre_JiTJi.setZero();
        Pre_JiTJj.setZero();
        Pre_JjTJi.setZero();
        Pre_JjTJj.setZero();
        if (valid_3d_cnt < minimum_3d_correspondence)
        {
            return;
        }
        //prepare data
        pthread_mutex_lock (&mutex_pose);
        Eigen::Matrix3d R_ref = fC.frame_ref.pose_sophus[0].rotationMatrix();
        Eigen::Vector3d t_ref = fC.frame_ref.pose_sophus[0].translation();
        Eigen::Matrix3d R_new = fC.frame_new.pose_sophus[0].rotationMatrix();
        Eigen::Vector3d t_new = fC.frame_new.pose_sophus[0].translation();
        pthread_mutex_unlock(&mutex_pose);
        Eigen::Matrix3d Eye3x3;

        Eye3x3.setIdentity();
        Eigen::Matrix3d riWrj, riWri, rjWrj;
        riWrj = R_ref * fC.sum_p_ref_new * R_new.transpose();
        riWri = R_ref * fC.sum_p_ref_ref * R_ref.transpose();
        rjWrj = R_new * fC.sum_p_new_new * R_new.transpose();

        Eigen::Vector3d R_ref_sum_p_ref = R_ref * fC.sum_p_ref;
        Eigen::Vector3d R_new_sum_p_new = R_new * fC.sum_p_new;
        Eigen::Vector3d residual = R_ref_sum_p_ref + fC.sum_weight * (t_ref - t_new) - R_new_sum_p_new;
        //calculating JTr, see ProblemFormulation.pdf
        Pre_JiTr.block<3, 1>(0, 0) = residual;
        Pre_JiTr.block<3, 1>(3, 0) = Eigen::Vector3d(riWrj(2, 1) - riWrj(1, 2), -riWrj(2, 0) + riWrj(0, 2), riWrj(1, 0) - riWrj(0, 1))
                                     + R_ref_sum_p_ref.cross(t_ref - t_new);

        Pre_JjTr.block<3, 1>(0, 0) = residual;
        Pre_JjTr.block<3, 1>(3, 0) = Eigen::Vector3d(riWrj(2, 1) - riWrj(1, 2), -riWrj(2, 0) + riWrj(0, 2), riWrj(1, 0) - riWrj(0, 1))
                                     + R_new_sum_p_new.cross(t_ref - t_new);
        Pre_JjTr = -Pre_JjTr;

        //calculating JTJ
        Pre_JiTJi.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
        Pre_JiTJi.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_ref_sum_p_ref);
        Pre_JiTJi.block<3, 3>(3, 0) = -Pre_JiTJi.block<3, 3>(0, 3);
        Pre_JiTJi(3, 3) = riWri(2, 2) + riWri(1, 1);	Pre_JiTJi(3, 4) = -riWri(1, 0);					Pre_JiTJi(3, 5) = -riWri(2, 0);
        Pre_JiTJi(4, 3) = -riWri(0, 1);					Pre_JiTJi(4, 4) = riWri(0, 0) + riWri(2, 2);	Pre_JiTJi(4, 5) = -riWri(2, 1);
        Pre_JiTJi(5, 3) = -riWri(0, 2);					Pre_JiTJi(5, 4) = -riWri(1, 2);					Pre_JiTJi(5, 5) = riWri(0, 0) + riWri(1, 1);

        Pre_JjTJj.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
        Pre_JjTJj.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_new_sum_p_new);
        Pre_JjTJj.block<3, 3>(3, 0) = -Pre_JjTJj.block<3, 3>(0, 3);
        Pre_JjTJj(3, 3) = rjWrj(2, 2) + rjWrj(1, 1);	Pre_JjTJj(3, 4) = -rjWrj(1, 0);					Pre_JjTJj(3, 5) = -rjWrj(2, 0);
        Pre_JjTJj(4, 3) = -rjWrj(0, 1);					Pre_JjTJj(4, 4) = rjWrj(0, 0) + rjWrj(2, 2);	Pre_JjTJj(4, 5) = -rjWrj(2, 1);
        Pre_JjTJj(5, 3) = -rjWrj(0, 2);					Pre_JjTJj(5, 4) = -rjWrj(1, 2);					Pre_JjTJj(5, 5) = rjWrj(0, 0) + rjWrj(1, 1);

        Pre_JiTJj.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
        Pre_JiTJj.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_new_sum_p_new );
        Pre_JiTJj.block<3, 3>(3, 0) = -getSkewSymmetricMatrix(R_ref_sum_p_ref ).transpose();
        Pre_JiTJj(3, 3) = riWrj(2, 2) + riWrj(1, 1);	Pre_JiTJj(3, 4) = -riWrj(1, 0);	Pre_JiTJj(3, 5) = -riWrj(2, 0);
        Pre_JiTJj(4, 3) = -riWrj(0, 1);	Pre_JiTJj(4, 4) = riWrj(0, 0) + riWrj(2, 2);	Pre_JiTJj(4, 5) = -riWrj(2, 1);
        Pre_JiTJj(5, 3) = -riWrj(0, 2);	Pre_JiTJj(5, 4) = -riWrj(1, 2);		Pre_JiTJj(5, 5) = riWrj(0, 0) + riWrj(1, 1);
        Pre_JiTJj = -Pre_JiTJj;
        Pre_JjTJi = Pre_JiTJj.transpose();
    }




  template <class NumType>
  void addBlockToTriplets(std::vector<Eigen::Triplet<NumType>> &coeff, Eigen::MatrixXd b,
    int start_x,int start_y)
  {
    int rows = b.rows();
    int cols = b.cols();
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        coeff.push_back(Eigen::Triplet<NumType>(start_x+i,start_y+j,b(i,j)));
      }
    }


  }

typedef double SPARSE_MATRIX_NUM_TYPE;
#define USE_ROBUST_COST

  float initGraphHuberNorm(std::vector<FrameCorrespondence> &fCList, std::vector<Frame> &F)
  {

      int origin = 0;
      vector<float> average_error_per_frame(F.size());
      vector<int> keyframe_candidate_fcorrs;

      std::vector<int> keyframes;
      for (int i = 0; i < F.size(); i++)
      {
        if (F[i].is_keyframe && F[i].origin_index == origin)
        {
          keyframes.push_back(i);
        }
      }
      for (int i = 0; i < keyframes.size(); i++)
      {
        LOG_INFO(1) << i << " " << keyframes[i] << endl;
      }
      if (keyframes.size() < 3)
      {
        LOG_INFO(1) << "no need to optimize!" << endl;
        return -1;
      }
      int N = F.size();
      std::vector<int> getKeyFramePos(N);
      for (int i = 0; i < N; i++)
      {
        getKeyFramePos[i] = -1;
      }
      for (int i = 0; i < keyframes.size(); i++)
      {
        getKeyFramePos[keyframes[i]] = i;
      }

      for (int i = 0; i < fCList.size(); i++)
      {
          Frame &frame_ref = fCList[i].frame_ref;
        Frame &frame_new = fCList[i].frame_new;
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        if (frame_ref_pos < 0 || frame_new_pos < 0)
        {
          continue;
        }
        keyframe_candidate_fcorrs.push_back(i);
        float error = reprojection_error_3Dto3D(fCList[i]);
        if(g_para.debug_mode)
        {
  #if 0
          LOG_INFO(1) << frame_ref.frame_index << " " << frame_new.frame_index
                   << " " << error
                   << " " << fCList[i].sum_weight
                   << endl;
  #endif
        }

      }


      for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
      {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        if (frame_ref_pos < 0 || frame_new_pos < 0)
        {
          continue;
        }

        fCList[keyframe_candidate_fcorrs[i]].preIntegrateWithHuberNorm();

      }

  }

float optimizeKeyFrameMapRobust_local(vector<FrameCorrespondence> &fCList, vector<Frame> &F,
                                  std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,
                                  int origin, float robust_u,
                                  int flag_opti_count=0)
  {

    vector<int> keyframe_candidate_fcorrs;
    std::vector<int> keyframes_temp;
    std::vector<int> keyframes;


   // cout<<"优化的主序列中的关键帧序号:"<<endl;
    for (int i = 0; i < F.size(); i++)
    {
      if (F[i].is_keyframe && F[i].origin_index == origin)  //跟踪成功,而且关键帧在主序列中   主序列:如果track失败,回环检测成功才会回到主序列
      {
        keyframes_temp.push_back(i);
   //     cout<<" "<<i;
      }
    } 

    int local_start=0;
    if(keyframes_temp.size()>G_parameter.sliding_window_length)
    {
      for (int i = keyframes_temp.size()-G_parameter.sliding_window_length; i < keyframes_temp.size(); i++)
      {
        keyframes.push_back(keyframes_temp[i]);
      }
      local_start=1;
      g_global_start=1;
    }
    else
    {
      for (int i = 0; i < keyframes_temp.size(); i++)
      {
        keyframes.push_back(keyframes_temp[i]);
      }
    }
    
    // cout<<"参与优化的关键帧:"<<endl;
    // for (int i = 0; i < keyframes.size(); i++)
    // {
    //   cout<<keyframes[i]<<" ";
    //  }
    //  cout<<endl;

    if (keyframes.size() < 3)
    {
      return -1;
    }

    int N = F.size();
    std::vector<int> getKeyFramePos(N);
    std::vector<int> getKeyFramePos_all(N);
    for (int i = 0; i < N; i++)
    {
      getKeyFramePos[i] = -1;
      getKeyFramePos_all[i]=-1;
    }
    //是不是关键帧 是第几个关键帧
    for (int i = 0; i < keyframes.size(); i++)
    {
      getKeyFramePos[keyframes[i]] = i;
    }
    //所有的关键帧
    for (int i = 0; i < keyframes_temp.size(); i++)
    {
      getKeyFramePos_all[keyframes_temp[i]] = i;
    }
    static int revelent=0;  //when find wrong loop, cancel the long correspondig
    int latest_keyframe_index = 0;

    //帧对只在update keyframe经过回环检测后才加入，因为帧对一定是关于关键帧的
    for (int i = 0; i < fCList.size(); i++)
    {
      Frame &frame_ref = fCList[i].frame_ref;
      Frame &frame_new = fCList[i].frame_new;

      // 表明了是不是关键帧 是段中的第几个关键帧 
      int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
      int frame_new_pos = getKeyFramePos[frame_new.frame_index];

      int frame_ref_pos_all = getKeyFramePos_all[frame_ref.frame_index];
      int frame_new_pos_all = getKeyFramePos_all[frame_new.frame_index];

      //说明不是关键帧优化段中的关键帧
      if (frame_ref_pos < 0 || frame_new_pos < 0)
      {
        continue;
      }

      // 检测到回环时进行全局优化
      // static int record_=0;
      // if (i>record_)
      // {
      //   if((frame_new_pos_all-frame_ref_pos_all)>G_parameter.sliding_window_length)  
      //   {
      //     //进行一次视觉的全局优化
      //     cout<<"侦测到全局回环"<<endl;
      //     cout<<frame_new_pos_all<<"  "<<frame_ref_pos_all<<"  "<<G_parameter.sliding_window_length<<endl;
      //     record_=i; 
      //     if(G_parameter.visual_loop==1)
      //     {
      //       cout<<"进行一次全局优化"<<endl;
      //       optimizeKeyFrameMapRobust(fCList,F,kflist,origin,robust_u);  
      //       cout<<"全局优化结束"<<endl;
      //       return 0;
      //     }
      //     else
      //     {
      //       cout<<"检测到段之外的回环，但不进行全局优化"<<endl;
      //     }   
      //   } 
      // }      

      latest_keyframe_index = frame_ref.frame_index > latest_keyframe_index ? frame_ref.frame_index : latest_keyframe_index;
      latest_keyframe_index = frame_new.frame_index > latest_keyframe_index ? frame_new.frame_index : latest_keyframe_index;

      keyframe_candidate_fcorrs.push_back(i);
      // float error = reprojection_error_3Dto3D(fCList[i]);
    }

    // 如果检测到错误的回环，那么接下来的几帧都不使用之前的回环
    // if(revelent>0)
    // {
      
    //     revelent--;
    //     // static int start_serch;
    //     //the last keyframe find wrong loop, it's easy to find the wrong loop now ,so cancel the ahead corresponding
    //     // find the lasted keyframe,cancel the ahead correxponding
    //     int min_dist = 1e4;  
    //     for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
    //     {
    //         Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
    //         Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;
    //         // 保留最近的关键帧的位置产生的帧对
    //         int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
    //         int frame_new_pos = getKeyFramePos[frame_new.frame_index];
    //         if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
    //         {
    //           if(fabs(frame_ref_pos - frame_new_pos) < min_dist && fCList[keyframe_candidate_fcorrs[i]].matches.size() > 0)
    //           {
    //             cout << "try to remove: "<< fabs(frame_ref_pos - frame_new_pos) << " " << min_dist << " " << frame_ref.frame_index << " " << frame_new.frame_index << endl;
    //             min_dist = fabs(frame_ref_pos - frame_new_pos);
    //           }
    //         }
    //     }

    //     for (int i = 0; i < keyframe_candidate_f corrs.size(); i++)
    //     {
    //       Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
    //       Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

    //       int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
    //       int frame_new_pos = getKeyFramePos[frame_new.frame_index];
    //       if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
    //       {
    //         if(fabs(frame_ref_pos - frame_new_pos) > min_dist) // if the keyframe internal bigger than 7,cancel the corresponding
    //         {
    //           fCList[keyframe_candidate_fcorrs[i]].reset();
    //         }
    //       }
    //     }
    // }

    std::vector<float> weight_per_pair(keyframe_candidate_fcorrs.size());

    // will be replaced by conjugate gradient descent.
    int optNum = keyframes.size() - 1;
    Eigen::MatrixXd J, err;
    Eigen::MatrixXd delta(6 * optNum, 1), JTe(6 * optNum, 1);
    Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> JTJ(6 * optNum, 6 * optNum);

    // the solver is only built at the first iteration
    Eigen::SimplicialLDLT	<Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> > SimplicialLDLTSolver;
    std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> coeff; //jtj
    coeff.reserve(6 * 6 * 4 * fCList.size());
    Eigen::MatrixXd JiTJi_pre(6, 6), JiTJj_pre(6, 6), JjTJi_pre(6, 6), JjTJj_pre(6, 6), JiTe_pre(6, 1), JjTe_pre(6, 1);

    // // 保留优化前的结果
    // PoseSE3dList frame_poses;
    // vector<Vector3d> sudu;
    // vector<Vector3d> db_g;
    // vector<Vector3d> db_a;
    // Matrix3d old_rotation=imu_to_cam_rota;
    // Vector3d old_transformation=imu_to_cam_trans;

    // pthread_mutex_lock (&mutex_pose);
    // for(int i = 0; i < F.size(); i++)
    // {
    //     frame_poses.push_back(F[i].pose_sophus[0]);
    //     sudu.push_back(F[i]._V);
    //     db_g.push_back(F[i]._dBias_g);
    //     db_a.push_back(F[i]._dBias_a);
    // }
    // pthread_mutex_unlock(&mutex_pose);

    // 获得除了上一个关键帧之外的所有帧对
    vector<FrameCorrespondence> optimized_fc;
    for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
    {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

        //expect of the corresponding pair including the last keyframe
        if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
        {
            continue;
        }
        optimized_fc.push_back(fCList[keyframe_candidate_fcorrs[i]]);
    }

    pthread_mutex_lock (&mutex_pose);
    float init_error = reprojection_error_3Dto3D(optimized_fc);
    float init_total_error = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);
    pthread_mutex_unlock(&mutex_pose);
    
    // 前面的部分进行初始化，只进行视觉的优化
    int count_vis=G_parameter.ini_window_length;
    if(keyframes.size()==count_vis)
    {
      G_parameter.GN_number=8;
    }
    else
    {
      G_parameter.GN_number=3;
    }

    vector<Corr_plane> corr_plane; // 相关平面

    vector<double> ba_variation(G_parameter.GN_number);
    vector<double> bg_variation(G_parameter.GN_number);
    vector<double> r_variation(G_parameter.GN_number);
    vector<double> t_variation(G_parameter.GN_number);
    vector<double> v_variation(G_parameter.GN_number);

    vector<double> error_imu(G_parameter.GN_number+1);
    vector<double> error_imu_local(G_parameter.GN_number+1);
    vector<double> error_imu_with_bias(G_parameter.GN_number+1);
    vector<double> error_imu_local_with_bias(G_parameter.GN_number+1);

    // double st_time_gn= (double)cv::getTickCount();
    for (int iter = 0; iter < G_parameter.GN_number; iter++)
    {
      JTJ.setZero();
      coeff.clear();
      JTe.setZero(); 
      err.setZero();
      corr_plane.clear();

      double time_framePair;
      double time_generatingJacobian;
      double time_buildSolver;

      float robust_weight;

      static vector<int> wrong_loop;
      
      // keyframe_candidate_fcorrs： 相关帧都是关键帧的帧对的序号
      for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
      {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

        //得到帧作为关键帧的序号
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        // if (frame_new_pos < 0)
        if (frame_ref_pos < 0 || frame_new_pos < 0) 
        {
          continue;
        }
        //fCList[keyframe_candidate_fcorrs[i]].preIntegrateWithHuberNorm();

#if 0
        float error = reprojection_error_3Dto3D(fCList[keyframe_candidate_fcorrs[i]]);
        robust_weight = robust_u / (robust_u + error);
#else
        robust_weight = 1.0f;
#endif

        //1 根据关键帧位姿计算平面在世界坐标系下的位置
        //  通过标志位防止重复计算 每次帧的平面在世界坐标系下的位置只计算一次
        //2 进行平面匹配
        // Matrix3d rota_ref=frame_ref.pose_sophus[0].rotationMatrix();
        // Vector3d trans_ref=frame_ref.pose_sophus[0].translation();
        // Matrix3d rota_new=frame_new.pose_sophus[0].rotationMatrix();
        // Vector3d trans_new=frame_new.pose_sophus[0].translation();

        // double error_plane_all;
        // int plane_count=0;
        // cout<<"开始平面匹配"<<endl;
        // cout<<frame_ref.plane_v.size()<<endl;
        // cout<<frame_new.plane_v.size()<<endl;
        // Mat temp_ref=frame_ref.rgb.clone();
        // Mat temp_new=frame_new.rgb.clone();
        // 不能这样来匹配  对于每个帧相关，保存匹配的平面，只匹配一次
        // for(int k1=0;k1<frame_ref.plane_v.size();k1++)
        // {
        //   for(int k2=0;k2<frame_new.plane_v.size();k2++)
        //   {
        //     //平面参数转化到世界坐标系
        //     frame_ref.plane_v[k1].transform_plane(rota_ref,trans_ref);
        //     frame_new.plane_v[k2].transform_plane(rota_new,trans_new);
        //     double angle_error,dis_error;
        //     int res=PLANE::cal_plane_error(frame_ref.plane_v[k1],frame_new.plane_v[k2],angle_error,dis_error);
        //     //在这里显示匹配的平面进行debug,如果所有匹配的平面都正确,那么就ok
        //     if(res==1)
        //     {
        //         plane_count++;
        //         // cout<<"帧序号 平面序号"<<endl;
        //         // cout<<frame_ref_pos<<"   "<<frame_new_pos<<endl;
        //         // cout<<k1<<"   "<<k2<<endl<<endl;
                
        //         int height_t=frame_ref.refined_depth.rows;
        //         int width_t=frame_ref.refined_depth.cols;
        //         Mat pic_ref_t=PLANE::get_single_plane(height_t,width_t,frame_ref.plane_v[k1].point_order);
        //         Mat pic_new_t=PLANE::get_single_plane(height_t,width_t,frame_new.plane_v[k2].point_order);          

        //         Vec3b color_add;
        //         switch(plane_count)
        //         {
        //           case 1:color_add=Vec3b(0,150,0);       break;
        //           case 2:color_add=Vec3b(200,0,0);       break;
        //           case 3:color_add=Vec3b(0,0,200);       break;
        //           case 4:color_add=Vec3b(0,150,150);     break;
        //           default:color_add=Vec3b(150,150,0);    break;
        //         }

        //         for (int i= 0; i < temp_ref.rows; i++) //访问 
        //         {
        //             for (int j = 0; j < temp_ref.cols; j++)   
        //             { 
        //                 if(pic_ref_t.at<uchar>(i,j)>0)
        //                 {
        //                   temp_ref.at<Vec3b>(i,j)=temp_ref.at<Vec3b>(i,j)+color_add;
        //                 }
        //                 if(pic_new_t.at<uchar>(i,j)>0)
        //                 {
        //                   temp_new.at<Vec3b>(i,j)=temp_new.at<Vec3b>(i,j)+color_add;
        //                 } 
        //                     // temp_new.at<Vec3b>(i,j)=temp_new.at<Vec3b>(i,j)+Vec3b(200,0,0);
        //             }
        //         }

        //         // PLANE::show_by_order("plane_ref",pic_ref_t,0);
        //         // PLANE::show_by_order("rgb_ref",frame_ref.rgb,1);
        //         // PLANE::show_by_order("seg_ref",frame_ref.seg_plane,2);

        //         // // frame_new_pos
        //         // PLANE::show_by_order("plane_new",pic_new_t,3);
        //         // PLANE::show_by_order("rgb_new",frame_new.rgb,4);
        //         // PLANE::show_by_order("seg_new",frame_new.seg_plane,5);
        //         // waitKey(0);

        //         // error_plane_all+=error_t;    //通过计算总的平面error来判断是否正确 
        //         //算法正确的判断依据:1 平面匹配正确  2 误差下降正确

        //         Corr_plane corr_t;
        //         corr_t.frame_1=frame_ref;
        //         corr_t.plane_1=k1;
        //         corr_t.frame_pose1=frame_ref_pos;
        //         corr_t.frame_2=frame_new;
        //         corr_t.plane_2=k2;
        //         corr_t.frame_pose2=frame_new_pos;
        //         corr_plane.push_back(corr_t);
        //     }

        //   }
        // }

        //如果有匹配平面,那么显示出来
        // if(plane_count>0)
        // {
        //   PLANE::show_by_order("plane_ref1",temp_ref,0);
        //   PLANE::show_by_order("plane_new1",temp_new,1);
        //   waitKey(0);
        // }
        
      if(G_parameter.flag_youhua==2)
      {
        ComputeJacobianInfo(fCList[keyframe_candidate_fcorrs[i]],
                          JiTe_pre,
                          JjTe_pre,
                          JiTJi_pre,
                          JiTJj_pre,
                          JjTJi_pre,
                          JjTJj_pre);
      }
      else if(G_parameter.flag_youhua==3)
      {
        ComputeJacobianInfo_simplify(fCList[keyframe_candidate_fcorrs[i]],
                          JiTe_pre,
                          JjTe_pre,
                          JiTJi_pre,
                          JiTJj_pre,
                          JjTJi_pre,
                          JjTJj_pre);
      }

        JiTe_pre *= robust_weight;
        JjTe_pre *= robust_weight;
        JiTJi_pre *= robust_weight;
        JiTJj_pre *= robust_weight; 
        JjTJj_pre *= robust_weight;
        JjTJi_pre *= robust_weight;

        if (frame_ref_pos == 0)
        {
          addBlockToTriplets(coeff, JjTJj_pre, (frame_new_pos - 1) * 6, (frame_new_pos - 1) * 6);
          JTe.block<6, 1>((frame_new_pos - 1) * 6, 0) += JjTe_pre;
        }
        else
        {
          addBlockToTriplets(coeff, JiTJi_pre, (frame_ref_pos - 1) * 6, (frame_ref_pos - 1) * 6);
          addBlockToTriplets(coeff, JiTJj_pre, (frame_ref_pos - 1) * 6, (frame_new_pos - 1) * 6);
          addBlockToTriplets(coeff, JjTJi_pre, (frame_new_pos - 1) * 6, (frame_ref_pos - 1) * 6);
          addBlockToTriplets(coeff, JjTJj_pre, (frame_new_pos - 1) * 6, (frame_new_pos - 1) * 6);
          JTe.block<6, 1>((frame_ref_pos - 1) * 6, 0) += JiTe_pre;
          JTe.block<6, 1>((frame_new_pos - 1) * 6, 0) += JjTe_pre;
        }
      }

      //----------------------------------------------------------------------plane jacobian
      // //构建平面jacobian 和 residual
      // Eigen::MatrixXd J_plane(4*corr_plane.size() , 6 * optNum );
      // Eigen::MatrixXd r_plane(4*corr_plane.size() , 1 );
      // J_plane.setZero();
      // r_plane.setZero();
      // // cout<<"corresponding plane count:"<<corr_plane.size()<<endl;

      // for(int k2=0;k2<corr_plane.size();k2++)
      // {
      //   if (corr_plane[k2].frame_pose1 == 0||corr_plane[k2].frame_pose2 == 0)
      //   {
      //       //这里加上之后,在后面直接加在大矩阵上,就ok了

      //   }
      //   else
      //   {       
      //       //                      3       3  ...   3     3
      //       //                     ref_r  ref_t    new_r  new_t 
      //       // normal error 3       o                o
      //       // dis error    1       o       o        o      o

      //       Eigen::Matrix3d r_1_ =corr_plane[k2].frame_1.pose_sophus[0].matrix().block<3, 3>(0, 0);
      //       Eigen::Vector3d t_1_ =corr_plane[k2].frame_1.pose_sophus[0].matrix().block<3, 1>(0, 3);
      //       Eigen::Matrix3d r_2_ =corr_plane[k2].frame_2.pose_sophus[0].matrix().block<3, 3>(0, 0);
      //       Eigen::Vector3d t_2_ =corr_plane[k2].frame_2.pose_sophus[0].matrix().block<3, 1>(0, 3);

      //       PLANE::Plane_param corr_1 = corr_plane[k2].frame_1.plane_v[corr_plane[k2].plane_1];
      //       PLANE::Plane_param corr_2 = corr_plane[k2].frame_2.plane_v[corr_plane[k2].plane_2];

      //         // ref error
      //       J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose1 - 1) * 6)  =  -corr_1.flag_* Sophus::SO3d::hat(r_1_*corr_1.normal_cam);
      //       // J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose1- 1) * 6+3) = 0;
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose1 - 1) * 6)=-corr_1.flag_*corr_1.normal_cam.transpose()*r_1_.transpose()* Sophus::SO3d::hat(t_1_);
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose1 - 1) * 6+3)=-corr_1.flag_*corr_1.normal_cam.transpose()*r_1_.transpose();

      //       J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose2 - 1) * 6)=corr_2.flag_* Sophus::SO3d::hat(r_2_*corr_2.normal_cam);
      //       // J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose2 - 1) * 6+3)=0;
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose2 - 1) * 6)=corr_2.flag_*corr_2.normal_cam.transpose()*r_2_.transpose()*Sophus::SO3d::hat(t_2_);
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose2 - 1) * 6)=corr_2.flag_*corr_2.normal_cam.transpose()*r_2_.transpose();

      //       r_plane.block<3, 1>(4*k2, 0) =corr_1.flag_*r_1_*corr_1.normal_cam-corr_2.flag_*r_2_*corr_2.normal_cam;
      //       r_plane(4*k2+3, 0)=
      //         corr_1.flag_*(corr_1.dis_cam-corr_1.normal_cam.transpose()*r_1_.transpose()*t_1_)
      //       -corr_2.flag_*(corr_2.dis_cam-corr_2.normal_cam.transpose()*r_2_.transpose()*t_2_);
      //   }       
      // }

      // //直接计算JTJ 和 JTR
      // Eigen::MatrixXd JTJ_plane(6 * optNum, 6 * optNum);
      // JTJ_plane=J_plane.transpose()*J_plane;
      // Eigen::MatrixXd JTR_plane(6 * optNum , 1 );
      // JTR_plane=J_plane.transpose()*r_plane;
     
      if(G_parameter.flag_youhua==2)
      {
        JTJ.setFromTriplets(coeff.begin(), coeff.end());

        SimplicialLDLTSolver.compute(JTJ);
        delta = SimplicialLDLTSolver.solve(JTe);

        cout<<"updated delta  "<<delta.norm()<<endl;
        cout<<"average updated delta：  "<<delta.norm()/(double)keyframes.size()<<endl;

        pthread_mutex_lock (&mutex_pose);
        for (int i = 1; i < keyframes.size(); i++)
        {
          Eigen::VectorXd delta_i = delta.block<6, 1>(6 * (i - 1), 0);
          if(isnan(delta_i(0)))
          {
            cout << "nan detected in pose update! " << endl;
            continue;
          }

          F[keyframes[i]].pose_sophus[0] = Sophus::SE3d::exp(delta_i).inverse() *
                  F[keyframes[i]].pose_sophus[0];
        }
        pthread_mutex_unlock(&mutex_pose);
      }
      else
      {
        //initialize all variables
        VIA_ALL via_all(optNum);
        via_all.initial_variable();
        via_all.set_zero_big(); 

        std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> cov_imu_all_inverse; 
        cov_imu_all_inverse.reserve(9*9*(optNum-1)+6*6*(optNum-1));
        cov_imu_all_inverse.clear();

        std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> J_sparce_15N_9; // 15N维变量加9维的变换矩阵和重力
        if(local_start==0)
        {
                            // (r v t bg ba)   (bg_d ba_d)     (gravity R T)
          int size_J_sparce_15N_9=9*(optNum-1)*30+6*(optNum-1)*12+9*(optNum-1)*9;
          J_sparce_15N_9.reserve(size_J_sparce_15N_9);
          J_sparce_15N_9.clear();
        }

        std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> J_sparce_15N; // 15N维变量
                          // (r v t bg ba)   (bg_d ba_d)     (gravity R T)
        int size_J_sparce_15N=9*(optNum-1)*30+6*(optNum-1)*12;
        J_sparce_15N.reserve(size_J_sparce_15N);
        J_sparce_15N.clear();


        for (int i = 1; i < keyframes.size() - 1; i++)
        {
          via_all.set_zero_small();

          Frame *frame_i, *frame_j;
          Eigen::Vector3d r_i;
          Eigen::Vector3d r_j;
          Eigen::Matrix3d R_i;
          Eigen::Matrix3d R_j;

          Eigen::Vector3d V_i;
          Eigen::Vector3d V_j;

          Eigen::Vector3d P_i;
          Eigen::Vector3d P_j;

          //运动误差
          Eigen::Vector3d R_error;
          Eigen::Vector3d V_error;
          Eigen::Vector3d P_error;
          double del_time;

          frame_i = &F[keyframes[i]];
          frame_j = &F[keyframes[i + 1]];

          pthread_mutex_lock (&mutex_pose);
          //得到旋转向量和旋转矩阵
          r_i = frame_i->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
          r_j = frame_j->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
          R_i = frame_i->pose_sophus[0].matrix().block<3, 3>(0, 0);
          R_j = frame_j->pose_sophus[0].matrix().block<3, 3>(0, 0);

          V_i = frame_i->_V;
          V_j = frame_j->_V;

          P_i = frame_i->pose_sophus[0].matrix().block<3, 1>(0, 3);
          P_j = frame_j->pose_sophus[0].matrix().block<3, 1>(0, 3);
          pthread_mutex_unlock(&mutex_pose);

          del_time = frame_i->imu_res._delta_time;

          //通过比较积分段的序号,可以保证积分段正确
          if((frame_j->time_stamp-frame_i->time_stamp)!=del_time
              ||frame_i->imu_res.frame_index_qian!=frame_i->frame_index
              ||frame_i->imu_res.frame_index_hou!=frame_j->frame_index)
          {
              cout<<"积分段错误"<<endl;
              cout<<endl;
              // cout<<F[294].imu_res._delta_time<<endl;

              cout<<frame_i->frame_index<<endl;
              cout<<frame_i->is_keyframe<<endl;
              cout<<frame_i->tracking_success<<endl;
              cout<<frame_i->origin_index<<endl<<endl;

              cout<<frame_j->frame_index<<endl;
              cout<<frame_j->is_keyframe<<endl;
              cout<<frame_j->tracking_success<<endl;
              cout<<frame_j->origin_index<<endl<<endl;

              cout<<frame_j->time_stamp-frame_i->time_stamp<<endl;
              cout<<del_time<<endl;
              //这里的时间相等,说明imu数据的积分没有问题
              cout<<"integration time not equal!";
              exit(1);
          }

          pthread_mutex_lock (&mutex_pose);
          pthread_mutex_lock (&mutex_g_R_T);

          Eigen::Matrix3d RiR0=R_i*imu_to_cam_rota;
          Eigen::Matrix3d RjR0=R_j*imu_to_cam_rota;
          Eigen::Vector3d PiP0=P_i+R_i*imu_to_cam_trans;
          Eigen::Vector3d PjP0=P_j+R_j*imu_to_cam_trans;
          
          set_orthogonal(frame_i->imu_res._delta_R);
          //计算residual errors
          Eigen::Matrix3d Eb = Sophus::SO3d::exp(frame_i->imu_res._J_R_Biasg * frame_i->_dBias_g).matrix();
          Eigen::Matrix3d temp2 = frame_i->imu_res._delta_R * Eb;
          Eigen::Matrix3d temp = temp2.transpose() * RiR0.transpose() * RjR0;

          Sophus::SO3d SO3_R(temp);
          R_error = SO3_R.log();

          V_error = RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time)
                    - (frame_i->imu_res._delta_V + frame_i->imu_res._J_V_Biasg * frame_i->_dBias_g
                        + frame_i->imu_res._J_V_Biasa * frame_i->_dBias_a);

          P_error = RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 * rota_gravity*initial_gravity * del_time * del_time)
                    - (frame_i->imu_res._delta_P + frame_i->imu_res._J_P_Biasg * frame_i->_dBias_g
                        + frame_i->imu_res._J_P_Biasa * frame_i->_dBias_a);

          pthread_mutex_unlock(&mutex_g_R_T);
          pthread_mutex_unlock(&mutex_pose);

          Eigen::Matrix3d jrinv = JacobianRInv(R_error);
          Matrix3d danwei = Matrix3d::Identity();

          //-----------------------------------------------------r t v
          //                      0   3   6   9      0    3
          //                      t1  r1  t2  r2     v1   v2
          //               0 dr       o       o      
          //               3 dv       o              o    o
          //               6 dp   o   o   o   o      o      

          //dr
          via_all.via_t_r.Jm.block<3, 3>(0, 3)=-jrinv* RjR0.transpose();
          via_all.via_t_r.Jm.block<3, 3>(0, 9)=jrinv* RjR0.transpose();
          //dv
          pthread_mutex_lock (&mutex_g_R_T);
          via_all.via_t_r.Jm.block<3, 3>(3, 3)= RiR0.transpose()*Sophus::SO3d::hat(V_j - V_i - rota_gravity*initial_gravity*del_time);
          //dp
          via_all.via_t_r.Jm.block<3, 3>(6, 3)=RiR0.transpose()*Sophus::SO3d::hat(PjP0-PiP0-V_i*del_time-0.5*rota_gravity*initial_gravity*del_time*del_time)
                                              +RiR0.transpose()*Sophus::SO3d::hat(R_i*imu_to_cam_trans);
          via_all.via_t_r.Jm.block<3, 3>(6, 9)=-RiR0.transpose()*Sophus::SO3d::hat(R_j*imu_to_cam_trans);
          pthread_mutex_unlock(&mutex_g_R_T);
          via_all.via_t_r.Jm.block<3, 3>(6, 0)=-RiR0.transpose();
          via_all.via_t_r.Jm.block<3, 3>(6, 6)=RiR0.transpose();


          via_all.via_t_r.rx.block<3, 1>(0, 0) = R_error;
          via_all.via_t_r.rx.block<3, 1>(3, 0) = V_error;
          via_all.via_t_r.rx.block<3, 1>(6, 0) = P_error;
          
          //---
          via_all.via_t_r.J.block<9, 12>(9 * (i - 1), 6 * (i - 1)) = via_all.via_t_r.Jm;
          via_all.via_t_r.err.block<9, 1>(9 * (i - 1), 0) = via_all.via_t_r.rx; 
          //---

          via_all.via_v2.Jm.block<3, 3>(3, 0)=-RiR0.transpose();
          via_all.via_v2.Jm.block<3, 3>(3, 3)=RiR0.transpose();
          via_all.via_v2.Jm.block<3, 3>(6, 0)=-RiR0.transpose()*del_time;
          //---
          via_all.via_v2.J.block<9, 6>(9 * (i - 1), 3 * (i - 1)) = via_all.via_v2.Jm;
          //----


          MatrixXd cov_temp=frame_i->imu_res._cov_rvp;
          cov_temp=cov_temp.inverse();
          //协方差稀疏矩阵
          for(int i2=0;i2<9;i2++)
          {
            for(int i3=0;i3<9;i3++)
            {
              cov_imu_all_inverse.push_back(Eigen::Triplet<double>( 9*(i-1)+i2,9*(i-1)+i3 ,cov_temp(i2,i3)));
            }
          }


        if(local_start==0)
        {
          //稀疏矩阵赋值
          for(int i2=0;i2<9;i2++)
          {
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,6*(i-1)+i3,via_all.via_t_r.Jm(i2,i3)));
            }
            for(int i3=0;i3<6;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,6*optNum+3*(i-1)+i3,via_all.via_v2.Jm(i2,i3))); 
            }
          }
        }
        else
        {
          //稀疏矩阵赋值
          for(int i2=0;i2<9;i2++)
          {
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N.push_back(Eigen::Triplet<double>(9*(i-1)+i2,6*(i-1)+i3,via_all.via_t_r.Jm(i2,i3)));
            }
            for(int i3=0;i3<6;i3++)
            {
              J_sparce_15N.push_back(Eigen::Triplet<double>(9*(i-1)+i2,6*optNum+3*(i-1)+i3,via_all.via_v2.Jm(i2,i3))); 
            }
          }
        }
                  
          //---------------------------------------------bias
          //                        0     3     6     9
          //                        bg1   ba1   bg2   ba2
          //               0  dr    o    
          //               3  dv    o     o
          //               6  dp    o     o
          //               9  d_bg  o            o
          //               12 d_ba        o            o

          Eigen::Matrix3d Epb = Sophus::SO3d::exp(R_error).matrix();

          pthread_mutex_lock (&mutex_pose);
          // bg1
          via_all.via_bias.Jm.block<3, 3>(0, 0) =-jrinv * Epb.transpose() * JacobianR(frame_i->imu_res._J_R_Biasg * frame_i->_dBias_g) *frame_i->imu_res._J_R_Biasg;
          via_all.via_bias.Jm.block<3, 3>(3, 0) = -frame_i->imu_res._J_V_Biasg;
          via_all.via_bias.Jm.block<3, 3>(6, 0) = -frame_i->imu_res._J_P_Biasg;

          // ba1
          via_all.via_bias.Jm.block<3, 3>(3, 3) = -frame_i->imu_res._J_V_Biasa;
          via_all.via_bias.Jm.block<3, 3>(6, 3) = -frame_i->imu_res._J_P_Biasa;
    
          //bg
          via_all.via_bias.Jm.block<3, 3>(9, 0) = -G_parameter.xishu_imu_bias_change*danwei;
          via_all.via_bias.Jm.block<3, 3>(9, 6) = G_parameter.xishu_imu_bias_change*danwei;
          //ba
          via_all.via_bias.Jm.block<3, 3>(12, 3) = -G_parameter.xishu_imu_bias_change*danwei;
          via_all.via_bias.Jm.block<3, 3>(12, 9) = G_parameter.xishu_imu_bias_change*danwei;

          via_all.via_bias.rx.block<3, 1>(0, 0) = R_error;
          via_all.via_bias.rx.block<3, 1>(3, 0) = V_error;
          via_all.via_bias.rx.block<3, 1>(6, 0) = P_error;
          via_all.via_bias.rx.block<3, 1>(9, 0) = G_parameter.xishu_imu_bias_change*((frame_j->_BiasGyr+frame_j->_dBias_g)-(frame_i->_BiasGyr+frame_i->_dBias_g));
          via_all.via_bias.rx.block<3, 1>(12, 0)= G_parameter.xishu_imu_bias_change*((frame_j->_BiasAcc+frame_j->_dBias_a)-(frame_i->_BiasAcc+frame_i->_dBias_a));

          pthread_mutex_unlock(&mutex_pose);
          via_all.via_bias.J.block<15, 12>(15 * (i - 1), 6 * (i - 1)) = via_all.via_bias.Jm;
          via_all.via_bias.err.block<15, 1>(15 * (i - 1), 0) = via_all.via_bias.rx;
      

          Matrix<double,6,6> temp_bias_c=del_time*cov_bias_instability;
          temp_bias_c=temp_bias_c.inverse();

          for(int i2=0;i2<6;i2++)
          {
            for(int i3=0;i3<6;i3++)
            {
              cov_imu_all_inverse.push_back(Eigen::Triplet<double>( 9*(optNum-1)+6*(i-1)+i2,9*(optNum-1)+6*(i-1)+i3 ,temp_bias_c(i2,i3)));
            }
          }


          //稀疏矩阵赋值  零偏
        if(local_start==0)
        {
          for(int i2=0;i2<9;i2++)
          {            
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,9*optNum+6*(i-1)+i3,via_all.via_bias.Jm(i2,i3))); 
            }
          }
          //            零偏变化
          for(int i2=0;i2<6;i2++)
          {            
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(optNum-1)+6*(i-1)+i2,9*optNum+6*(i-1)+i3,via_all.via_bias.Jm(9+i2,i3))); 
            }
          }
        }
        else
        {
          for(int i2=0;i2<9;i2++)
          {            
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N.push_back(Eigen::Triplet<double>(9*(i-1)+i2,9*optNum+6*(i-1)+i3,via_all.via_bias.Jm(i2,i3))); 
            }
          }
          //            零偏变化
          for(int i2=0;i2<6;i2++)
          {            
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N.push_back(Eigen::Triplet<double>(9*(optNum-1)+6*(i-1)+i2,9*optNum+6*(i-1)+i3,via_all.via_bias.Jm(9+i2,i3))); 
            }
          }   
        }
        
          //-----------------------------------------------------R P gravity
          //                      0   3   6
          //                      R0  P0  R_g
          //               0 dr   o   
          //               3 dv   o       o
          //               6 dp   o   o   o


          // R0:
          via_all.via_g_R_P.Jm.block<3, 3>(0, 0) = jrinv * (-RjR0.transpose() * RiR0+danwei);
          via_all.via_g_R_P.Jm.block<3, 3>(3, 0) = Sophus::SO3d::hat(RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time));
          via_all.via_g_R_P.Jm.block<3, 3>(6, 0) =
          Sophus::SO3d::hat( RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 *rota_gravity*initial_gravity * del_time * del_time));

          // P0:
          via_all.via_g_R_P.Jm.block<3, 3>(6, 3) = RiR0.transpose() *(R_j-R_i);

          if(G_parameter.gravity_opti_method==0)
          {
            // R_g:
            via_all.via_g_R_P.Jm.block<3, 3>(3, 6) = RiR0.transpose()*del_time*rota_gravity*Sophus::SO3d::hat(initial_gravity);
            via_all.via_g_R_P.Jm.block<3, 3>(6, 6) = 0.5*RiR0.transpose()*del_time*del_time*rota_gravity*Sophus::SO3d::hat(initial_gravity);
          }
          else if(G_parameter.gravity_opti_method==1)
          {
            via_all.via_g_R_P.Jm.block<3, 3>(3, 6) = -RiR0.transpose()*del_time;
            via_all.via_g_R_P.Jm.block<3, 3>(6, 6) = -0.5*RiR0.transpose()*del_time*del_time;
          }
          
          via_all.via_g_R_P.J.block<9, 9>(9 * (i - 1), 0) = via_all.via_g_R_P.Jm;

          if(local_start==0)
          {
            //稀疏矩阵赋值
            for(int i2=0;i2<9;i2++)
            {            
              for(int i3=0;i3<9;i3++)
              {
                J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,15*optNum+i3,via_all.via_g_R_P.Jm(i2,i3))); 
              }
            }
            
          }
        }

        if(local_start)   
        {

          Eigen::MatrixXd delta_grt(9,1);

          int number_for_detect=15;   //用于检测imu位置变换的error段个数

          // 只取部分优化的一部分来判断imu位置的变化
          Eigen::MatrixXd  j_grt=via_all.via_g_R_P.J.block(9*(optNum-1 -  number_for_detect),0,9*number_for_detect,9);
          Eigen::MatrixXd  r_grt=via_all.via_t_r.err.block(9*(optNum-1 -  number_for_detect),0,9*number_for_detect,1);

          Eigen::MatrixXd jtj_grt=j_grt.transpose()*j_grt;
          Eigen::MatrixXd jtr_grt=j_grt.transpose()*r_grt;

          delta_grt=jtj_grt.ldlt().solve(-jtr_grt);
          // cout<<"变换矩阵delta_norm: "<<delta_grt.norm()<<endl;

          if(delta_grt.norm()>1.6)
          {
            cout<<"imu 位置变化"<<endl;
            // exit(1);
          }
        }

        Eigen::MatrixXd delta_imu_vis(9 * optNum,1);
        delta_imu_vis.setZero();

        // bias change error
        Eigen::MatrixXd r_bias_re(6*optNum-6, 1);
        r_bias_re.setZero();

        // 通过状态机选择优化方式
        int opti_method=0;  // 不优化
        if(keyframes.size()<count_vis||(keyframes.size()==count_vis&&iter<2))
        {
          opti_method=1;  // 视觉优化
        }
        else if(  ( (keyframes.size()>count_vis&&iter>=0) ||   (keyframes.size()==count_vis&&iter>=2)  )    &&local_start==0)  
        {
          opti_method=2;  // 优化全部变量
        }
        else if(local_start==1)
        {
          opti_method=3;  // 不优化变换矩阵和重力
        }
        // 在部分优化中通过视觉来更新新的帧的位姿

        //先优化r t 然后优化v gtavity 然后优化全部变量  -> 第二部没必要   先优化r t ，然后直接优化全部变量就可以了
        if(opti_method==1)   // optimize r and t by visual
        {
          Eigen::SparseMatrix<double> JTJ_fastgo(6 * optNum, 6 * optNum);
          JTJ_fastgo.setFromTriplets(coeff.begin(), coeff.end());
          Eigen::MatrixXd JTJ_visual(6 * optNum, 6 * optNum);
          JTJ_visual.setZero();
          matrix_sparse_to_dense(JTJ_visual,JTJ_fastgo);

          Eigen::MatrixXd delta_rt_(6*optNum,1);
          delta_rt_.setZero();
          delta_rt_= JTJ_visual.ldlt().solve(-JTe);

          delta_imu_vis.setZero();
          via_all.via_bias.delta.setZero();
          via_all.via_g_R_P.delta.setZero();

          delta_imu_vis.block(0,0,6*optNum,1) =delta_rt_;
        }

        //优化全部变量  
        if(opti_method==2)   //  otimize r t v bg ba g R T
        {
            Eigen::SparseMatrix<double> JTJ_fastgo_15n9(15 * optNum+9, 15 * optNum+9);
            JTJ_fastgo_15n9.setFromTriplets(coeff.begin(), coeff.end());

            for(int i=0;i<optNum-1;i++)
            {
              r_bias_re.block<6, 1>(6*i, 0)   =via_all.via_bias.err.block<6, 1>(15*i+9, 0);
            }

            //验证得到了正确的稀疏矩阵
            Eigen::SparseMatrix<double> J_s_15n9(15*optNum-15, 15 * optNum+9);
            J_s_15n9.setZero();
            J_s_15n9.setFromTriplets(J_sparce_15N_9.begin(), J_sparce_15N_9.end());
            
            Eigen::MatrixXd r_all_with_r_p_g(15*optNum-15,1);
            r_all_with_r_p_g.setZero();
            r_all_with_r_p_g.block(0,0,9*optNum-9,1)=via_all.via_t_r.err;
            r_all_with_r_p_g.block(9*optNum-9,0,6*optNum-6,1)=r_bias_re;
            
            Eigen::SparseMatrix<double> r_s_15n9(15*optNum-15,1);
            matrix_dense_to_sparse(r_s_15n9,r_all_with_r_p_g);
 
            Eigen::SparseMatrix<double> JTJ_s_15n9=J_s_15n9.transpose()*J_s_15n9;
            Eigen::SparseMatrix<double> JTr_s_15n9=J_s_15n9.transpose()*r_s_15n9;

            JTJ_s_15n9=JTJ_s_15n9+G_parameter.xishu_visual*JTJ_fastgo_15n9;

            Eigen::MatrixXd JTr_d_15n9(15 * optNum+9,1);
            matrix_sparse_to_dense(JTr_d_15n9,JTr_s_15n9);
            JTr_d_15n9.block(0,0,6*optNum,1)=JTr_d_15n9.block(0,0,6*optNum,1)+G_parameter.xishu_visual*JTe;

            for(int i1=0;i1<6*optNum;i1++)
            {
              JTr_s_15n9.coeffRef(i1,0)+=G_parameter.xishu_visual*JTe(i1,0);
            }

            Eigen::MatrixXd delta_15n9(15 * optNum+9,1);
            Eigen::SimplicialLDLT	<Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> > sldlt;
            sldlt.compute(JTJ_s_15n9);
            delta_15n9 = sldlt.solve(-JTr_d_15n9);
   
            delta_imu_vis =delta_15n9.block(0,0,9*optNum,1);
            via_all.via_bias.delta =delta_15n9.block(9*optNum,0,6*optNum,1);
            // via_all.via_r_p_g.delta.setZero();
            via_all.via_g_R_P.delta=delta_15n9.block(15*optNum,0,9,1);
        }

        if(opti_method==3)  //  otimize r t v bg ba 
        {
            Eigen::SparseMatrix<double> COV_all_INV(15 * optNum-15, 15 * optNum-15);
            COV_all_INV.setFromTriplets(cov_imu_all_inverse.begin(), cov_imu_all_inverse.end());

            Eigen::SparseMatrix<double> JTJ_fastgo_15n(15 * optNum, 15 * optNum);
            JTJ_fastgo_15n.setFromTriplets(coeff.begin(), coeff.end());

            for(int i=0;i<optNum-1;i++)
            {
              r_bias_re.block<6, 1>(6*i, 0)   =via_all.via_bias.err.block<6, 1>(15*i+9, 0);
            }

            Eigen::SparseMatrix<double> J_s_15n(15*optNum-15, 15 * optNum);
            J_s_15n.setZero();
            J_s_15n.setFromTriplets(J_sparce_15N.begin(), J_sparce_15N.end());
            
            Eigen::MatrixXd r_all_no_rtg(15*optNum-15,1);
            r_all_no_rtg.setZero();
            r_all_no_rtg.block(0,0,9*optNum-9,1)=via_all.via_t_r.err;
            r_all_no_rtg.block(9*optNum-9,0,6*optNum-6,1)=r_bias_re;
            
            Eigen::SparseMatrix<double> r_s_15n(15*optNum-15,1);
            matrix_dense_to_sparse(r_s_15n,r_all_no_rtg);
 
            Eigen::SparseMatrix<double> JTJ_s_15n;
            Eigen::SparseMatrix<double> JTr_s_15n;

            if(G_parameter.use_cov)
            {
              JTJ_s_15n=J_s_15n.transpose()*COV_all_INV*J_s_15n;
              JTr_s_15n=J_s_15n.transpose()*COV_all_INV*r_s_15n;
            }
            else
            {
              JTJ_s_15n=J_s_15n.transpose()*J_s_15n;
              JTr_s_15n=J_s_15n.transpose()*r_s_15n;
            }


            JTJ_s_15n=JTJ_s_15n+G_parameter.xishu_visual*JTJ_fastgo_15n;

            Eigen::MatrixXd JTr_d_15n(15 * optNum,1);
            matrix_sparse_to_dense(JTr_d_15n,JTr_s_15n);
            JTr_d_15n.block(0,0,6*optNum,1)=JTr_d_15n.block(0,0,6*optNum,1)+G_parameter.xishu_visual*JTe;


            Eigen::MatrixXd delta_15n(15 * optNum,1);
            Eigen::SimplicialLDLT	<Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> > sldlt;
            sldlt.compute(JTJ_s_15n);
            delta_15n = sldlt.solve(-JTr_d_15n);
          
            delta_imu_vis =delta_15n.block(0,0,9*optNum,1);
            via_all.via_bias.delta =delta_15n.block(9*optNum,0,6*optNum,1);
            via_all.via_g_R_P.delta.setZero();
        }

        if(G_parameter.out_residual)
        {
          Eigen::MatrixXd r_temp(3*optNum,1);
          Eigen::MatrixXd t_temp(3*optNum,1);
          Eigen::MatrixXd v_temp(3*optNum,1);
          Eigen::MatrixXd bg_temp(3*optNum,1);
          Eigen::MatrixXd ba_temp(3*optNum,1);
          for(int i=0;i<optNum;i++)
          {
              r_temp.block(3*i,0,3,1)=delta_imu_vis.block(6*i,0,3,1);
              t_temp.block(3*i,0,3,1)=delta_imu_vis.block(6*i+3,0,3,1);
              v_temp.block(3*i,0,3,1)=delta_imu_vis.block(6*optNum+3*i,0,3,1);
              bg_temp.block(3*i,0,3,1)=via_all.via_bias.delta.block(6*i,0,3,1);
              ba_temp.block(3*i,0,3,1)=via_all.via_bias.delta.block(6*i+3,0,3,1);
          }
          
          r_variation[iter]=r_temp.norm();
          t_variation[iter]=t_temp.norm();
          v_variation[iter]=v_temp.norm();
          bg_variation[iter]=bg_temp.norm();
          ba_variation[iter]=ba_temp.norm();

          error_imu[iter]=via_all.via_t_r.err.norm();
          error_imu_local[iter]=via_all.via_t_r.err.block(0, 0,9 * optNum-18,1).norm();

          error_imu_with_bias[iter]=via_all.via_bias.err.norm();
          error_imu_local_with_bias[iter]=via_all.via_bias.err.block(0,0,15*optNum-30,1).norm();
        }

        pthread_mutex_lock (&mutex_pose);
        for (int i = 1; i < keyframes.size(); i++)
        {
          //不更新全局优化处理的帧
          if(F[keyframes[i]].frame_index<count_global_opti)
          {
            continue;
          }

          Eigen::Vector3d delta_v;
          Eigen::Vector3d delta_r;
          Eigen::Vector3d delta_t;

       
          delta_v  =  delta_imu_vis.block<3, 1>(6*optNum+3*(i-1), 0);

          F[keyframes[i]]._V += G_parameter.xishu_V * delta_v;

          // 注意，这里不更新位姿是为了让位姿不抖动； 但是导致错误回环产生的时候不能更新位姿，  
          //两种解决方法：1 所以当产生大回环的时候，这里的位姿要进行更新 2 除非只有回环帧对，否则在回环检测中不通过回环帧定位
          if(local_start==0)
          {
            delta_t  =  delta_imu_vis.block<3, 1>(6 * (i - 1), 0);
            delta_r  =  delta_imu_vis.block<3, 1>(6 * (i - 1)+3, 0);
            Eigen::Matrix3d r1 = F[keyframes[i]].pose_sophus[0].matrix().block<3, 3>(0, 0);
            Eigen::Vector3d t1 = F[keyframes[i]].pose_sophus[0].matrix().block<3, 1>(0, 3);

            Sophus::SO3d Rd = Sophus::SO3d::exp(G_parameter.xishu_R*delta_r);
      
            Eigen::Matrix3d r2;
            Eigen::Vector3d t2;
            r2 = Rd.matrix()*r1;
            t2 = t1 + G_parameter.xishu_T * delta_t;

            set_orthogonal(r2);

            Sophus::SE3d SE3_Rt(r2, t2);    
            F[keyframes[i]].pose_sophus[0] = SE3_Rt;
          }
          // else   // 通过视觉的优化来更新帧位姿
          // {
          //   delta_t  =  delta_rt_.block<3, 1>(6 * (i - 1), 0);
          //   delta_r  =  delta_rt_.block<3, 1>(6 * (i - 1)+3, 0);
          //   // 这时候只通过视觉来更新新帧的位姿
          //   Eigen::Matrix3d r1 = F[keyframes[i]].pose_sophus[0].matrix().block<3, 3>(0, 0);
          //   Eigen::Vector3d t1 = F[keyframes[i]].pose_sophus[0].matrix().block<3, 1>(0, 3);

          //   Sophus::SO3d Rd = Sophus::SO3d::exp(G_parameter.xishu_R*delta_r);
      
          //   Eigen::Matrix3d r2;
          //   Eigen::Vector3d t2;
          //   r2 = Rd.matrix()*r1;
          //   t2 = t1 + G_parameter.xishu_T * delta_t;

          //   set_orthogonal(r2);

          //   Sophus::SE3d SE3_Rt(r2, t2);    
          //   F[keyframes[i]].pose_sophus[0] = SE3_Rt;
          // }
          

          Eigen::Vector3d delta_bg_d = via_all.via_bias.delta.block<3, 1>(6 * (i - 1), 0);
          Eigen::Vector3d delta_ba_d = via_all.via_bias.delta.block<3, 1>(6 * (i - 1)+3, 0);

          F[keyframes[i]]._dBias_g += G_parameter.xishu_bg_d * delta_bg_d;       
          F[keyframes[i]]._dBias_a += G_parameter.xishu_ba_d * delta_ba_d;

          //output the bias of a keyframe
          // if(G_parameter.out_bias==1&& i == (keyframes.size()-2))
          // if(G_parameter.out_bias==1)
          // {
          //   cout<<i<<" bias:bg ba  projection:"<<endl;
          //   cout<<F[keyframes[i]]._dBias_g.transpose()+F[keyframes[i]]._BiasGyr.transpose()<<endl;
          //   cout<<F[keyframes[i]]._dBias_a.transpose()+F[keyframes[i]]._BiasAcc.transpose()<<endl;  

          //   // // gravity projection
          //   // Vector3d gravity_( 0.3332, -8.575,-4.7432);
          //   // Eigen::Matrix3d r_temp = F[keyframes[i]].pose_sophus[0].matrix().block<3, 3>(0, 0);
          //   // Vector3d projection=r_temp.transpose()*(rota_gravity*initial_gravity-gravity_);
          //   // cout<<projection.transpose()<<endl;

          // }

          // if (i < keyframes.size() - 1)    //注意零偏是不包含最后一帧的！！！
          // {
          //   // Eigen::Vector3d delta_bg = via_all.via_bg.delta.block<3, 1>(3 * (i - 1), 0);
          //   // Eigen::Vector3d delta_ba = via_all.via_ba.delta.block<3, 1>(3 * (i - 1), 0);
          //   // F[keyframes[i]]._dBias_g += xishu_bg * delta_bg;
          //   // F[keyframes[i]]._dBias_a += xishu_ba * delta_ba;
          // }
        }
        pthread_mutex_unlock(&mutex_pose);

        pthread_mutex_lock (&mutex_g_R_T);
        
        if(G_parameter.gravity_opti_method==0)
        {
          Sophus::SO3d G_ro=Sophus::SO3d::exp(G_parameter.xishu_gravity*via_all.via_g_R_P.delta.block<3, 1>(6, 0));
          rota_gravity=rota_gravity*G_ro.matrix();
          set_orthogonal(rota_gravity);
        }
        else if(G_parameter.gravity_opti_method==1)
        {
          initial_gravity=initial_gravity+G_parameter.xishu_gravity*via_all.via_g_R_P.delta.block<3, 1>(6, 0);
          initial_gravity=G_parameter.gravity_norm*initial_gravity.normalized();
        }

        //update the transformation matrix
        Sophus::SO3d Rd0 = Sophus::SO3d::exp(G_parameter.xishu_rote*via_all.via_g_R_P.delta.block<3, 1>(0, 0));
        imu_to_cam_rota=imu_to_cam_rota*Rd0.matrix();
        set_orthogonal(imu_to_cam_rota);
        imu_to_cam_trans=imu_to_cam_trans+G_parameter.xishu_trans*via_all.via_g_R_P.delta.block<3, 1>(3, 0);
        pthread_mutex_unlock(&mutex_g_R_T);

        // out the transformation matrix 
        if(G_parameter.out_transformation && local_start==0)  
        {
          // cout<<"transformation matrix rotation translation gravity："<<endl;
          Sophus::SO3d R_ini(ini_imu_to_cam_rota);
          Eigen::Vector3d r_ini=R_ini.log();

          Sophus::SO3d R_now(imu_to_cam_rota);
          Eigen::Vector3d r_new=R_now.log();

          Sophus::SO3d R_gravity(rota_gravity);
          Eigen::Vector3d r_gravity=R_gravity.log();

          record_vector(r_new[0],125,"x");
          record_vector(r_new[1],125,"y");
          record_vector(r_new[2],125,"z");

          // double angle_r_g=PLANE::angle_cal(r_new,r_gravity);
          double angle_r_ini_r=PLANE::angle_cal(r_new,r_ini);

          cout<<local_start<<" "<<keyframes.size()<<" "<<iter<<" "
          <<" gravity: "<<endl<<rota_gravity*initial_gravity<<endl;
        }
        
          //if the bias of imu change too much,make the repeated integration to invent the  numerical error
          pthread_mutex_lock (&mutex_pose);
          for (int i = 1; i < keyframes.size()-1; i++)
          {
            if(F[keyframes[i]]._dBias_a.norm()>0.2||F[keyframes[i]]._dBias_g.norm()>0.1)
            {
              // cout<<endl<<endl;
              // cout<<"keyframe "<<i<<" make the repeated integration"<<endl;
              // cout<<F[keyframes[i]]._BiasAcc<<endl;
              // cout<<F[keyframes[i]]._dBias_a<<endl;
              // cout<<endl<<endl;

              F[keyframes[i]]._BiasGyr=F[keyframes[i]]._BiasGyr+F[keyframes[i]]._dBias_g;
              F[keyframes[i]]._BiasAcc=F[keyframes[i]]._BiasAcc+F[keyframes[i]]._dBias_a;
              F[keyframes[i]]._dBias_g.setZero();
              F[keyframes[i]]._dBias_a.setZero();

              IMUPreintegrator IMUPreInt;
              IMUPreInt.imu_index_qian=F[keyframes[i]].imu_res.imu_index_qian;
              IMUPreInt.imu_index_hou=F[keyframes[i]].imu_res.imu_index_hou;
              IMUPreInt.frame_index_qian=F[keyframes[i]].imu_res.frame_index_qian;
              IMUPreInt.frame_index_hou=F[keyframes[i]].imu_res.frame_index_hou;

              Vector3d _bg= F[keyframes[i]]._BiasGyr;
              Vector3d _ba= F[keyframes[i]]._BiasAcc;
      
              pthread_mutex_lock (&mutex_imu);
              double last_imu_time=F[keyframes[i]].time_stamp;
              int count_imu11=IMUPreInt.imu_index_qian;
              while(IMU_data_raw[count_imu11].time_stamp<F[keyframes[i+1]].time_stamp)
              {
                double dt = IMU_data_raw[count_imu11].time_stamp - last_imu_time;
                Vector3d g_=(IMU_data_raw[count_imu11]._g +IMU_data_raw[count_imu11-1]._g )/2;
                Vector3d a_=(IMU_data_raw[count_imu11]._a +IMU_data_raw[count_imu11-1]._a )/2;
                IMUPreInt.update(g_ - _bg, a_ - _ba, dt);
                last_imu_time=IMU_data_raw[count_imu11].time_stamp;
                count_imu11++;
              }
              if(count_imu11!=IMUPreInt.imu_index_hou)
              {
                cout<<"repeated integration error"<<endl;
                exit(1);
              }
              Vector3d g_=(IMU_data_raw[count_imu11]._g +IMU_data_raw[count_imu11-1]._g )/2;
              Vector3d a_=(IMU_data_raw[count_imu11]._a +IMU_data_raw[count_imu11-1]._a )/2;
              IMUPreInt.update(g_ - _bg,a_ - _ba, F[keyframes[i+1]].time_stamp-IMU_data_raw[count_imu11-1].time_stamp);
              pthread_mutex_unlock (&mutex_imu);

              F[keyframes[i]].imu_res=IMUPreInt;
            }
          }
          pthread_mutex_unlock(&mutex_pose);

      } //flag 3,imu optimization
    }  //GN optimization

    // double consume_time_gn= (double)cv::getTickCount() - st_time_gn;
    // consume_time_gn=consume_time_gn*1000/ cv::getTickFrequency();
    // cout<<"GN time2:"<<consume_time_gn<<endl;


    // if(G_parameter.out_residual)
    // {
    //   Eigen::MatrixXd  error_gn(9*optNum-9,1);
    //   Eigen::MatrixXd  error_gn_right_bias(9*optNum-9,1);
    //   Eigen::MatrixXd  error_gn_biaschange(15*optNum-15,1);
    //   for (int i = 1; i < keyframes.size() - 1; i++)
    //   {

    //     Frame *frame_i, *frame_j;
    //     Eigen::Vector3d r_i;
    //     Eigen::Vector3d r_j;
    //     Eigen::Matrix3d R_i;
    //     Eigen::Matrix3d R_j;

    //     Eigen::Vector3d V_i;
    //     Eigen::Vector3d V_j;

    //     Eigen::Vector3d P_i;
    //     Eigen::Vector3d P_j;

    //     //运动误差
    //     Eigen::Vector3d R_error;
    //     Eigen::Vector3d V_error;
    //     Eigen::Vector3d P_error;
    //     double del_time;

    //     frame_i = &F[keyframes[i]];
    //     frame_j = &F[keyframes[i + 1]];

    //     //得到旋转向量和旋转矩阵
    //     r_i = frame_i->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
    //     r_j = frame_j->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
    //     R_i = frame_i->pose_sophus[0].matrix().block<3, 3>(0, 0);
    //     R_j = frame_j->pose_sophus[0].matrix().block<3, 3>(0, 0);

    //     V_i = frame_i->_V;
    //     V_j = frame_j->_V;

    //     P_i = (frame_i->pose_sophus[0].matrix().block<3, 1>(0, 3));
    //     P_j = (frame_j->pose_sophus[0].matrix().block<3, 1>(0, 3));

    //     del_time = frame_i->imu_res._delta_time;

    //     //通过比较积分段的序号,可以保证积分段正确
    //     if((frame_j->time_stamp-frame_i->time_stamp)!=del_time
    //         ||frame_i->imu_res.frame_index_qian!=frame_i->frame_index
    //         ||frame_i->imu_res.frame_index_hou!=frame_j->frame_index)
    //     {
    //         cout<<"积分段错误"<<endl;
    //         cout<<endl;
    //         // cout<<F[294].imu_res._delta_time<<endl;

    //         cout<<frame_i->frame_index<<endl;
    //         cout<<frame_i->is_keyframe<<endl;
    //         cout<<frame_i->tracking_success<<endl;
    //         cout<<frame_i->origin_index<<endl<<endl;

    //         cout<<frame_j->frame_index<<endl;
    //         cout<<frame_j->is_keyframe<<endl;
    //         cout<<frame_j->tracking_success<<endl;
    //         cout<<frame_j->origin_index<<endl<<endl;

    //         cout<<frame_j->time_stamp-frame_i->time_stamp<<endl;
    //         cout<<del_time<<endl;
    //         //这里的时间相等,说明imu数据的积分没有问题
    //         cout<<"integration time not equal!";
    //         exit(1);
    //     }
    //     Eigen::Matrix3d RiR0=R_i*imu_to_cam_rota;
    //     Eigen::Matrix3d RjR0=R_j*imu_to_cam_rota;
    //     Eigen::Vector3d PiP0=P_i+R_i*imu_to_cam_trans;
    //     Eigen::Vector3d PjP0=P_j+R_j*imu_to_cam_trans;

    //     //计算residual errors
    //     Eigen::Matrix3d Eb = Sophus::SO3d::exp(frame_i->imu_res._J_R_Biasg * frame_i->_dBias_g).matrix();
    //     Eigen::Matrix3d temp2 = frame_i->imu_res._delta_R * Eb;
    //     Eigen::Matrix3d temp = temp2.transpose() * RiR0.transpose() * RjR0;

    //     Sophus::SO3d SO3_R(temp);
    //     R_error = SO3_R.log();

    //     V_error = RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time)
    //               - (frame_i->imu_res._delta_V + frame_i->imu_res._J_V_Biasg * frame_i->_dBias_g
    //                   + frame_i->imu_res._J_V_Biasa * frame_i->_dBias_a);

    //     P_error = RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 * rota_gravity*initial_gravity * del_time * del_time)
    //               - (frame_i->imu_res._delta_P + frame_i->imu_res._J_P_Biasg * frame_i->_dBias_g
    //                   + frame_i->imu_res._J_P_Biasa * frame_i->_dBias_a);

    //     //使用正确的bias，看是否因为error对bias不敏感
    //     // Eigen::Vector3d R_error_bias;
    //     // Eigen::Vector3d V_error_bias;
    //     // Eigen::Vector3d P_error_bias;

    //     // Eigen::Vector3d bias_a(0.1,-0.1,0.1);
    //     // R_error_bias = R_error;

    //     // V_error_bias = RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time)
    //     //           - (frame_i->imu_res._delta_V + frame_i->imu_res._J_V_Biasg * frame_i->_dBias_g
    //     //               + frame_i->imu_res._J_V_Biasa * bias_a);

    //     // P_error_bias = RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 * rota_gravity*initial_gravity * del_time * del_time)
    //     //           - (frame_i->imu_res._delta_P + frame_i->imu_res._J_P_Biasg * frame_i->_dBias_g
    //     //               + frame_i->imu_res._J_P_Biasa * bias_a);


    //     // error_gn_right_bias.block<3, 1>(9*i-9, 0)   = R_error_bias;
    //     // error_gn_right_bias.block<3, 1>(9*i-9+3, 0) = V_error_bias;
    //     // error_gn_right_bias.block<3, 1>(9*i-9+6, 0) = P_error_bias;


    //     error_gn.block<3, 1>(9*i-9, 0)   = R_error;
    //     error_gn.block<3, 1>(9*i-9+3, 0) = V_error;
    //     error_gn.block<3, 1>(9*i-9+6, 0) = P_error;

    //     error_gn_biaschange.block<3, 1>(15*i-15, 0) = R_error;
    //     error_gn_biaschange.block<3, 1>(15*i-15+3, 0) = V_error;
    //     error_gn_biaschange.block<3, 1>(15*i-15+6, 0) = P_error;
    //     error_gn_biaschange.block<3, 1>(15*i-15+9, 0)= (frame_j->_BiasGyr+frame_j->_dBias_g)-(frame_i->_BiasGyr+frame_i->_dBias_g);
    //     error_gn_biaschange.block<3, 1>(15*i-15+12, 0)= (frame_j->_BiasAcc+frame_j->_dBias_a)-(frame_i->_BiasAcc+frame_i->_dBias_a);

    //   }

    //   error_imu[G_parameter.GN_number]=error_gn.norm();
    //   error_imu_local[G_parameter.GN_number]=error_gn.block(0, 0,9 * optNum-18,1).norm();
    //   error_imu_with_bias[G_parameter.GN_number]=error_gn_biaschange.norm();
    //   error_imu_local_with_bias[G_parameter.GN_number]=error_gn_biaschange.block(0, 0,15 * optNum-30,1).norm();

    //   // cout<<"正确bias得到的error：  "<<error_gn_right_bias.norm()<<endl;
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  bias variation :"<<imu_bias_variation[iter]<<endl;
    //   // }
    //   // cout<<"local optimization:"<<endl;
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  r variation :"<<r_variation[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  t variation :"<<t_variation[iter]<<endl;
    //   // }      
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  v variation :"<<v_variation[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  bg variation :"<<bg_variation[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  ba variation :"<<ba_variation[iter]<<endl;
    //   // }
 

    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error :"<<error_imu[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error local :"<<error_imu_local[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error  with bias:"<<error_imu_with_bias[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error local with bias:"<<error_imu_local_with_bias[iter]<<endl;
    //   // }

    //   // // if the wrong loop happens,the total error decreases,
    //   // // while the error except of the frame correspondings of the last keyframe will increase.
    //   // float final_error = reprojection_error_3Dto3D(optimized_fc);
    //   // float final_total_error = reprojection_error_3Dto3D(fCList,keyframe_candidate_fcorrs);
    //   // cout << "init/final error " << init_error << "/" << final_error
    //   //     << "       " << init_total_error << "/" << final_total_error << endl;
    //   // cout<<"detect:  "<<(final_error - init_error) / init_error <<endl;

    // } 


    return 1;
  }

  float optimizeKeyFrameMap(vector<FrameCorrespondence> &fCList, vector<Frame> &F,
                            std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,int origin)
  {
    float robust_u = 1;
     for(int i = 0; i < 1; i++)
     {
       robust_u = 0.5 * robust_u;
       optimizeKeyFrameMapRobust_local(fCList,F,kflist,origin,robust_u);
     }
  }


  float reprojection_error_3Dto3D(const FrameCorrespondence &fC,  const Sophus::SE3d &relative_pose_from_ref_to_new)
  {
    Eigen::MatrixXd R_ref = relative_pose_from_ref_to_new.rotationMatrix();
    Eigen::Vector3d t_ref = relative_pose_from_ref_to_new.translation();
    // pre-integration method for norm-2 distance
    float total_error = 0;

    if (fC.sum_weight > 0)
    {
      total_error = fC.sum_p_ref_ref(0,0) + fC.sum_p_ref_ref(1,1) + fC.sum_p_ref_ref(2,2) +
          fC.sum_p_new_new(0,0) + fC.sum_p_new_new(1,1) + fC.sum_p_new_new(2,2) +
          fC.sum_weight * t_ref.transpose() * t_ref
        - 2 * (float)(t_ref.transpose() * fC.sum_p_new) + 2 * (float)(t_ref.transpose() * R_ref * fC.sum_p_ref)
        - 2 * R_ref.cwiseProduct(fC.sum_p_new_ref).sum();

      if(total_error < 0)
      {
        cout << "total error: " << total_error << endl;
      }
      else
      {
        total_error = sqrt(total_error)  / fC.sum_weight;
      }
    }
    return total_error;
  }
  float reprojection_error_3Dto3D(const FrameCorrespondence &fC)
  {
    return reprojection_error_3Dto3D(fC, fC.frame_new.pose_sophus[0].inverse() * fC.frame_ref.pose_sophus[0]);
  }

  float reprojection_error_3Dto3D(std::vector<FrameCorrespondence> fCList)
  {
    float average_reprojection_error = 0;
    float count = 0;
    for (int i = 0; i < fCList.size(); i++)
    {
      average_reprojection_error += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
      count += fCList[i].sum_weight;
    }
    return average_reprojection_error / count;
  }


  float reprojection_error_3Dto3D(std::vector<FrameCorrespondence> fCList, std::vector<int>candidates)
  {
    float average_reprojection_error = 0;
    float count = 0;
    for (int i = 0; i < candidates.size(); i++)
    {
      average_reprojection_error += reprojection_error_3Dto3D(fCList[candidates[i]]) * fCList[candidates[i]].sum_weight;
      count += fCList[candidates[i]].sum_weight;
    }
    return average_reprojection_error / count;
  }
  float reprojection_error_3Dto3D_perFrame(std::vector<FrameCorrespondence> fCList,
        Eigen::VectorXd &errorPerFrame,
        Eigen::VectorXd &pointsPerFrame,
        Eigen::VectorXd &connectionsPerFrame)
    {
        int frameNum = errorPerFrame.size();
        errorPerFrame.setZero();
        pointsPerFrame.setZero();
        connectionsPerFrame.setZero();
        float average_reprojection_error = 0;
        float count = 0;
        for (int i = 0; i < fCList.size(); i++)
        {
            average_reprojection_error += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            count += fCList[i].sum_weight;

            int ref_frame_index = fCList[i].frame_ref.frame_index;
            int new_frame_index = fCList[i].frame_new.frame_index;
            assert(ref_frame_index < frameNum);
            assert(new_frame_index < frameNum);
            errorPerFrame[ref_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            errorPerFrame[new_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            pointsPerFrame[ref_frame_index] += fCList[i].sum_weight;
            pointsPerFrame[new_frame_index] += fCList[i].sum_weight;
            connectionsPerFrame[ref_frame_index] += 1;
            connectionsPerFrame[new_frame_index] += 1;
        }

        for (int i = 0; i < frameNum; i++)
        {
            if (pointsPerFrame[i] < 1)
                errorPerFrame[i] = 1e8;
            else
                errorPerFrame[i] /= pointsPerFrame[i];
        }
        return average_reprojection_error / count;
    }
  float reprojection_error_3Dto3D_perFrame(std::vector<FrameCorrespondence> fCList,
        Eigen::VectorXd &errorPerFrame,
        Eigen::VectorXd &pointsPerFrame,
        Eigen::VectorXd &connectionsPerFrame,
    std::vector<std::vector<int> > &related_connections)
    {
        int frameNum = errorPerFrame.size();
        errorPerFrame.setZero();
        pointsPerFrame.setZero();
        float average_reprojection_error = 0;
        int count = 0;
        for (int i = 0; i < fCList.size(); i++)
        {
            average_reprojection_error += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            count += fCList[i].sum_weight;

            int ref_frame_index = fCList[i].frame_ref.frame_index;
            int new_frame_index = fCList[i].frame_new.frame_index;
            assert(ref_frame_index < frameNum);
            assert(new_frame_index < frameNum);
            errorPerFrame[ref_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            errorPerFrame[new_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            pointsPerFrame[ref_frame_index] += fCList[i].sum_weight;
            pointsPerFrame[new_frame_index] += fCList[i].sum_weight;
            connectionsPerFrame[ref_frame_index] += 1;
            connectionsPerFrame[new_frame_index] += 1;
            related_connections[ref_frame_index].push_back(i);
            related_connections[new_frame_index].push_back(i);
        }

        for (int i = 0; i < frameNum; i++)
        {
            if (pointsPerFrame[i] < 1)
                errorPerFrame[i] = 1e8;
            else
                errorPerFrame[i] /= pointsPerFrame[i];
        }
        return average_reprojection_error / count;
    }

  float reprojection_error_3Dto3D(Point3dList pt_ref,
    Point3dList pt_new,
        Sophus::SE3d relative_pose_from_ref_to_new,
        int use_huber_norm,
        float huber_norm_threshold)
    {
        float reprojection_error_3d = 0;

        if (use_huber_norm)
        {
            for (int i = 0; i < pt_ref.size(); i++)
            {
                Eigen::Vector3d reprojection_error = applyPose(relative_pose_from_ref_to_new,pt_ref[i]) - pt_new[i];

                float error = reprojection_error.norm() / pt_ref[i].z();
                float weight_huber = 1;
                if (error > 0.008)
                {
                    weight_huber = 0.008 / error;
                }
                error = error * error * weight_huber;
                reprojection_error_3d += error;
            }
        }
        else
        {
            for (int i = 0; i < pt_ref.size(); i++)
            {
                Eigen::Vector3d reprojection_error = applyPose(relative_pose_from_ref_to_new,pt_ref[i]) - pt_new[i];
                float error = reprojection_error.norm() / pt_ref[i].z();
                error = error * error;
                reprojection_error_3d += error;
            }
        }
    reprojection_error_3d = sqrt(reprojection_error_3d);
        reprojection_error_3d /= pt_ref.size();
        return reprojection_error_3d;
    }
  float reprojection_error_3Dto3D(Frame frame_ref,
        Frame frame_new,
    std::vector< cv::DMatch > inlier_matches,
        Sophus::SE3d relative_pose_from_ref_to_new,
        int use_huber_norm,
        float huber_norm_threshold)
    {
        Point3dList p_ref, p_new;
        p_ref.clear(); p_new.clear();
        p_ref.reserve(inlier_matches.size());
        p_new.reserve(inlier_matches.size());
        for (size_t i = 0; i < inlier_matches.size(); i++)
        {
            Eigen::Vector3d pt_ref(frame_ref.local_points[inlier_matches[i].queryIdx]);
            Eigen::Vector3d pt_new(frame_new.local_points[inlier_matches[i].trainIdx]);
            p_ref.push_back(pt_ref);
            p_new.push_back(pt_new);
        }
        return reprojection_error_3Dto3D(p_ref, p_new, relative_pose_from_ref_to_new, use_huber_norm, huber_norm_threshold);
    }


  Eigen::Matrix3d skewMatrixProduct(Eigen::Vector3d t1, Eigen::Vector3d t2)
  {
      Eigen::Matrix3d M;
      M(0, 0) = -t1(1)*t2(1) - t1(2)*t2(2); M(0, 1) = t1(1)*t2(0); M(0, 2) = t1(2)*t2(0);
      M(1, 0) = t1(0)*t2(1);	 M(1, 1) = -t1(2)*t2(2) - t1(0)*t2(0); M(1, 2) = t1(2)*t2(1);
      M(2, 0) = t1(0)*t2(2);   M(2, 1) = t1(1)*t2(2); M(2, 2) = -t1(1)*t2(1) - t1(0)*t2(0);
      return M;
  }

  Eigen::Matrix3d getSkewSymmetricMatrix(Eigen::Vector3d t)
  {
      Eigen::Matrix3d t_hat;
      t_hat << 0, -t(2), t(1),
          t(2), 0, -t(0),
          -t(1), t(0), 0;
      return t_hat;
  }


  float optimizeKeyFrameMapRobust_global(vector<FrameCorrespondence> &fCList, vector<Frame> &F,
                                std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,
                                int origin, float robust_u,
                                int flag_opti_count=0)
  {

    static int count_last_opti=0;
    int count_last_global_opti=count_last_opti;
    count_last_opti=count_global_opti;

    pthread_mutex_lock (&mutex_pose);
    int N = F.size();
    pthread_mutex_unlock(&mutex_pose);
   
   std::vector<int> keyframes;
   std::vector<int> keyframes_new; //全局优化新加的段中的关键帧
    //cout<<"优化的主序列中的关键帧序号:"<<endl;
    for (int i = 0; i < N; i++)
    {
      if (F[i].is_keyframe && F[i].origin_index == origin)  //跟踪成功,而且关键帧在主序列中   主序列:如果track失败,回环检测成功才会回到主序列
      {
        keyframes.push_back(i);
   //     cout<<" "<<i;
      }
      if(i>=count_last_global_opti && count_last_global_opti>0)
      {
        if (F[i].is_keyframe && F[i].origin_index == origin)  
        {
          keyframes_new.push_back(i);
        }
      }
    } 

    // cout<<"参与优化的关键帧:"<<endl;
    // for (int i = 0; i < keyframes.size(); i++)
    // {
    //   cout<<keyframes[i]<<" ";
    //  }
    //  cout<<endl;

    if (keyframes.size() < 3)
    {
      return -1;
    }

    std::vector<int> getKeyFramePos(N);

    for (int i = 0; i < N; i++)
    {
      getKeyFramePos[i] = -1;
    }
    //优化的关键帧
    for (int i = 0; i < keyframes.size(); i++)
    {
      getKeyFramePos[keyframes[i]] = i;
    }

    static int revelent=0;  //when find wrong loop, cancel the long correspondig
    int latest_keyframe_index = 0;

    //两个相关帧都是关键帧的帧相关
    vector<int> keyframe_candidate_fcorrs;
    for (int i = 0; i < fCList.size(); i++)
    {
      Frame &frame_ref = fCList[i].frame_ref;
      Frame &frame_new = fCList[i].frame_new;
      int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
      int frame_new_pos = getKeyFramePos[frame_new.frame_index];

      //说明不是关键帧优化段中的关键帧帧
      if (frame_ref_pos < 0 || frame_new_pos < 0)
      {
        continue;
      }

      //如果是sliding window，在这里可以进行检测到回环时的全局优化
      // static int record_=0;
      // if (i>record_)
      // {
      //   if((frame_new_pos_all-frame_ref_pos_all)>G_parameter.sliding_window_length)  
      //   {
      //     //进行一次视觉的全局优化
      //     cout<<"侦测到全局回环"<<endl;
      //     cout<<frame_new_pos_all<<"  "<<frame_ref_pos_all<<"  "<<G_parameter.sliding_window_length<<endl;
      //     record_=i; 
      //     if(G_parameter.visual_loop==1)
      //     {
      //       cout<<"进行一次全局优化"<<endl;
      //       optimizeKeyFrameMapRobust(fCList,F,kflist,origin,robust_u);  
      //       cout<<"全局优化结束"<<endl;
      //       return 0;
      //     }
      //     else
      //     {
      //       cout<<"检测到段之外的回环，但不进行全局优化"<<endl;
      //     }   
      //   } 
      // }      

      latest_keyframe_index = frame_ref.frame_index > latest_keyframe_index ? frame_ref.frame_index : latest_keyframe_index;
      latest_keyframe_index = frame_new.frame_index > latest_keyframe_index ? frame_new.frame_index : latest_keyframe_index;

      keyframe_candidate_fcorrs.push_back(i);
      // float error = reprojection_error_3Dto3D(fCList[i]);
    }


    // 如果检测到错误的回环，那么接下来的几次全局优化都不使用之前的回环
    if(revelent>0)
    {
      revelent--;

      for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
      {
          Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
          Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

          int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
          int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        // 新来的段中的帧
        if(frame_new.frame_index>(count_last_global_opti-1))
        {
          if(fabs(frame_new_pos - frame_ref_pos) > G_parameter.drop_corres_length)
          {
              fCList[keyframe_candidate_fcorrs[i]].reset();
          }    
        }
      }
    }

    std::vector<float> weight_per_pair(keyframe_candidate_fcorrs.size());

    // will be replaced by conjugate gradient descent.
    int optNum = keyframes.size() - 1;
    Eigen::MatrixXd J, err;
    Eigen::MatrixXd delta(6 * optNum, 1), JTe(6 * optNum, 1);
    Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> JTJ(6 * optNum, 6 * optNum);


    // the solver is only built at the first iteration
    Eigen::SimplicialLDLT	<Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> > SimplicialLDLTSolver;
    std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> coeff; //jtj
    coeff.reserve(6 * 6 * 4 * fCList.size());
    Eigen::MatrixXd JiTJi_pre(6, 6), JiTJj_pre(6, 6), JjTJi_pre(6, 6), JjTJj_pre(6, 6), JiTe_pre(6, 1), JjTe_pre(6, 1);

    //保留优化前的结果
    pthread_mutex_lock (&mutex_g_R_T);
    Matrix3d old_rotation=imu_to_cam_rota;
    Matrix3d old_rota_gravity=rota_gravity;
    Vector3d old_transformation=imu_to_cam_trans;
    pthread_mutex_unlock(&mutex_g_R_T);

    PoseSE3dList frame_poses;
    vector<Vector3d> sudu;
    vector<Vector3d> db_g;
    vector<Vector3d> db_a;
    frame_poses.reserve(N);
    pthread_mutex_lock (&mutex_pose);
    for(int i = 0; i < N; i++)
    {
        frame_poses.push_back(F[i].pose_sophus[0]);
        sudu.push_back(F[i]._V);
        db_g.push_back(F[i]._dBias_g);
        db_a.push_back(F[i]._dBias_a);
    }

    pthread_mutex_unlock(&mutex_pose);

    vector<FrameCorrespondence> optimized_fc;
    for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
    {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

        //expect of the corresponding pair including the last keyframe
        if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
        {
            continue;
        }

        if(count_last_global_opti!=0)
        {
          if(frame_ref.frame_index > (count_last_global_opti-1) || frame_new.frame_index > (count_last_global_opti-1))
          {
              continue;
          }
        }
        optimized_fc.push_back(fCList[keyframe_candidate_fcorrs[i]]);
    }

    vector<Corr_plane> corr_plane; // 相关平面

    vector<float> visual_local_error(G_parameter.GN_number+1);
    vector<float> visual_global_error(G_parameter.GN_number+1);

    vector<double> bg_variation(G_parameter.GN_number);
    vector<double> ba_variation(G_parameter.GN_number);    
    vector<double> bg_local_variation(G_parameter.GN_number);
    vector<double> ba_local_variation(G_parameter.GN_number);
    
    // 总的error  不包含新加的优化段的error  包含零偏的error
    vector<double> error_imu(G_parameter.GN_number+1);
    vector<double> error_imu_with_bias(G_parameter.GN_number+1);
    vector<double> error_imu_local(G_parameter.GN_number+1);
    vector<double> error_imu_local_with_bias(G_parameter.GN_number+1);


    vector<double> r_variation(G_parameter.GN_number);
    vector<double> t_variation(G_parameter.GN_number);
    vector<double> v_variation(G_parameter.GN_number);

    vector<double> r_variation_local(G_parameter.GN_number);
    vector<double> t_variation_local(G_parameter.GN_number);
    vector<double> v_variation_local(G_parameter.GN_number);

    int exit_flag_=0;


    double st_time_gn= (double)cv::getTickCount();
    for (int iter = 0; iter < G_parameter.GN_number; iter++)
    {
      double time_0= (double)cv::getTickCount();
      JTJ.setZero();
      coeff.clear();
      JTe.setZero(); 
      err.setZero();
      corr_plane.clear();

      double time_framePair;
      double time_generatingJacobian;
      double time_buildSolver;

      float robust_weight;

      static vector<int> wrong_loop;
      
      for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
      {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        // if (frame_new_pos < 0)
        if (frame_ref_pos < 0 || frame_new_pos < 0) 
        {
          continue;
        }
        //fCList[keyframe_candidate_fcorrs[i]].preIntegrateWithHuberNorm();

#if 0
        float error = reprojection_error_3Dto3D(fCList[keyframe_candidate_fcorrs[i]]);
        robust_weight = robust_u / (robust_u + error);
#else
        robust_weight = 1.0f;
#endif

        //1 根据关键帧位姿计算平面在世界坐标系下的位置
        //  通过标志位防止重复计算 每次帧的平面在世界坐标系下的位置只计算一次
        //2 进行平面匹配
        // Matrix3d rota_ref=frame_ref.pose_sophus[0].rotationMatrix();
        // Vector3d trans_ref=frame_ref.pose_sophus[0].translation();
        // Matrix3d rota_new=frame_new.pose_sophus[0].rotationMatrix();
        // Vector3d trans_new=frame_new.pose_sophus[0].translation();

        // double error_plane_all;
        int plane_count=0;
        // cout<<"开始平面匹配"<<endl;
        // cout<<frame_ref.plane_v.size()<<endl;
        // cout<<frame_new.plane_v.size()<<endl;
        // Mat temp_ref=frame_ref.rgb.clone();
        // Mat temp_new=frame_new.rgb.clone();
        // 不能这样来匹配  对于每个帧相关，保存匹配的平面，只匹配一次
        // for(int k1=0;k1<frame_ref.plane_v.size();k1++)
        // {
        //   for(int k2=0;k2<frame_new.plane_v.size();k2++)
        //   {
        //     //平面参数转化到世界坐标系
        //     frame_ref.plane_v[k1].transform_plane(rota_ref,trans_ref);
        //     frame_new.plane_v[k2].transform_plane(rota_new,trans_new);
        //     double angle_error,dis_error;
        //     int res=PLANE::cal_plane_error(frame_ref.plane_v[k1],frame_new.plane_v[k2],angle_error,dis_error);
        //     //在这里显示匹配的平面进行debug,如果所有匹配的平面都正确,那么就ok
        //     if(res==1)
        //     {
        //         plane_count++;
        //         // cout<<"帧序号 平面序号"<<endl;
        //         // cout<<frame_ref_pos<<"   "<<frame_new_pos<<endl;
        //         // cout<<k1<<"   "<<k2<<endl<<endl;
                
        //         int height_t=frame_ref.refined_depth.rows;
        //         int width_t=frame_ref.refined_depth.cols;
        //         Mat pic_ref_t=PLANE::get_single_plane(height_t,width_t,frame_ref.plane_v[k1].point_order);
        //         Mat pic_new_t=PLANE::get_single_plane(height_t,width_t,frame_new.plane_v[k2].point_order);          

        //         Vec3b color_add;
        //         switch(plane_count)
        //         {
        //           case 1:color_add=Vec3b(0,150,0);       break;
        //           case 2:color_add=Vec3b(200,0,0);       break;
        //           case 3:color_add=Vec3b(0,0,200);       break;
        //           case 4:color_add=Vec3b(0,150,150);     break;
        //           default:color_add=Vec3b(150,150,0);    break;
        //         }

        //         for (int i= 0; i < temp_ref.rows; i++) //访问 
        //         {
        //             for (int j = 0; j < temp_ref.cols; j++)   
        //             { 
        //                 if(pic_ref_t.at<uchar>(i,j)>0)
        //                 {
        //                   temp_ref.at<Vec3b>(i,j)=temp_ref.at<Vec3b>(i,j)+color_add;
        //                 }
        //                 if(pic_new_t.at<uchar>(i,j)>0)
        //                 {
        //                   temp_new.at<Vec3b>(i,j)=temp_new.at<Vec3b>(i,j)+color_add;
        //                 } 
        //                     // temp_new.at<Vec3b>(i,j)=temp_new.at<Vec3b>(i,j)+Vec3b(200,0,0);
        //             }
        //         }

        //         // PLANE::show_by_order("plane_ref",pic_ref_t,0);
        //         // PLANE::show_by_order("rgb_ref",frame_ref.rgb,1);
        //         // PLANE::show_by_order("seg_ref",frame_ref.seg_plane,2);

        //         // // frame_new_pos
        //         // PLANE::show_by_order("plane_new",pic_new_t,3);
        //         // PLANE::show_by_order("rgb_new",frame_new.rgb,4);
        //         // PLANE::show_by_order("seg_new",frame_new.seg_plane,5);
        //         // waitKey(0);

        //         // error_plane_all+=error_t;    //通过计算总的平面error来判断是否正确 
        //         //算法正确的判断依据:1 平面匹配正确  2 误差下降正确

        //         Corr_plane corr_t;
        //         corr_t.frame_1=frame_ref;
        //         corr_t.plane_1=k1;
        //         corr_t.frame_pose1=frame_ref_pos;
        //         corr_t.frame_2=frame_new;
        //         corr_t.plane_2=k2;
        //         corr_t.frame_pose2=frame_new_pos;
        //         corr_plane.push_back(corr_t);
        //     }

        //   }
        // }

        //如果有匹配平面,那么显示出来
        // if(plane_count>0)
        // {
        //   PLANE::show_by_order("plane_ref1",temp_ref,0);
        //   PLANE::show_by_order("plane_new1",temp_new,1);
        //   waitKey(0);
        // }
        
      if(G_parameter.flag_youhua==2)
      {
        ComputeJacobianInfo(fCList[keyframe_candidate_fcorrs[i]],
                          JiTe_pre,
                          JjTe_pre,
                          JiTJi_pre,
                          JiTJj_pre,
                          JjTJi_pre,
                          JjTJj_pre);
      }
      else if(G_parameter.flag_youhua==3)
      {
        ComputeJacobianInfo_simplify(fCList[keyframe_candidate_fcorrs[i]],
                          JiTe_pre,
                          JjTe_pre,
                          JiTJi_pre,
                          JiTJj_pre,
                          JjTJi_pre,
                          JjTJj_pre);
      }

        JiTe_pre *= robust_weight;
        JjTe_pre *= robust_weight;
        JiTJi_pre *= robust_weight;
        JiTJj_pre *= robust_weight;
        JjTJj_pre *= robust_weight;
        JjTJi_pre *= robust_weight;

        if (frame_ref_pos == 0)
        {
          addBlockToTriplets(coeff, JjTJj_pre, (frame_new_pos - 1) * 6, (frame_new_pos - 1) * 6);
          JTe.block<6, 1>((frame_new_pos - 1) * 6, 0) += JjTe_pre;
        }
        else
        {
          addBlockToTriplets(coeff, JiTJi_pre, (frame_ref_pos - 1) * 6, (frame_ref_pos - 1) * 6);
          addBlockToTriplets(coeff, JiTJj_pre, (frame_ref_pos - 1) * 6, (frame_new_pos - 1) * 6);
          addBlockToTriplets(coeff, JjTJi_pre, (frame_new_pos - 1) * 6, (frame_ref_pos - 1) * 6);
          addBlockToTriplets(coeff, JjTJj_pre, (frame_new_pos - 1) * 6, (frame_new_pos - 1) * 6);
          JTe.block<6, 1>((frame_ref_pos - 1) * 6, 0) += JiTe_pre;
          JTe.block<6, 1>((frame_new_pos - 1) * 6, 0) += JjTe_pre;
        }
      }
      double time_1= (double)cv::getTickCount();

      //----------------------------------------------------------------------plane jacobian
      //构建平面jacobian 和 residual
      // Eigen::MatrixXd J_plane(4*corr_plane.size() , 6 * optNum );
      // Eigen::MatrixXd r_plane(4*corr_plane.size() , 1 );
      // J_plane.setZero();
      // r_plane.setZero();
      // cout<<"corresponding plane count:"<<corr_plane.size()<<endl;

      // for(int k2=0;k2<corr_plane.size();k2++)
      // {
      //   if (corr_plane[k2].frame_pose1 == 0||corr_plane[k2].frame_pose2 == 0)
      //   {
      //       //这里加上之后,在后面直接加在大矩阵上,就ok了

      //   }
      //   else
      //   {       
      //       //                      3       3  ...   3     3
      //       //                     ref_r  ref_t    new_r  new_t 
      //       // normal error 3       o                o
      //       // dis error    1       o       o        o      o
      //       pthread_mutex_lock (&mutex_pose);
      //       Eigen::Matrix3d r_1_ =corr_plane[k2].frame_1.pose_sophus[0].matrix().block<3, 3>(0, 0);
      //       Eigen::Vector3d t_1_ =corr_plane[k2].frame_1.pose_sophus[0].matrix().block<3, 1>(0, 3);
      //       Eigen::Matrix3d r_2_ =corr_plane[k2].frame_2.pose_sophus[0].matrix().block<3, 3>(0, 0);
      //       Eigen::Vector3d t_2_ =corr_plane[k2].frame_2.pose_sophus[0].matrix().block<3, 1>(0, 3);
      //       pthread_mutex_unlock(&mutex_pose);

      //       PLANE::Plane_param corr_1 = corr_plane[k2].frame_1.plane_v[corr_plane[k2].plane_1];
      //       PLANE::Plane_param corr_2 = corr_plane[k2].frame_2.plane_v[corr_plane[k2].plane_2];

      //         // ref error
      //       J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose1 - 1) * 6)  =  -corr_1.flag_* Sophus::SO3d::hat(r_1_*corr_1.normal_cam);
      //       // J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose1- 1) * 6+3) = 0;
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose1 - 1) * 6)=-corr_1.flag_*corr_1.normal_cam.transpose()*r_1_.transpose()* Sophus::SO3d::hat(t_1_);
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose1 - 1) * 6+3)=-corr_1.flag_*corr_1.normal_cam.transpose()*r_1_.transpose();

      //       J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose2 - 1) * 6)=corr_2.flag_* Sophus::SO3d::hat(r_2_*corr_2.normal_cam);
      //       // J_plane.block<3,3>(4*k2,(corr_plane[k2].frame_pose2 - 1) * 6+3)=0;
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose2 - 1) * 6)=corr_2.flag_*corr_2.normal_cam.transpose()*r_2_.transpose()*Sophus::SO3d::hat(t_2_);
      //       J_plane.block<1,3>(4*k2+3,(corr_plane[k2].frame_pose2 - 1) * 6)=corr_2.flag_*corr_2.normal_cam.transpose()*r_2_.transpose();

      //       r_plane.block<3, 1>(4*k2, 0) =corr_1.flag_*r_1_*corr_1.normal_cam-corr_2.flag_*r_2_*corr_2.normal_cam;
      //       r_plane(4*k2+3, 0)=
      //         corr_1.flag_*(corr_1.dis_cam-corr_1.normal_cam.transpose()*r_1_.transpose()*t_1_)
      //       -corr_2.flag_*(corr_2.dis_cam-corr_2.normal_cam.transpose()*r_2_.transpose()*t_2_);
      //   }       
      // }

      // //直接计算JTJ 和 JTR
      // Eigen::MatrixXd JTJ_plane(6 * optNum, 6 * optNum);
      // JTJ_plane=J_plane.transpose()*J_plane;
      // Eigen::MatrixXd JTR_plane(6 * optNum , 1 );
      // JTR_plane=J_plane.transpose()*r_plane;
     
      if(G_parameter.flag_youhua==2)
      {

        visual_local_error[iter] = reprojection_error_3Dto3D(optimized_fc);
        visual_global_error[iter] = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);

        JTJ.setFromTriplets(coeff.begin(), coeff.end());

        SimplicialLDLTSolver.compute(JTJ);
        delta = SimplicialLDLTSolver.solve(JTe);

        cout<<"updated delta  "<<delta.norm()<<endl;
        cout<<"average updated delta：  "<<delta.norm()/(double)keyframes.size()<<endl;

        pthread_mutex_lock(&mutex_pose);
        for (int i = 1; i < keyframes.size(); i++)
        {
          Eigen::VectorXd delta_i = delta.block<6, 1>(6 * (i - 1), 0);
          if(isnan(delta_i(0)))
          {
            cout << "nan detected in pose update! " << endl;
            continue;
          }

          F[keyframes[i]].pose_sophus[0] = Sophus::SE3d::exp(delta_i).inverse() *
                  F[keyframes[i]].pose_sophus[0];
        }
        pthread_mutex_unlock(&mutex_pose);
      }
      else
      {
        
        Eigen::MatrixXd r_ALL(15*optNum-15 , 1);

        std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> cov_imu_all_inverse; 
        cov_imu_all_inverse.reserve(9*9*(optNum-1)+6*6*(optNum-1));
        cov_imu_all_inverse.clear();
   
        std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> J_sparce_15N_9; // 15N维变量加9维的变换矩阵和重力
                          // (r v t bg ba)   (bg_d ba_d)     (gravity R T)
        int size_J_sparce_15N_9=9*(optNum-1)*30+6*(optNum-1)*12+9*(optNum-1)*9;
        J_sparce_15N_9.reserve(size_J_sparce_15N_9);
        J_sparce_15N_9.clear();

        Eigen::MatrixXd tr_temp_j(9 , 12);
        Eigen::MatrixXd v_temp_j(9 , 6);
        Eigen::MatrixXd bias_temp_j(15, 12);
        Eigen::MatrixXd g_R_P_temp_j(9, 9);

        for (int i = 1; i < keyframes.size() - 1; i++)
        {
  
          Frame *frame_i, *frame_j;
          Eigen::Vector3d r_i;
          Eigen::Vector3d r_j;
          Eigen::Matrix3d R_i;
          Eigen::Matrix3d R_j;

          Eigen::Vector3d V_i;
          Eigen::Vector3d V_j;

          Eigen::Vector3d P_i;
          Eigen::Vector3d P_j;

          //运动误差
          Eigen::Vector3d R_error;
          Eigen::Vector3d V_error;
          Eigen::Vector3d P_error;
          double del_time;

          frame_i = &F[keyframes[i]];
          frame_j = &F[keyframes[i + 1]];

          pthread_mutex_lock (&mutex_pose);
          //得到旋转向量和旋转矩阵
          r_i = frame_i->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
          r_j = frame_j->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
          R_i = frame_i->pose_sophus[0].matrix().block<3, 3>(0, 0);
          R_j = frame_j->pose_sophus[0].matrix().block<3, 3>(0, 0);

          V_i = frame_i->_V;
          V_j = frame_j->_V;

          P_i = frame_i->pose_sophus[0].matrix().block<3, 1>(0, 3);
          P_j = frame_j->pose_sophus[0].matrix().block<3, 1>(0, 3);
          pthread_mutex_unlock(&mutex_pose);
          del_time = frame_i->imu_res._delta_time;

          //通过比较积分段的序号,可以保证积分段正确
          if((frame_j->time_stamp-frame_i->time_stamp)!=del_time
              ||frame_i->imu_res.frame_index_qian!=frame_i->frame_index
              ||frame_i->imu_res.frame_index_hou!=frame_j->frame_index)
          {
              cout<<"积分段错误"<<endl;
              cout<<endl;
              // cout<<F[294].imu_res._delta_time<<endl;

              cout<<frame_i->frame_index<<endl;
              cout<<frame_i->is_keyframe<<endl;
              cout<<frame_i->tracking_success<<endl;
              cout<<frame_i->origin_index<<endl<<endl;

              cout<<frame_j->frame_index<<endl;
              cout<<frame_j->is_keyframe<<endl;
              cout<<frame_j->tracking_success<<endl;
              cout<<frame_j->origin_index<<endl<<endl;

              cout<<frame_j->time_stamp-frame_i->time_stamp<<endl;
              cout<<del_time<<endl;
              //这里的时间相等,说明imu数据的积分没有问题
              cout<<"integration time not equal!";
              exit(1);
          }
          pthread_mutex_lock (&mutex_pose);
          pthread_mutex_lock (&mutex_g_R_T);
          Eigen::Matrix3d RiR0=R_i*imu_to_cam_rota;
          Eigen::Matrix3d RjR0=R_j*imu_to_cam_rota;
          Eigen::Vector3d PiP0=P_i+R_i*imu_to_cam_trans;
          Eigen::Vector3d PjP0=P_j+R_j*imu_to_cam_trans;
          

          set_orthogonal(frame_i->imu_res._delta_R);
          //计算residual errors
          Eigen::Matrix3d Eb = Sophus::SO3d::exp(frame_i->imu_res._J_R_Biasg * frame_i->_dBias_g).matrix();
          Eigen::Matrix3d temp2 = frame_i->imu_res._delta_R * Eb;
          Eigen::Matrix3d temp = temp2.transpose() * RiR0.transpose() * RjR0;

          Sophus::SO3d SO3_R(temp);
          R_error = SO3_R.log();

          V_error = RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time)
                    - (frame_i->imu_res._delta_V + frame_i->imu_res._J_V_Biasg * frame_i->_dBias_g
                        + frame_i->imu_res._J_V_Biasa * frame_i->_dBias_a);

          P_error = RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 * rota_gravity*initial_gravity * del_time * del_time)
                    - (frame_i->imu_res._delta_P + frame_i->imu_res._J_P_Biasg * frame_i->_dBias_g
                        + frame_i->imu_res._J_P_Biasa * frame_i->_dBias_a);
          pthread_mutex_unlock(&mutex_g_R_T);
          pthread_mutex_unlock(&mutex_pose);

          Eigen::Matrix3d jrinv = JacobianRInv(R_error);
          Matrix3d danwei = Matrix3d::Identity();


          //-----------------------------------------------------r t v
          //                      0   3   6   9      0    3
          //                      t1  r1  t2  r2     v1   v2
          //               0 dr       o       o      
          //               3 dv       o              o    o
          //               6 dp   o   o   o   o      o      

          tr_temp_j.setZero();

          //dr
          tr_temp_j.block<3, 3>(0, 3)=-jrinv* RjR0.transpose();
          tr_temp_j.block<3, 3>(0, 9)=jrinv* RjR0.transpose();
          //dv
          pthread_mutex_lock (&mutex_g_R_T);
          tr_temp_j.block<3, 3>(3, 3)= RiR0.transpose()*Sophus::SO3d::hat(V_j - V_i - rota_gravity*initial_gravity*del_time);
          //dp
          tr_temp_j.block<3, 3>(6, 3)=RiR0.transpose()*Sophus::SO3d::hat(PjP0-PiP0-V_i*del_time-0.5*rota_gravity*initial_gravity*del_time*del_time)
                                              +RiR0.transpose()*Sophus::SO3d::hat(R_i*imu_to_cam_trans);
          tr_temp_j.block<3, 3>(6, 9)=-RiR0.transpose()*Sophus::SO3d::hat(R_j*imu_to_cam_trans);
          pthread_mutex_unlock(&mutex_g_R_T);
          tr_temp_j.block<3, 3>(6, 0)=-RiR0.transpose();
          tr_temp_j.block<3, 3>(6, 6)=RiR0.transpose();

          r_ALL.block<3, 1>(9 * (i - 1), 0) = R_error;
          r_ALL.block<3, 1>(9 * (i - 1)+3, 0) = V_error;
          r_ALL.block<3, 1>(9 * (i - 1)+6, 0) = P_error;
          
          v_temp_j.setZero();
          v_temp_j.block<3, 3>(3, 0)=-RiR0.transpose();
          v_temp_j.block<3, 3>(3, 3)=RiR0.transpose();
          v_temp_j.block<3, 3>(6, 0)=-RiR0.transpose()*del_time;


          MatrixXd cov_temp=frame_i->imu_res._cov_rvp;
          cov_temp=cov_temp.inverse();
          //协方差稀疏矩阵
          for(int i2=0;i2<9;i2++)
          {
            for(int i3=0;i3<9;i3++)
            {
              cov_imu_all_inverse.push_back(Eigen::Triplet<double>( 9*(i-1)+i2,9*(i-1)+i3 ,cov_temp(i2,i3)));
            }
          }

          //稀疏矩阵赋值
          for(int i2=0;i2<9;i2++)
          {
            for(int i3=0;i3<12;i3++)
            {
              //通过这种方式可以大大加快速度
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,6*(i-1)+i3,tr_temp_j(i2,i3)));
            }
            for(int i3=0;i3<6;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,6*optNum+3*(i-1)+i3,v_temp_j(i2,i3))); 
            }
          }
          
          //---------------------------------------------bias
          //                        0     3     6     9
          //                        bg1   ba1   bg2   ba2
          //               0  dr    o    
          //               3  dv    o     o
          //               6  dp    o     o
          //               9  d_bg  o            o
          //               12 d_ba        o            o

          Eigen::Matrix3d Epb = Sophus::SO3d::exp(R_error).matrix();

          bias_temp_j.setZero();

          pthread_mutex_lock (&mutex_pose);
          // bg1
          bias_temp_j.block<3, 3>(0, 0) =-jrinv * Epb.transpose() * JacobianR(frame_i->imu_res._J_R_Biasg * frame_i->_dBias_g) *frame_i->imu_res._J_R_Biasg;
          bias_temp_j.block<3, 3>(3, 0) = -frame_i->imu_res._J_V_Biasg;
          bias_temp_j.block<3, 3>(6, 0) = -frame_i->imu_res._J_P_Biasg;

          // ba1
          bias_temp_j.block<3, 3>(3, 3) = -frame_i->imu_res._J_V_Biasa;
          bias_temp_j.block<3, 3>(6, 3) = -frame_i->imu_res._J_P_Biasa;
    
          //bg
          bias_temp_j.block<3, 3>(9, 0) = -G_parameter.xishu_imu_bias_change*danwei;
          bias_temp_j.block<3, 3>(9, 6) = G_parameter.xishu_imu_bias_change*danwei;
          //ba
          bias_temp_j.block<3, 3>(12, 3) = -G_parameter.xishu_imu_bias_change*danwei;
          bias_temp_j.block<3, 3>(12, 9) = G_parameter.xishu_imu_bias_change*danwei;


          r_ALL.block<3, 1>(9*(optNum-1)+6*(i-1), 0) = G_parameter.xishu_imu_bias_change*((frame_j->_BiasGyr+frame_j->_dBias_g)-(frame_i->_BiasGyr+frame_i->_dBias_g));
          r_ALL.block<3, 1>(9*(optNum-1)+6*(i-1)+3, 0)= G_parameter.xishu_imu_bias_change*((frame_j->_BiasAcc+frame_j->_dBias_a)-(frame_i->_BiasAcc+frame_i->_dBias_a));

          pthread_mutex_unlock(&mutex_pose);

          Matrix<double,6,6> temp_bias_c=del_time*cov_bias_instability;
          temp_bias_c=temp_bias_c.inverse();

          for(int i2=0;i2<6;i2++)
          {
            for(int i3=0;i3<6;i3++)
            {
              cov_imu_all_inverse.push_back(Eigen::Triplet<double>( 9*(optNum-1)+6*(i-1)+i2,9*(optNum-1)+6*(i-1)+i3 ,temp_bias_c(i2,i3)));
            }
          }

          //稀疏矩阵赋值  零偏
          for(int i2=0;i2<9;i2++)
          {            
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,9*optNum+6*(i-1)+i3,bias_temp_j(i2,i3))); 
            }
          }
          //            零偏变化
          for(int i2=0;i2<6;i2++)
          {            
            for(int i3=0;i3<12;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(optNum-1)+6*(i-1)+i2,9*optNum+6*(i-1)+i3,bias_temp_j(9+i2,i3))); 
            }
          }

          //-----------------------------------------------------R P gravity
          //                      0   3   6
          //                      R0  P0  R_g Eigen::MatrixXd bias_temp_j(15, 12);
          //               0 dr   o   
          //               3 dv   o       o
          //               6 dp   o   o   o

          pthread_mutex_lock (&mutex_g_R_T);


          g_R_P_temp_j.setZero();

          // R0:
          g_R_P_temp_j.block<3, 3>(0, 0) = jrinv * (-RjR0.transpose() * RiR0+danwei);
          g_R_P_temp_j.block<3, 3>(3, 0) = Sophus::SO3d::hat(RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time));
          g_R_P_temp_j.block<3, 3>(6, 0) =
          Sophus::SO3d::hat( RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 *rota_gravity*initial_gravity * del_time * del_time));

          // P0:
          g_R_P_temp_j.block<3, 3>(6, 3) = RiR0.transpose() *(R_j-R_i);

          if(G_parameter.gravity_opti_method==0)
          {
            // R_g:
            g_R_P_temp_j.block<3, 3>(3, 6) = RiR0.transpose()*del_time*rota_gravity*Sophus::SO3d::hat(initial_gravity);
            g_R_P_temp_j.block<3, 3>(6, 6) = 0.5*RiR0.transpose()*del_time*del_time*rota_gravity*Sophus::SO3d::hat(initial_gravity);

          }
          else if(G_parameter.gravity_opti_method==1)
          {
            g_R_P_temp_j.block<3, 3>(3, 6) = -RiR0.transpose()*del_time;
            g_R_P_temp_j.block<3, 3>(6, 6) = -0.5*RiR0.transpose()*del_time*del_time;
          }

          pthread_mutex_unlock(&mutex_g_R_T);
          // via_all.via_g_R_P.J.block<9, 9>(9 * (i - 1), 0) = g_R_P_temp_j;

          //稀疏矩阵赋值
          for(int i2=0;i2<9;i2++)
          {            
            for(int i3=0;i3<9;i3++)
            {
              J_sparce_15N_9.push_back(Eigen::Triplet<double>(9*(i-1)+i2,15*optNum+i3,g_R_P_temp_j(i2,i3))); 
            }
          }

          //---------------------numerical value to verify the Jacobian  
          // check the past code
        }
        double time_2= (double)cv::getTickCount();

        Eigen::MatrixXd delta_imu_vis(9 * optNum,1);
        Eigen::MatrixXd delta_bias(6 * optNum,1);
        Eigen::MatrixXd delta_g_R_P(9,1);

        delta_imu_vis.setZero();
        delta_bias.setZero();
        delta_g_R_P.setZero();

        // v r t bg ba g R T
        if(keyframes.size()>G_parameter.sliding_window_length)
        {
            Eigen::SparseMatrix<double> COV_all_INV(15 * optNum-15, 15 * optNum-15);
            COV_all_INV.setFromTriplets(cov_imu_all_inverse.begin(), cov_imu_all_inverse.end());

            Eigen::SparseMatrix<double> JTJ_fastgo_15n9(15 * optNum+9, 15 * optNum+9);
            JTJ_fastgo_15n9.setFromTriplets(coeff.begin(), coeff.end());

            //验证得到了正确的稀疏矩阵
            Eigen::SparseMatrix<double> J_s_15n9(15*optNum-15, 15 * optNum+9);
            J_s_15n9.setZero();
            J_s_15n9.setFromTriplets(J_sparce_15N_9.begin(), J_sparce_15N_9.end());    
            
            Eigen::SparseMatrix<double> r_s_15n9(15*optNum-15,1);
            matrix_dense_to_sparse(r_s_15n9,r_ALL);

            Eigen::SparseMatrix<double> JTJ_s_15n9;
            Eigen::SparseMatrix<double> JTr_s_15n9;

            if(G_parameter.use_cov)
            {
              JTJ_s_15n9=J_s_15n9.transpose()*COV_all_INV*J_s_15n9;
              JTr_s_15n9=J_s_15n9.transpose()*COV_all_INV*r_s_15n9;
            }
            else
            {
              JTJ_s_15n9=J_s_15n9.transpose()*J_s_15n9;
              JTr_s_15n9=J_s_15n9.transpose()*r_s_15n9;
            }

            JTJ_s_15n9=JTJ_s_15n9+G_parameter.xishu_visual*JTJ_fastgo_15n9;

            Eigen::MatrixXd JTr_d_15n9(15 * optNum+9,1);
            matrix_sparse_to_dense(JTr_d_15n9,JTr_s_15n9);
            JTr_d_15n9.block(0,0,6*optNum,1)=JTr_d_15n9.block(0,0,6*optNum,1)+G_parameter.xishu_visual*JTe;

            // for(int i1=0;i1<6*optNum;i1++)
            // {
            //   JTr_s_15n9.coeffRef(i1,0)+=G_parameter.xishu_visual*JTe(i1,0);
            // }

            Eigen::MatrixXd delta_15n9(15 * optNum+9,1);
            Eigen::SimplicialLDLT	<Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> > sldlt;
            sldlt.compute(JTJ_s_15n9);
            delta_15n9 = sldlt.solve(-JTr_d_15n9);

            delta_imu_vis =delta_15n9.block(0,0,9*optNum,1);
            delta_bias =delta_15n9.block(9*optNum,0,6*optNum,1);
            // via_all.via_r_p_g.delta.setZero();
            delta_g_R_P=delta_15n9.block(15*optNum,0,9,1);
        }

        double time_3= (double)cv::getTickCount();

        if(G_parameter.out_residual)
        {
          Eigen::MatrixXd r_temp(3*optNum,1);
          Eigen::MatrixXd t_temp(3*optNum,1);
          Eigen::MatrixXd v_temp(3*optNum,1);
          Eigen::MatrixXd bg_temp(3*optNum,1);
          Eigen::MatrixXd ba_temp(3*optNum,1);
          for(int i=0;i<optNum;i++)
          {
              r_temp.block(3*i,0,3,1)=delta_imu_vis.block(6*i,0,3,1);
              t_temp.block(3*i,0,3,1)=delta_imu_vis.block(6*i+3,0,3,1);
              v_temp.block(3*i,0,3,1)=delta_imu_vis.block(6*optNum+3*i,0,3,1);
              bg_temp.block(3*i,0,3,1)=delta_bias.block(6*i,0,3,1);
              ba_temp.block(3*i,0,3,1)=delta_bias.block(6*i+3,0,3,1);
          }
          
          int count_duan=keyframes_new.size();

          pthread_mutex_lock (&mutex_pose);
          visual_local_error[iter] = reprojection_error_3Dto3D(optimized_fc);
          visual_global_error[iter] = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);
          pthread_mutex_unlock(&mutex_pose);

          bg_variation[iter]=bg_temp.norm();
          ba_variation[iter]=ba_temp.norm();

          bg_local_variation[iter]=bg_temp.block(0,0,3*(optNum-count_duan),1).norm();
          ba_local_variation[iter]=ba_temp.block(0,0,3*(optNum-count_duan),1).norm();
       
          r_variation[iter]=r_temp.norm();
          t_variation[iter]=t_temp.norm();
          v_variation[iter]=v_temp.norm();

          r_variation_local[iter]=r_temp.block(0,0,3*(optNum-count_duan),1).norm();
          t_variation_local[iter]=t_temp.block(0,0,3*(optNum-count_duan),1).norm();
          v_variation_local[iter]=v_temp.block(0,0,3*(optNum-count_duan),1).norm();


          error_imu[iter]=        r_ALL.block(0,0,9 *(optNum-1),1).norm();
          error_imu_local[iter]=  r_ALL.block(0, 0,9 * (optNum-count_duan-1),1).norm();

          error_imu_with_bias[iter]=        r_ALL.norm();

          // cout<<15*(optNum-count_duan-1)<<endl;
          // cout<<optNum<<endl;
          // cout<<count_duan<<endl;
          // exit(1);

          Eigen::MatrixXd error_temp(15*(optNum-count_duan-1),1);
       
          error_temp.block(0,0,9*(optNum-count_duan-1),1)=r_ALL.block(0,0,9*(optNum-count_duan-1),1);
        
          error_temp.block(9*(optNum-count_duan-1),0,6*(optNum-count_duan-1),1)=r_ALL.block(9*(optNum-1),0,6*(optNum-count_duan-1),1);
        
          error_imu_local_with_bias[iter]= error_temp.norm();

        }
       

        pthread_mutex_lock (&mutex_pose);
        for (int i = 1; i < keyframes.size(); i++)
        {
          Eigen::Vector3d delta_v;
          Eigen::Vector3d delta_r;
          Eigen::Vector3d delta_t;

          delta_t  =  delta_imu_vis.block<3, 1>(6 * (i - 1), 0);
          delta_r  =  delta_imu_vis.block<3, 1>(6 * (i - 1)+3, 0);
          delta_v  =  delta_imu_vis.block<3, 1>(6*optNum+3*(i-1), 0);

          F[keyframes[i]]._V += G_parameter.xishu_V * delta_v;

          Eigen::Matrix3d r1 = F[keyframes[i]].pose_sophus[0].matrix().block<3, 3>(0, 0);
          Eigen::Vector3d t1 = F[keyframes[i]].pose_sophus[0].matrix().block<3, 1>(0, 3);

          Sophus::SO3d Rd = Sophus::SO3d::exp(G_parameter.xishu_R*delta_r);
    
          Eigen::Matrix3d r2;
          Eigen::Vector3d t2;
          r2 = Rd.matrix()*r1;
          t2 = t1 + G_parameter.xishu_T * delta_t;

          set_orthogonal(r2);

          Sophus::SE3d SE3_Rt(r2, t2);    
          F[keyframes[i]].pose_sophus[0] = SE3_Rt;

          Eigen::Vector3d delta_bg_d = delta_bias.block<3, 1>(6 * (i - 1), 0);
          Eigen::Vector3d delta_ba_d = delta_bias.block<3, 1>(6 * (i - 1)+3, 0);

          F[keyframes[i]]._dBias_g += G_parameter.xishu_bg_d * delta_bg_d;       
          F[keyframes[i]]._dBias_a += G_parameter.xishu_ba_d * delta_ba_d;

          //output the bias of a keyframe
          // if(G_parameter.out_bias==1&& i == (keyframes.size()-2))

          if(G_parameter.out_bias==1)
          {
            cout<<i<<" bias:bg ba  projection:"<<endl;
            cout<<F[keyframes[i]]._dBias_g.transpose()+F[keyframes[i]]._BiasGyr.transpose()<<endl;
            cout<<F[keyframes[i]]._dBias_a.transpose()+F[keyframes[i]]._BiasAcc.transpose()<<endl;  
          }
        }
        pthread_mutex_unlock(&mutex_pose);

        pthread_mutex_lock (&mutex_g_R_T);


        if(G_parameter.gravity_opti_method==0)
        {
          Sophus::SO3d G_ro=Sophus::SO3d::exp(G_parameter.xishu_gravity*delta_g_R_P.block<3, 1>(6, 0));
          rota_gravity=rota_gravity*G_ro.matrix();
          set_orthogonal(rota_gravity);
        }
        else if(G_parameter.gravity_opti_method==1)
        {
          initial_gravity=initial_gravity+G_parameter.xishu_gravity*delta_g_R_P.block<3, 1>(6, 0);
          initial_gravity=G_parameter.gravity_norm*initial_gravity.normalized();
        }

        //update the transformation matrix
        Sophus::SO3d Rd0 = Sophus::SO3d::exp(G_parameter.xishu_rote*delta_g_R_P.block<3, 1>(0, 0));
        imu_to_cam_rota=imu_to_cam_rota*Rd0.matrix();
        set_orthogonal(imu_to_cam_rota);
        imu_to_cam_trans=imu_to_cam_trans+G_parameter.xishu_trans*delta_g_R_P.block<3, 1>(3, 0);
        pthread_mutex_unlock(&mutex_g_R_T);

        // out the transformation matrix 
        if(G_parameter.out_transformation)  
        {
          // cout<<"transformation matrix rotation translation gravity："<<endl;
          Sophus::SO3d R_ini(ini_imu_to_cam_rota);
          Eigen::Vector3d r_ini=R_ini.log();

          Sophus::SO3d R_now(imu_to_cam_rota);
          Eigen::Vector3d r_new=R_now.log();
          // cout<<"cam-imu rotation: "<<r_new.transpose()<<endl;
          // cout<<"cam-imu trans: "<<imu_to_cam_trans.transpose()<<endl;

          // record_vector(r_new[0],125,"x");
          // record_vector(r_new[1],125,"y");
          // record_vector(r_new[2],125,"z");

          // record_vector(imu_to_cam_trans[0],120,"tx");
          // record_vector(imu_to_cam_trans[1],120,"ty");
          // record_vector(imu_to_cam_trans[2],120,"tz");


          double angle_r_ini_r=PLANE::angle_cal(r_new,r_ini);

          Vector3d g_now=rota_gravity*initial_gravity;
          cout<<" cam-imu旋转向量变化的角度: "<<angle_r_ini_r<<"  gravity: "<<g_now.transpose()<<endl;
          // <<" "<<initial_gravity.transpose()<<endl;
        }
        
          // double time_01=(time_1-time_0)*1000/ cv::getTickFrequency();
          // double time_12=(time_2-time_1)*1000/ cv::getTickFrequency();
          // double time_23=(time_3-time_2)*1000/ cv::getTickFrequency();
          // double time_34=(time_4-time_3)*1000/ cv::getTickFrequency();
          // double time_04=(time_4-time_0)*1000/ cv::getTickFrequency();
          // cout<<"01 time: "<<time_01<<endl;
          // cout<<"12 time: "<<time_12<<endl;
          // cout<<"23 time: "<<time_23<<endl;
          // cout<<"34 time: "<<time_34<<endl;
          // cout<<"04 time: "<<time_04<<endl;

      } //flag 3,imu optimization

      if(iter==1)
      {
        int wrong_loop=0;
        // visual judge
        if( ((visual_local_error[1] - visual_local_error[0]) / visual_local_error[0]>1) )
        { 
          wrong_loop=1;
        }

        // imu local error judge
        if(G_parameter.flag_youhua==3)
        {

          // record_vector(error_imu_local[1],120,"localerror");
          // record_vector(bg_variation[0],120,"bg");
          // record_vector(ba_variation[0],120,"ba");
          // record_vector(bg_local_variation[0],120,"bglocal");
          // record_vector(ba_local_variation[0],120,"balocal");
     
          if(((visual_local_error[1] - visual_local_error[0]) / visual_local_error[0]>0.2) && ( (error_imu_local[1]-error_imu_local[0])/error_imu_local[0] )>1 )
          {
            wrong_loop=1;
          } 
          //根据零偏变化
          // if(ba_variation[0]>0.4 ||bg_variation[0]>0.2)
          if(bg_variation[0]>0.1)
          {
            wrong_loop=1;
          } 
        }
        cout<<"loop detector: "<<((visual_local_error[1] - visual_local_error[0]) / visual_local_error[0]);
        cout<<"  "<<((error_imu_local[1]-error_imu_local[0])/error_imu_local[0]) ;
        cout<<"  "<<ba_variation[0]<<"  "<<bg_variation[0]<<endl;

        if(G_parameter.drop_wrong_loop==0)
        {
          wrong_loop=0;
        }

        if(wrong_loop)
        { 
          if(revelent==0)
          {
            revelent=G_parameter.drop_wrong_loop_relevant;
          }

          cout<<"产生了错误的回环"<<endl;
          cout<<"视觉error判断： "<<(visual_local_error[1] - visual_local_error[0]) / visual_local_error[0]<<endl;
          cout<<"imu local error判断： "<<(error_imu_local[1]-error_imu_local[0])/error_imu_local[0]<<endl;

          // Mat ting=Mat(10,10,CV_16UC1);
          // cv::imshow("a",ting);
          // cv::waitKey(0);

          //新的段中的帧， 保留连接前几个关键帧的帧对,去掉前面的回环    问题是如果是跟踪失败了,然后回环的定位   通过imulocality来判断
          for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
          {
              Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
              Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

              int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
              int frame_new_pos = getKeyFramePos[frame_new.frame_index];

              // 新来的段中的帧
              if(frame_new.frame_index>(count_last_global_opti-1))
              {
                if(fabs(frame_new_pos - frame_ref_pos) > G_parameter.drop_corres_length)
                {
                    fCList[keyframe_candidate_fcorrs[i]].reset();
                }    
              }
          }
          // 计算去掉部分帧对之后的视觉error
          // visual_local_error[G_parameter.GN_number] = reprojection_error_3Dto3D(optimized_fc);
          // visual_global_error[G_parameter.GN_number] = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);
        
          //  回到原状态
          pthread_mutex_lock(&mutex_pose);
          for(int i = 0; i < N; i++)
          {
              F[i].pose_sophus[0] = frame_poses[i];

              // cout<<F[i].pose_sophus[0].matrix()<<endl;

              F[i]._V = sudu[i];
              F[i]._dBias_g = db_g[i];
              F[i]._dBias_a = db_a[i];  
          }
          pthread_mutex_unlock(&mutex_pose);
      
          
          pthread_mutex_lock (&mutex_g_R_T);
          imu_to_cam_rota= old_rotation;
          imu_to_cam_trans=old_transformation;
          rota_gravity=old_rota_gravity;
          pthread_mutex_unlock(&mutex_g_R_T);
        
        }
      }
    }  //GN optimization
    
    double consume_time_gn= (double)cv::getTickCount() - st_time_gn;
    consume_time_gn=consume_time_gn*1000/ cv::getTickFrequency();
    // cout<<"GN time3:"<<consume_time_gn<<endl;

    // update local frame
    pthread_mutex_lock (&mutex_pose);
    for (int i = 0; i < kflist.size(); i++)
    {
      for (int j = 0; j < kflist[i].corresponding_frames.size(); j++)
      {
          // local_pose  =keypose * relative_from_key_to_current;
        F[kflist[i].corresponding_frames[j]].pose_sophus[0] = F[kflist[i].keyFrameIndex].pose_sophus[0] *
                kflist[i].relative_pose_from_key_to_current[j];
      }
    }
    pthread_mutex_unlock(&mutex_pose);

    // if(G_parameter.out_residual)
    // {
    //   Eigen::MatrixXd  error_gn(9*optNum-9,1);
    //   Eigen::MatrixXd  error_gn_biaschange(15*optNum-15,1);
    //   // Eigen::MatrixXd  error_gn_right_bias(9*optNum-9,1);  //使用正确的bias，看是否因为error对bias不敏感
    //   for (int i = 1; i < keyframes.size() - 1; i++)  
    //   {

    //     Frame *frame_i, *frame_j;
    //     Eigen::Vector3d r_i;
    //     Eigen::Vector3d r_j;
    //     Eigen::Matrix3d R_i;
    //     Eigen::Matrix3d R_j;

    //     Eigen::Vector3d V_i;
    //     Eigen::Vector3d V_j;

    //     Eigen::Vector3d P_i;
    //     Eigen::Vector3d P_j;

    //     //运动误差
    //     Eigen::Vector3d R_error;
    //     Eigen::Vector3d V_error;
    //     Eigen::Vector3d P_error;
    //     double del_time;

    //     frame_i = &F[keyframes[i]];
    //     frame_j = &F[keyframes[i + 1]];

    //     pthread_mutex_lock (&mutex_pose);
    //     //得到旋转向量和旋转矩阵
    //     r_i = frame_i->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
    //     r_j = frame_j->pose_sophus[0].log().matrix().block<3, 1>(3, 0);
    //     R_i = frame_i->pose_sophus[0].matrix().block<3, 3>(0, 0);
    //     R_j = frame_j->pose_sophus[0].matrix().block<3, 3>(0, 0);

    //     V_i = frame_i->_V;
    //     V_j = frame_j->_V;

    //     P_i = (frame_i->pose_sophus[0].matrix().block<3, 1>(0, 3));
    //     P_j = (frame_j->pose_sophus[0].matrix().block<3, 1>(0, 3));
    //     pthread_mutex_unlock(&mutex_pose);

    //     del_time = frame_i->imu_res._delta_time;
       
    //     //通过比较积分段的序号,可以保证积分段正确
    //     if((frame_j->time_stamp-frame_i->time_stamp)!=del_time
    //         ||frame_i->imu_res.frame_index_qian!=frame_i->frame_index
    //         ||frame_i->imu_res.frame_index_hou!=frame_j->frame_index)
    //     {
    //         cout<<"积分段错误"<<endl;
    //         cout<<endl;
    //         // cout<<F[294].imu_res._delta_time<<endl;

    //         cout<<frame_i->frame_index<<endl;
    //         cout<<frame_i->is_keyframe<<endl;
    //         cout<<frame_i->tracking_success<<endl;
    //         cout<<frame_i->origin_index<<endl<<endl;

    //         cout<<frame_j->frame_index<<endl;
    //         cout<<frame_j->is_keyframe<<endl;
    //         cout<<frame_j->tracking_success<<endl;
    //         cout<<frame_j->origin_index<<endl<<endl;

    //         cout<<frame_j->time_stamp-frame_i->time_stamp<<endl;
    //         cout<<del_time<<endl;
    //         //这里的时间相等,说明imu数据的积分没有问题
    //         cout<<"integration time not equal!";
    //         exit(1);
    //     }
    //     Eigen::Matrix3d RiR0=R_i*imu_to_cam_rota;
    //     Eigen::Matrix3d RjR0=R_j*imu_to_cam_rota;
    //     Eigen::Vector3d PiP0=P_i+R_i*imu_to_cam_trans;
    //     Eigen::Vector3d PjP0=P_j+R_j*imu_to_cam_trans;

    //     //计算residual errors
    //     Eigen::Matrix3d Eb = Sophus::SO3d::exp(frame_i->imu_res._J_R_Biasg * frame_i->_dBias_g).matrix();
    //     Eigen::Matrix3d temp2 = frame_i->imu_res._delta_R * Eb;
    //     Eigen::Matrix3d temp = temp2.transpose() * RiR0.transpose() * RjR0;

    //     Sophus::SO3d SO3_R(temp);
    //     R_error = SO3_R.log();

    //     V_error = RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time)
    //               - (frame_i->imu_res._delta_V + frame_i->imu_res._J_V_Biasg * frame_i->_dBias_g
    //                   + frame_i->imu_res._J_V_Biasa * frame_i->_dBias_a);

    //     P_error = RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 * rota_gravity*initial_gravity * del_time * del_time)
    //               - (frame_i->imu_res._delta_P + frame_i->imu_res._J_P_Biasg * frame_i->_dBias_g
    //                   + frame_i->imu_res._J_P_Biasa * frame_i->_dBias_a);

    //     //使用正确的bias，看是否因为error对bias不敏感
    //     // Eigen::Vector3d R_error_bias;
    //     // Eigen::Vector3d V_error_bias;
    //     // Eigen::Vector3d P_error_bias;

    //     // Eigen::Vector3d bias_a(0.1,-0.1,0.1);
    //     // R_error_bias = R_error;

    //     // V_error_bias = RiR0.transpose() * (V_j - V_i - rota_gravity*initial_gravity * del_time)
    //     //           - (frame_i->imu_res._delta_V + frame_i->imu_res._J_V_Biasg * frame_i->_dBias_g
    //     //               + frame_i->imu_res._J_V_Biasa * bias_a);

    //     // P_error_bias = RiR0.transpose() * (PjP0 - PiP0 - V_i * del_time - 0.5 * rota_gravity*initial_gravity * del_time * del_time)
    //     //           - (frame_i->imu_res._delta_P + frame_i->imu_res._J_P_Biasg * frame_i->_dBias_g
    //     //               + frame_i->imu_res._J_P_Biasa * bias_a);

    //     error_gn.block<3, 1>(9*i-9, 0)   = R_error;
    //     error_gn.block<3, 1>(9*i-9+3, 0) = V_error;
    //     error_gn.block<3, 1>(9*i-9+6, 0) = P_error;

    //     error_gn_biaschange.block<3, 1>(15*i-15, 0) = R_error;
    //     error_gn_biaschange.block<3, 1>(15*i-15+3, 0) = V_error;
    //     error_gn_biaschange.block<3, 1>(15*i-15+6, 0) = P_error;
    //     error_gn_biaschange.block<3, 1>(15*i-15+9, 0)= (frame_j->_BiasGyr+frame_j->_dBias_g)-(frame_i->_BiasGyr+frame_i->_dBias_g);
    //     error_gn_biaschange.block<3, 1>(15*i-15+12, 0)= (frame_j->_BiasAcc+frame_j->_dBias_a)-(frame_i->_BiasAcc+frame_i->_dBias_a);

    //   }
    //   int count_duan=keyframes_new.size();
    //   error_imu[G_parameter.GN_number]                        =error_gn.norm();
    //   error_imu_local[G_parameter.GN_number]                  =error_gn.block(0, 0,9 * (optNum-count_duan-1),1).norm();
    //   error_imu_with_bias[G_parameter.GN_number]              =error_gn_biaschange.norm();
    //   error_imu_local_with_bias[G_parameter.GN_number]        =error_gn_biaschange.block(0, 0,15* (optNum-count_duan-1),1).norm();

    //   pthread_mutex_lock (&mutex_pose);
    //   visual_local_error[G_parameter.GN_number] = reprojection_error_3Dto3D(optimized_fc);
    //   visual_global_error[G_parameter.GN_number] = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);
    //   pthread_mutex_unlock(&mutex_pose);

    //   // cout<<"正确bias得到的error：  "<<error_gn_right_bias.norm()<<endl;
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  bias variation :"<<imu_bias_variation[iter]<<endl;
    //   // }
    //   // cout<<"global optimization:"<<endl;
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  r variation :"<<r_variation[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  t variation :"<<t_variation[iter]<<endl;
    //   // }      
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  v variation :"<<v_variation[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  bg variation :"<<bg_variation[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  ba variation :"<<ba_variation[iter]<<endl;
    //   // }
 

    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error :"<<error_imu[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error local :"<<error_imu_local[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error  with bias:"<<error_imu_with_bias[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"  imu error local with bias:"<<error_imu_local_with_bias[iter]<<endl;
    //   // }

    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"visual local error:"<<visual_local_error[iter]<<endl;
    //   // }
    //   // for(int iter=0;iter<G_parameter.GN_number+1;iter++)
    //   // {
    //   //   cout<<"GN:"<<iter<<"visual global error::"<<visual_global_error[iter]<<endl;
    //   // }

    //   // if the wrong loop happens,the total error decreases,
    //   // while the error except of the frame correspondings of the last keyframe will increase.


    // } 


    
    //if the bias of imu change too much,make the repeated integration to invent the  numerical error
    pthread_mutex_lock (&mutex_pose);
    for (int i = 1; i < keyframes.size()-1; i++)
    {
      if(F[keyframes[i]]._dBias_a.norm()>0.2||F[keyframes[i]]._dBias_g.norm()>0.1)
      {
        // cout<<endl<<endl;
        // cout<<"keyframe "<<i<<" make the repeated integration"<<endl;
        // cout<<F[keyframes[i]]._BiasAcc<<endl;
        // cout<<F[keyframes[i]]._dBias_a<<endl;
        // cout<<endl<<endl;

        F[keyframes[i]]._BiasGyr=F[keyframes[i]]._BiasGyr+F[keyframes[i]]._dBias_g;
        F[keyframes[i]]._BiasAcc=F[keyframes[i]]._BiasAcc+F[keyframes[i]]._dBias_a;
        F[keyframes[i]]._dBias_g.setZero();
        F[keyframes[i]]._dBias_a.setZero();

        IMUPreintegrator IMUPreInt;
        IMUPreInt.imu_index_qian=F[keyframes[i]].imu_res.imu_index_qian;
        IMUPreInt.imu_index_hou=F[keyframes[i]].imu_res.imu_index_hou;
        IMUPreInt.frame_index_qian=F[keyframes[i]].imu_res.frame_index_qian;
        IMUPreInt.frame_index_hou=F[keyframes[i]].imu_res.frame_index_hou;

        Vector3d _bg= F[keyframes[i]]._BiasGyr;
        Vector3d _ba= F[keyframes[i]]._BiasAcc;

        pthread_mutex_lock (&mutex_imu);
        double last_imu_time=F[keyframes[i]].time_stamp;
        int count_imu11=IMUPreInt.imu_index_qian;
        while(IMU_data_raw[count_imu11].time_stamp<F[keyframes[i+1]].time_stamp)
        {
          double dt = IMU_data_raw[count_imu11].time_stamp - last_imu_time;
          Vector3d g_=(IMU_data_raw[count_imu11]._g +IMU_data_raw[count_imu11-1]._g )/2;
          Vector3d a_=(IMU_data_raw[count_imu11]._a +IMU_data_raw[count_imu11-1]._a )/2;
          IMUPreInt.update(g_ - _bg, a_ - _ba, dt);
          last_imu_time=IMU_data_raw[count_imu11].time_stamp;
          count_imu11++;
        }
        if(count_imu11!=IMUPreInt.imu_index_hou)
        {
          cout<<"repeated integration error"<<endl;
          exit(1);
        }
        Vector3d g_=(IMU_data_raw[count_imu11]._g +IMU_data_raw[count_imu11-1]._g )/2;
        Vector3d a_=(IMU_data_raw[count_imu11]._a +IMU_data_raw[count_imu11-1]._a )/2;
        IMUPreInt.update(g_ - _bg,a_ - _ba, F[keyframes[i+1]].time_stamp-IMU_data_raw[count_imu11-1].time_stamp);
        pthread_mutex_unlock (&mutex_imu);

        F[keyframes[i]].imu_res=IMUPreInt;
      }
    }
    double time_4= (double)cv::getTickCount();
    pthread_mutex_unlock(&mutex_pose);


    if(exit_flag_)
    {
      exit(1);
    }

    return 1;
  }

}



