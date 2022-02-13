

#include<cmath>
#include "MobileFusion.h"
#include "MapMaintain.hpp"
using namespace std;

#define DEBUG_MODE 0
#define VERTEX_HASHING 1


// for range 1 only.


void GetMapDynamics(const std::vector<Frame> &frame_list,
                    std::vector<int> &keyframesToUpdate,
                    int corrKeyFrameIndex)

{
    keyframesToUpdate.clear();
    std::vector<int> keyframeIDList;
    std::vector<float> keyframeCostList;
    keyframeIDList.clear();
    keyframeCostList.clear();
    for(int i = 0; i < corrKeyFrameIndex ; i++)
    {
        if(frame_list[i].is_keyframe && frame_list[i].tracking_success && frame_list[i].origin_index == 0)
        {
            Eigen::Matrix4f prePose = frame_list[i].pose_sophus[1].matrix().cast<float>();
            Eigen::Matrix4f curPose = frame_list[i].pose_sophus[0].matrix().cast<float>();

            float cost = GetPoseDifference(prePose, curPose);
            keyframeIDList.push_back(i);
            keyframeCostList.push_back(cost);
//            cout << "frames: " << i << " " << cost << endl;
        }
    }


    // only consider dynamic map when keyframe number is larger than movingAveregaeLength
    // Deintegrate 10 keyframes at each time slot
    int movingAverageLength = 5;

    SelectLargestNValues(movingAverageLength,
                      keyframeIDList,
                      keyframeCostList,
                      keyframesToUpdate);
#if MobileCPU
    SelectLargestNValues(movingAverageLength,
                      keyframeIDList,
                      keyframeCostList,
                      keyframesToUpdate);
    SelectLargestNValues(movingAverageLength,
                      keyframeIDList,
                      keyframeCostList,
                      keyframesToUpdate);
#endif
    movingAverageLength = 2;
    SelectLargestNValues(movingAverageLength,
                      keyframeIDList,
                      keyframeCostList,
                      keyframesToUpdate);
    SelectLargestNValues(movingAverageLength,
                      keyframeIDList,
                      keyframeCostList,
                      keyframesToUpdate);
    if(keyframesToUpdate.size() < 5)
    {
        movingAverageLength = 1;
        SelectLargestNValues(movingAverageLength,
                          keyframeIDList,
                          keyframeCostList,
                          keyframesToUpdate);
        SelectLargestNValues(movingAverageLength,
                          keyframeIDList,
                          keyframeCostList,
                          keyframesToUpdate);
        SelectLargestNValues(movingAverageLength,
                          keyframeIDList,
                          keyframeCostList,
                          keyframesToUpdate);

    }

//    for(int i = 0; i < keyframesToUpdate.size();i++)
//    {
//        cout << "reintegrating keyframe: " << i << " " << keyframesToUpdate[i] << endl;
//    }
}


void MobileFusion::inter_all(std::vector<Frame> &frame_list, const std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist)
{
    double angular_thres=0.7;
    int  detect_count=4;

    for(int i =0 ; i < kflist.size(); i++)
    {
        // int frame_index = keyframesToUpdate[i];

        // if(frame_list[frame_index].imu_locality==1||frame_list[frame_index].bluriness<G_parameter.blur_threshold)continue; 
    
        int angular_flag=0;
        if(i>detect_count)
        {
            for(int k=0;k<detect_count;k++)
            {
                double an_norm=frame_list[kflist[i-k].keyFrameIndex].angular_V.norm();
                if(an_norm>angular_thres)
                {
                    angular_flag=1;
                    break;
                }
            }
        }

        if(angular_flag==0)
        {
            // TICK("CHISEL::Reintegration::Deintegration");
            // 参数0是先deintgration,把原来的结果去掉
            ReIntegrateKeyframe(frame_list,kflist[i],0);
            ReIntegrateKeyframe(frame_list,kflist[i],1);
            cout<<"in: "<<i<<endl;
        }
        else
        {
            cout<<"out: "<<i<<endl;
        }
        
        // TOCK("CHISEL::Reintegration::Deintegration");
        // cout << "chisel::finish deintegrate frame: " << frame_index  << " time: "
        //     << Stopwatch::getInstance().getTiming("CHISEL::Reintegration::Deintegration") << "ms" << endl;
    }

    chiselMap->UpdateMeshes(cameraModel);

    tsdf_vertice_num = chiselMap->GetFullMeshes(tsdf_visualization_buffer);
    vertex_data_updated =1;

}


void MobileFusion::IntegrateFrame(const Frame & frame_ref)
{
    int totalPixelNum =  cameraModel.GetWidth() * cameraModel.GetHeight();

    chisel::Transform lastPose;
    lastPose = frame_ref.pose_sophus[0].matrix().cast<float>();
    if(frame_ref.refined_depth.empty())
    {
        return;
    }
    float * depthImageData = (float *)frame_ref.refined_depth.data;
    unsigned char *colorImageData;
    if(frame_ref.rgb.empty())
    {
        colorImageData = NULL;
    }
    else
    {
        colorImageData = new unsigned char[totalPixelNum * 4];
        for(int j = 0; j < totalPixelNum ; j++)
        {
            colorImageData[j*4 + 0] = frame_ref.rgb.at<unsigned char>(j*3+0);
            colorImageData[j*4 + 1] = frame_ref.rgb.at<unsigned char>(j*3+1);
            colorImageData[j*4 + 2] = frame_ref.rgb.at<unsigned char>(j*3+2);
            colorImageData[j*4 + 3] = 1;
        }
    }

    if(frame_ref.tracking_success && frame_ref.origin_index == 0)
    {
        chiselMap->IntegrateDepthScanColor(projectionIntegrator,
                                           depthImageData,
                                           colorImageData,
                                           lastPose,
                                           cameraModel);
    }
    free(colorImageData);
}

int MobileFusion::tsdfFusion(std::vector<Frame> &frame_list,
                              int CorrKeyframeIndex,
                              const std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,
                              int integrateKeyframeID)
{

//    printf("begin refine keyframes %d %d\r\n",CorrKeyframeIndex,frame_list.size());
    Frame &frame_ref = frame_list[CorrKeyframeIndex];

    if(CorrKeyframeIndex == 0)
    {
        return 0;
    }

//    IntegrateFrameChunks(frame_ref);

    int integrateKeyframeIndex = kflist[integrateKeyframeID].keyFrameIndex;
    if(frame_ref.is_keyframe)
    {
        TICK("GetMapDynamics");
         // get dynamic info
        std::vector<int> keyframesToUpdate;
         GetMapDynamics(frame_list,
                        keyframesToUpdate,
                        integrateKeyframeIndex);


         // create look up tables for kflist
         std::vector<int> frameIndexToKeyframeDB(CorrKeyframeIndex + 1);
         for(int i = 0; i < CorrKeyframeIndex; i++)
         {
             frameIndexToKeyframeDB[i] = -1;
         }
         for(int i = 0; i < kflist.size();i++)
         {
             frameIndexToKeyframeDB[kflist[i].keyFrameIndex] = i;
         }
         TOCK("GetMapDynamics");

        double angular_thres=0.7;
        int  detect_count=4;

#if 1
        TICK("CHISEL::Reintegration::ReintegrateAll");
        for(int i =0 ; i < keyframesToUpdate.size(); i++)
        {
            int frame_index = keyframesToUpdate[i];

            // if(frame_list[frame_index].imu_locality==1||frame_list[frame_index].bluriness<G_parameter.blur_threshold)continue; 
            
            int angular_flag=0;

            for(int k=0;k<detect_count;k++)
            {
                int order_key=frameIndexToKeyframeDB[frame_index]-k;
                if(order_key>0)
                {
                    double an_norm=frame_list[kflist[order_key].keyFrameIndex].angular_V.norm();
                    if(an_norm>angular_thres)
                    {
                        angular_flag=1;
                        break;
                    }
                }
            }
            
            // cout<<"帧序号: "<<kflist[frameIndexToKeyframeDB[frame_index]].keyFrameIndex<<"  "<<frame_index<<"  "
            //       <<kflist[frameIndexToKeyframeDB[frame_index]-1].keyFrameIndex<<endl;
            if(angular_flag==0)
            {
                TICK("CHISEL::Reintegration::Deintegration");
                // 参数0是先deintgration,把原来的结果去掉
                ReIntegrateKeyframe(frame_list,kflist[frameIndexToKeyframeDB[frame_index]],0);
                ReIntegrateKeyframe(frame_list,kflist[frameIndexToKeyframeDB[frame_index]],1);
                TOCK("CHISEL::Reintegration::Deintegration");
            }
            // cout << "chisel::finish deintegrate frame: " << frame_index  << " time: "
            //     << Stopwatch::getInstance().getTiming("CHISEL::Reintegration::Deintegration") << "ms" << endl;
        }
        TOCK("CHISEL::Reintegration::ReintegrateAll");
#endif

         //begin update

         TICK("CHISEL::IntegrateKeyFrame");
         if(integrateKeyframeID >= 0)
         {
             Frame &kf = frame_list[kflist[integrateKeyframeID].keyFrameIndex];

         //    OptimizeKeyframeVoxelDomain(frame_list,kflist[integrateKeyframeID]);
             float fx = cameraModel.GetFx();
             float fy = cameraModel.GetFy();
             float cx = cameraModel.GetCx();
             float cy = cameraModel.GetCy();
             float width = cameraModel.GetWidth();
             float height = cameraModel.GetHeight();

            //  if(kf.tracking_success && kf.origin_index == 0 && kf.imu_locality==0 && kf.bluriness>G_parameter.blur_threshold)
             if(kf.tracking_success && kf.origin_index == 0 )
             {
                int angular_flag=0;
                for(int k=0;k<detect_count;k++)
                {
                    int order_key=integrateKeyframeID-k;
                    if(order_key>0)
                    {
                        double an_norm=frame_list[kflist[order_key].keyFrameIndex].angular_V.norm();
                        if(an_norm>angular_thres)
                        {
                            angular_flag=1;
                            break;
                        }
                    }
                }
                    
                if(angular_flag==0)
                {
                    ReIntegrateKeyframe(frame_list,kflist[integrateKeyframeID],1);
                }                
             }
         }
         TOCK("CHISEL::IntegrateKeyFrame");

         TICK("CHISEL_MESHING::UpdateMeshes");

         Eigen::Matrix4f cur_pos = frame_ref.pose_sophus[0].matrix().cast<float>();

         chiselMap->UpdateMeshes(cameraModel);
         TOCK("CHISEL_MESHING::UpdateMeshes");
//         cout << "begin to get full meshes" << endl;
         TICK("CHISEL_MESHING::GetFullMeshes");
         tsdf_vertice_num = chiselMap->GetFullMeshes(tsdf_visualization_buffer);
        //  cout<<"显存使用 总分配显存:"<<endl;
        //  cout<<tsdf_vertice_num<<"  "<<GLOBLA_MODLE_VERTEX_NUM<<endl;
//         cout << "valid vertex num: " << tsdf_vertice_num << endl;
         TOCK("CHISEL_MESHING::GetFullMeshes");
         vertex_data_updated =1;
//         cout << "finish to get full meshes: " << tsdf_vertice_num << endl;


         // this should be implemented at each keyframe update firstly.
         // update the given vertices based on the updated camera pose
         // now we try to recreate a tsdf map, based on the given vertices


#if 0
         cout << "meshes after update: " << chiselMap->GetChunkManager().GetAllMeshes().size() << endl;

         TICK("CHISEL::SaveMeshes");
         char fileName[256];
         memset(fileName,0,256);
         sprintf(fileName,"%d.ply",kflist.size());
         chiselMap->SaveAllMeshesToPLY(fileName);
         TOCK("CHISEL::SaveMeshes");
#endif
     }

    return 1;

}
