#ifndef HEAD_H
#define HEAD_H

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
#include <ctime>
#include <pthread.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include "GCSLAM/frame.h"
#include "GCSLAM/MILD/loop_closure_detector.hpp"
#include "GCSLAM/MILD/BayesianFilter.hpp"
#include "GCSLAM/MultiViewGeometry.h"
#include "GCSLAM/GCSLAM.h"
#include "GCFusion/MapMaintain.hpp"
#include "GCFusion/MobileGUI.hpp"
#include "GCFusion/MobileFusion.h"
#include "BasicAPI.h"
#include "CHISEL/src/open_chisel/Chisel.h"
#include "CHISEL/src/open_chisel/ProjectionIntegrator.h"
#include "CHISEL/src/open_chisel/camera/PinholeCamera.h"
#include "CHISEL/src/open_chisel/Stopwatch.h"
#include "Tools/LogReader.h"
#include "Tools/LiveLogReader.h"
#include "Tools/RawLogReader.h"
#include "Tools/RealSenseInterface.h"
#include "GCSLAM/IMU/imudata.h"
#include "parameter.h"
#include "sparce_show.h"


using namespace std;
using namespace cv;


#endif // !HEAD_H
