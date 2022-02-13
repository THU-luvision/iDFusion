cmake_minimum_required(VERSION 3.2)
project(MILD)

#set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)


set(OpenCV_DIR "/home/computer/Documents/library/opencv-3.3.0/build")
set(OPENNI2_LIBRARY "/home/luvision/library/OpenNI-Linux-x64-2.2_obzg0/Redist/libOpenNI2.so")
set(OPENNI2_INCLUDE_DIR "/home/luvision/library/OpenNI-Linux-x64-2.2_obzg0/Include")



MESSAGE( STATUS "Remember to check Opencv dir: " ${OpenCV_DIR})
MESSAGE( STATUS "Remember to check OpenNI dir: " ${OPENNI2_LIBRARY})


set(GCFUSION_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")
message("${CMAKE_MODULE_PATH}")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  rospy
  std_msgs
)


find_package(OpenCV REQUIRED)
find_package(OpenGL)
find_package(GLUT)
find_package( PkgConfig )
find_package(Chisel REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(Boost COMPONENTS system thread  filesystem REQUIRED)

MESSAGE( STATUS "Remember to check SOPHUS_INCLUDE_DIR dir: " ${SOPHUS_INCLUDE_DIR})


find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()
MESSAGE( STATUS "Realsense dir: " ${ADDITIONAL_INCLUDE_DIRS})
MESSAGE( STATUS "Realsense lib: " ${REALSENSE2_FOUND})


file(GLOB helloworld_SRC
"*.cpp"
"*.h"
"Tools/*.h"
"Tools/*.cpp"
"GCSLAM/*.cpp"
"GCSLAM/*.h"
"GCSLAM/MILD/*.cpp"
"GCSLAM/MILD/*.h"
"GCSLAM/ORBSLAM/*.cpp"
"GCSLAM/ORBSLAM/*.h"
"GCFusion/*.cpp"
"GCFusion/*.h"
"Shaders/*.cpp"
"Shaders/*.h"
"GCSLAM/IMU/*.cpp"
"GCSLAM/IMU/*.h"
"GCSLAM/PLANE/*.cpp"
"GCSLAM/PLANE/*.h"
)

set(SOURCE_FILES
 ${helloworld_SRC})



set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -w")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
pkg_check_modules( EIGEN3 REQUIRED eigen3 )

include_directories(GCSLAM/peac)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS}  )
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories( ${SOPHUS_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories(${CHISEL_INCLUDE_DIR})
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${catkin_INCLUDE_DIRS})


  include_directories(${ADDITIONAL_INCLUDE_DIRS})
  add_definitions(-DWITH_REALSENSE)
  set(EXTRA_LIBS ${EXTRA_LIBS} ${REALSENSE2_FOUND})

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3  -msse4.1 -Wall -std=c++14 -DSHADER_DIR=${GCFUSION_SHADER_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -ftree-vectorize -lboost_system -lpthread -lboost_thread -lboost_filesystem")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")




add_executable(af ${SOURCE_FILES})
target_link_libraries(af ${OpenCV_LIBS} 
			${catkin_LIBRARIES}
			/opt/ros/kinetic/lib/libmessage_filters.so
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${CHISEL_LIBRARY}
                      ${OPENNI2_LIBRARY}
		      ${OPENGL_LIBRARIES} 
		      ${GLUT_LIBRARY}
                      ${EXTRA_LIBS}
		      ${pthread} 
		      ${Boost_LIBRARIES})








