cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_manipulation)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  pcl_conversions
  pointcloud_annotator
)

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(keypoints_detection src/keypoints_detection.cpp)
target_link_libraries(keypoints_detection ${catkin_LIBRARIES})
