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

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(keypoints_detection src/keypoints_detection.cpp)
target_link_libraries(keypoints_detection ${catkin_LIBRARIES})
