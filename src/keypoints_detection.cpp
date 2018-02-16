#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include "pointcloud_annotator/Update.h"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud2_iterator.h>


#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/filter.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

using namespace std;

vector<pcl::PointCloud<pcl::PointXYZRGB>> steps;
vector<pcl::PointCloud<pcl::PointXYZRGB>> step_keypoints;
vector<pcl::PointCloud<pcl::Normal>> step_points_normals;
vector<pcl::PointCloud<pcl::SHOT352>> step_points_descriptors;

pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;

void cloudCallback (const sensor_msgs::PointCloud2& msg){
    
    return;
}

int main (int argc, char** argv){
    ros::init (argc, argv, "poi_detection");
    ros::NodeHandle nh;

    string in_topic;
    nh.param("poi_detection/input_topic", in_topic, string("zed/point_cloud/cloud_registered"));
    nh.param("poi_detection/algorithm", in_topic, string("GC")); // the alternative is "Hough"

    ros::ServiceClient client = nh.serviceClient<pointcloud_annotator::Update>("annotated_pointcloud_publisher/update");

    ros::Subscriber sub = nh.subscribe (in_topic, 1, cloudCallback);

    pointcloud_annotator::Update update;
    update.request.annotation = "step";
    while(!client.call(update)){
        ROS_WARN("Failed to call the annotation service. Retrying every 2 seconds...");
        ros::Duration(2).sleep();
    }

    for(auto const pointcloud : update.response.set_of_points){
        pcl::PCLPointCloud2 pcl_cloud;
        pcl_conversions::toPCL(pointcloud, pcl_cloud);
        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_ptr = cloud.makeShared();
        cout << cloud_ptr->size() << endl;
        pcl::fromPCLPointCloud2(pcl_cloud, cloud);
        cout << cloud_ptr->size() << endl;

        steps.push_back(cloud);
        break;

        // Normals
        pcl::PointCloud<pcl::Normal> model_normals;

        norm_est.setKSearch (10);
        norm_est.setInputCloud (cloud_ptr);
        norm_est.compute (model_normals);

        step_points_normals.push_back(model_normals);

        // Keypoints
        //step_keypoints.push_back();

        // Descriptors
        //step_points_descriptors.push_back();
    }

    ros::spin();
}