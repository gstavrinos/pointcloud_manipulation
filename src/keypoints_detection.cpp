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
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>

using namespace std;

ros::Publisher pub;

vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> steps;
vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> step_keypoints;
vector<pcl::PointCloud<pcl::Normal>::Ptr> step_points_normals;
vector<pcl::PointCloud<pcl::SHOT352>::Ptr> step_points_descriptors;

pcl::UniformSampling<pcl::PointXYZRGB> uniform_sampling;
pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;
pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> descr_est;

bool viz = true;
bool use_gc_ = true;

void cloudCallback (const sensor_msgs::PointCloud2& msg){
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::toPCL(msg, pcl_cloud);

    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::fromPCLPointCloud2(pcl_cloud, *cloud_ptr);

    // Normals
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals (new pcl::PointCloud<pcl::Normal> ());

    norm_est.setKSearch (10);
    norm_est.setInputCloud (cloud_ptr);
    norm_est.compute (*scene_normals);

    // Keypoints
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
    uniform_sampling.setInputCloud (cloud_ptr);
    uniform_sampling.setRadiusSearch (0.1f);
    uniform_sampling.filter (*scene_keypoints);

    // Descriptors
    pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    descr_est.setRadiusSearch (0.12f);
    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (cloud_ptr);
    descr_est.compute (*scene_descriptors);

    // Find Correspondences with KdTree

    for(unsigned s=0; s < steps.size(); s++){
        pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
        pcl::KdTreeFLANN<pcl::SHOT352> match_search;
        match_search.setInputCloud (step_points_descriptors[s]);

        for (size_t i = 0; i < scene_descriptors->size (); ++i){
            std::vector<int> neigh_indices (1);
            std::vector<float> neigh_sqr_dists (1);
             //skip NaNs
            if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])){
                continue;
            }
            int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
             //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
            if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f){
                pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
                model_scene_corrs->push_back (corr);
            }

        }
        std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;


        vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rot_translations;
        vector<pcl::Correspondences> clustered_corrs;

        if (!use_gc_){
            // Hough
            pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());
            pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());

            pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::ReferenceFrame> rf_est;
            rf_est.setFindHoles (true);
            rf_est.setRadiusSearch (0.1f);

            rf_est.setInputCloud (step_keypoints[s]);
            rf_est.setInputNormals (step_points_normals[s]);
            rf_est.setSearchSurface (steps[s]);
            rf_est.compute (*model_rf);

            rf_est.setInputCloud (scene_keypoints);
            rf_est.setInputNormals (scene_normals);
            rf_est.setSearchSurface (cloud_ptr);
            rf_est.compute (*scene_rf);

            //  Clustering
            pcl::Hough3DGrouping<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
            clusterer.setHoughBinSize (0.1f);
            clusterer.setHoughThreshold (5.0f);
            clusterer.setUseInterpolation (true);
            clusterer.setUseDistanceWeight (false);

            clusterer.setInputCloud (step_keypoints[s]);
            clusterer.setInputRf (model_rf);
            clusterer.setSceneCloud (scene_keypoints);
            clusterer.setSceneRf (scene_rf);
            clusterer.setModelSceneCorrespondences (model_scene_corrs);

            //clusterer.cluster (clustered_corrs);
            clusterer.recognize (rot_translations, clustered_corrs);
        }
        // Geometric Consistency (GC)
        else{
            pcl::GeometricConsistencyGrouping<pcl::PointXYZRGB, pcl::PointXYZRGB> gc_clusterer;
            gc_clusterer.setGCSize (10000.0f);
            gc_clusterer.setGCThreshold (3.0f);

            gc_clusterer.setInputCloud (step_keypoints[s]);
            gc_clusterer.setSceneCloud (scene_keypoints);
            gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

            //gc_clusterer.cluster (clustered_corrs);
            gc_clusterer.recognize (rot_translations, clustered_corrs);
        }

        // Currently we only visualize the result of the recognition
        // major TODO here, if the results are interesting! :)
        if(viz){
            cout << rot_translations.size() << endl;
            cout << clustered_corrs.size() << endl;
            for (size_t i = 0; i < rot_translations.size (); ++i){
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_model (new pcl::PointCloud<pcl::PointXYZRGB> ());
                pcl::transformPointCloud (*steps[s], *rotated_model, rot_translations[i]);

                sensor_msgs::PointCloud2 output;
                pcl::PCLPointCloud2 pcl_cloud;
                pcl::toPCLPointCloud2(*rotated_model, pcl_cloud);
                pcl_conversions::fromPCL(pcl_cloud, output);
                pub.publish (output);



            }
        }
    }

}

int main (int argc, char** argv){
    ros::init (argc, argv, "poi_detection");
    ros::NodeHandle nh;

    string in_topic;
    string algorithm;
    nh.param("poi_detection/input_topic", in_topic, string("zed/point_cloud/cloud_registered"));
    nh.param("poi_detection/algorithm", algorithm, string("GC")); // the alternative is "Hough"
    nh.param("poi_detection/enable_viz", viz, true);

    pub = nh.advertise<sensor_msgs::PointCloud2> ("poi_detection/viz", 1);

    if(algorithm == "Hough"){
        use_gc_ = false;
    }
    else if(algorithm != "GC"){
        ROS_WARN("Wrong algorithm parameter. Using GC...");
    }


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

        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB> ());
        pcl::fromPCLPointCloud2(pcl_cloud, *cloud_ptr);

        steps.push_back(cloud_ptr);

        // Normals
        pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal> ());

        norm_est.setKSearch (10);
        norm_est.setInputCloud (cloud_ptr);
        norm_est.compute (*model_normals);

        step_points_normals.push_back(model_normals);

        // Keypoints
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
        uniform_sampling.setInputCloud (cloud_ptr);
        uniform_sampling.setRadiusSearch (0.01f);
        uniform_sampling.filter (*model_keypoints);
        step_keypoints.push_back(model_keypoints);

        // Descriptors
        pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
        descr_est.setRadiusSearch (0.06f);
        descr_est.setInputCloud (model_keypoints);
        descr_est.setInputNormals (model_normals);
        descr_est.setSearchSurface (cloud_ptr);
        descr_est.compute (*model_descriptors);
        step_points_descriptors.push_back(model_descriptors);
    }

    while(ros::ok()){
        ros::spin();
    }
}