#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include "pointcloud_annotator/Update.h"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud2_iterator.h>


#include <pcl/point_cloud.h>
#include <pcl/features/board.h>
#include <pcl/correspondence.h>
#include <pcl/features/shot_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/recognition/cg/geometric_consistency.h>

using namespace std;

ros::Publisher pub;

vector<double> model_resolutions;
vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> steps;
vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> step_keypoints;
vector<pcl::PointCloud<pcl::Normal>::Ptr> step_points_normals;
vector<pcl::PointCloud<pcl::SHOT352>::Ptr> step_points_descriptors;

pcl::UniformSampling<pcl::PointXYZRGB> uniform_sampling;
pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;
pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> descr_est;

bool viz = true;
bool use_gc_ = true;
bool super_debug = false;
string frame_id = "base_link";

int scene_k_means_search_size = 0;
double scene_uniform_sampling_radius = 0;
double scene_descriptors_search_radius = 0;

int model_k_means_search_size = 10;
double model_uniform_sampling_radius = 0;
double model_descriptors_search_radius = 0;

double cg_size = 0;
double cg_threshold = 5;
double maximum_neighbors_distance = 0;
double rf_radius = 0;

double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud){
    double res = 0;
    int n_points = 0;
    int nres;
    vector<int> indices (2);
    vector<float> sqr_distances (2);
    pcl::search::KdTree<pcl::PointXYZRGB> tree;
    tree.setInputCloud (cloud);

    for (size_t i = 0; i < cloud->size (); ++i){
        if (! pcl_isfinite ((*cloud)[i].x)){
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2){
            res += sqrt (sqr_distances[1]);
            n_points++;
        }
    }
    if (n_points != 0){
        res /= n_points;
    }
    return res;
}

void cloudCallback (const sensor_msgs::PointCloud2& msg){
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::toPCL(msg, pcl_cloud);

    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::fromPCLPointCloud2(pcl_cloud, *cloud_ptr);

    // Normals
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals (new pcl::PointCloud<pcl::Normal> ());

    norm_est.setKSearch (scene_k_means_search_size);
    norm_est.setInputCloud (cloud_ptr);
    norm_est.compute (*scene_normals);


    for(unsigned s=0; s < steps.size(); s++){
        // Keypoints
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
        uniform_sampling.setInputCloud (cloud_ptr);
        uniform_sampling.setRadiusSearch (scene_uniform_sampling_radius * model_resolutions[s]);
        uniform_sampling.filter (*scene_keypoints);

        // Descriptors
        pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
        descr_est.setRadiusSearch (scene_descriptors_search_radius * model_resolutions[s]);
        descr_est.setInputCloud (scene_keypoints);
        descr_est.setInputNormals (scene_normals);
        descr_est.setSearchSurface (cloud_ptr);
        descr_est.compute (*scene_descriptors);

        // Find Correspondences with KdTree
        pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
        pcl::KdTreeFLANN<pcl::SHOT352> match_search;
        match_search.setInputCloud (step_points_descriptors[s]);

        for (size_t i = 0; i < scene_descriptors->size (); ++i){
            vector<int> neigh_indices (1);
            vector<float> neigh_sqr_dists (1);
             //skip NaNs
            if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])){
                continue;
            }
            int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
             //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
            if(found_neighs == 1 && neigh_sqr_dists[0] < maximum_neighbors_distance){
                pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
                model_scene_corrs->push_back (corr);
            }

        }
        cout << "Correspondences found: " << model_scene_corrs->size () << endl;


        vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rot_translations;
        vector<pcl::Correspondences> clustered_corrs;

        if (!use_gc_){
            // Hough
            pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());
            pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());

            pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::ReferenceFrame> rf_est;
            rf_est.setFindHoles (true);
            rf_est.setRadiusSearch (rf_radius * model_resolutions[s]);

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
            clusterer.setHoughBinSize (cg_size * model_resolutions[s]);
            clusterer.setHoughThreshold (cg_threshold);
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
            gc_clusterer.setGCSize (cg_size * model_resolutions[s]);
            gc_clusterer.setGCThreshold (cg_threshold);

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
            for (size_t i = 0; i < rot_translations.size (); ++i){
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_model (new pcl::PointCloud<pcl::PointXYZRGB> ());
                pcl::transformPointCloud (*steps[s], *rotated_model, rot_translations[i]);

                sensor_msgs::PointCloud2 output;
                pcl::PCLPointCloud2 pcl_cloud;
                pcl::toPCLPointCloud2(*rotated_model, pcl_cloud);
                pcl_conversions::fromPCL(pcl_cloud, output);
                output.header.frame_id = frame_id;
                pub.publish (output);
            }

        }

        if(super_debug){
            pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
            viewer.addPointCloud (cloud_ptr, "scene_cloud");

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model (new pcl::PointCloud<pcl::PointXYZRGB> ());
            pcl::transformPointCloud (*steps[s], *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
            viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");

            for (size_t i = 0; i < rot_translations.size (); ++i){
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_model (new pcl::PointCloud<pcl::PointXYZRGB> ());
                pcl::transformPointCloud (*steps[s], *rotated_model, rot_translations[i]);

                stringstream ss_cloud;
                ss_cloud << "instance" << i;

                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> rotated_model_color_handler (rotated_model, 255, 0, 0);
                viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
            }
            while (!viewer.wasStopped ()){
                viewer.spinOnce ();
            }
        }
    }

}

int main (int argc, char** argv){
    ros::init (argc, argv, "poi_detection");
    ros::NodeHandle nh;

    string in_topic;
    string algorithm;
    nh.param("poi_detection/scene_descriptors_search_radius", scene_descriptors_search_radius, 15.0);
    nh.param("poi_detection/model_descriptors_search_radius", model_descriptors_search_radius, 15.0);
    nh.param("poi_detection/scene_uniform_sampling_radius", scene_uniform_sampling_radius, 20.0);
    nh.param("poi_detection/model_uniform_sampling_radius", model_uniform_sampling_radius, 7.5);
    nh.param("poi_detection/input_topic", in_topic, string("zed/point_cloud/cloud_registered"));
    nh.param("poi_detection/maximum_neighbors_distance", maximum_neighbors_distance, 0.25);
    nh.param("poi_detection/scene_k_means_search_size", scene_k_means_search_size, 10);
    nh.param("poi_detection/model_k_means_search_size", model_k_means_search_size, 10);
    nh.param("poi_detection/frame_id", frame_id, string("base_link"));
    nh.param("poi_detection/algorithm", algorithm, string("GC")); // the alternative is "Hough"
    nh.param("poi_detection/cg_threshold", cg_threshold, 5.0);
    nh.param("poi_detection/super_debug", super_debug, false);
    nh.param("poi_detection/rf_radius", rf_radius, 10.0);
    nh.param("poi_detection/cg_size", cg_size, 10.0);
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
        double resolution = computeCloudResolution(cloud_ptr);
        model_resolutions.push_back(resolution);

        // Normals
        pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal> ());

        norm_est.setKSearch (model_k_means_search_size);
        norm_est.setInputCloud (cloud_ptr);
        norm_est.compute (*model_normals);

        step_points_normals.push_back(model_normals);

        // Keypoints
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
        uniform_sampling.setInputCloud (cloud_ptr);
        uniform_sampling.setRadiusSearch (model_uniform_sampling_radius * resolution);
        uniform_sampling.filter (*model_keypoints);
        step_keypoints.push_back(model_keypoints);

        // Descriptors
        pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
        descr_est.setRadiusSearch (model_descriptors_search_radius * resolution);
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