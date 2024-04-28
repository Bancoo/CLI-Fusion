#include "include/point_xyzir.h"
#define PCL_NO_PRECOMPILE
#include "process_lidar.h"
#include "process_camera.h"
#include <list>
#include <cmath>
#include <algorithm>
#include <iterator>
#include "Cfg.h"

clock_t starttime,endtime;

bool compareZ(const pcl::PointXYZI& p1, const pcl::PointXYZI& p2)
{
    return p1.z < p2.z;
}
bool compareX(const pcl::PointXYZI& p1, const pcl::PointXYZI& p2)
{
    return p1.x < p2.x;
}

int main()
{
    // 1、initialization
    Cfg cfg;
	std::string max_number;
	cfg.readConfigFile("../params.cfg", "max_number", max_number);
	std::string min_number;
	cfg.readConfigFile("../params.cfg", "min_number", min_number);
    std::string input_file_path;
	cfg.readConfigFile("../params.cfg", "input_file_path", input_file_path);
    std::string output_file_path;
	cfg.readConfigFile("../params.cfg", "output_file_path", output_file_path);
    std::string iflook;
	cfg.readConfigFile("../params.cfg", "iflook", iflook);
    int look = atoi(iflook.c_str());
      
    PROCESS_LIDAR process;
    process_camera handle;

    std::ifstream file("../data/imu.txt");
    std::vector<std::vector<double>> imu_data;

    ofstream ofs;
    ofs.open("../data/times_and_angle.txt", ios::ate);

    imu_data = process.ReadImuFile(file);

	pcl::PointCloud<MyPointType>::Ptr cloud(new pcl::PointCloud<MyPointType>);
    cv::Mat image;
    
    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    // 2、Cyclic processing data
    int start_number = atoi(max_number.c_str());
    int end_number = atoi(min_number.c_str());
    for (int cal = start_number ; cal < end_number; ++cal)
    {
        cout << "$$$$$$$$$$$$$$$$$$$$$$$$  "<<cal << "  $$$$$$$$$$$$$$$$$$$$$$" << endl;
        std::stringstream pcd_in;
        pcd_in << "../data/" << input_file_path << "/pcd/" << cal << ".pcd";
        reader.read<MyPointType>(pcd_in.str(), *cloud);
        if (cloud->points.size() == 0)
        {
            PCL_ERROR("Couldn't read that pcd file\n");
            return (-1);
	    }
        cv::Mat image;
        std::stringstream image_in;
        image_in << "../data/" << input_file_path << "/image/" << cal << ".jpg";
        image = imread(image_in.str(), IMREAD_COLOR);
        if(image.empty())
        {
            cout << "请确认图像文件名称及路径是否正确" << std::endl;
            return -1;
        }
        
        starttime=clock();
        cv::Mat brightened_image;
        double aph = 1.0;
        image.convertTo(brightened_image, -1, aph, 0);

        cv::Rect rect2(600, 500, 720, 480); 
        cv::Mat image_roi = brightened_image(rect2);

        // 2.1、Im-Itti-ExG
        #pragma region
        std::vector<cv::Mat> rgbchannels;
        cv::split(image_roi, rgbchannels);
        cv::Mat r = rgbchannels[0] / (rgbchannels[1] + rgbchannels[0] + rgbchannels[2]) * 255;
        cv::Mat g = rgbchannels[1] / (rgbchannels[1] + rgbchannels[0] + rgbchannels[2]) * 255;
        cv::Mat b = rgbchannels[2] / (rgbchannels[1] + rgbchannels[0] + rgbchannels[2]) * 255;
        cv::Mat exg = (2 * g - b - r);
        cv::Mat binary_g;
        threshold(exg, binary_g, 0, 255, THRESH_BINARY | THRESH_OTSU);

        cv::Mat img1 = image_roi;
        cv::Mat gau = cv::getGaussianKernel(3, -1, CV_32F);
        cv::Mat img2, img3, img4, img5, img6, img7, img8, img9;
        cv::GaussianBlur(img1, img2, cv::Size(3, 3), 0);
        cv::resize(img2, img2, cv::Size(img1.cols / 2, img1.rows / 2));
        cv::GaussianBlur(img2, img3, cv::Size(3, 3), 0);
        cv::resize(img3, img3, cv::Size(img1.cols / 4, img1.rows / 4));
        cv::GaussianBlur(img3, img4, cv::Size(3, 3), 0);
        cv::resize(img4, img4, cv::Size(img1.cols / 8, img1.rows / 8));
        cv::GaussianBlur(img4, img5, cv::Size(3, 3), 0);
        cv::resize(img5, img5, cv::Size(img1.cols / 16, img1.rows / 16));
        cv::GaussianBlur(img5, img6, cv::Size(3, 3), 0);
        cv::resize(img6, img6, cv::Size(img1.cols / 32, img1.rows / 32));
        cv::GaussianBlur(img6, img7, cv::Size(3, 3), 0);
        cv::resize(img7, img7, cv::Size(img1.cols / 64, img1.rows / 64));
        cv::GaussianBlur(img7, img8, cv::Size(3, 3), 0);
        cv::resize(img8, img8, cv::Size(img1.cols / 128, img1.rows / 128));
        cv::GaussianBlur(img8, img9, cv::Size(3, 3), 0);
        cv::resize(img9, img9, cv::Size(img1.cols / 256, img1.rows / 256));

        cv::Mat I1 = handle.cal_I(img1);
        cv::Mat I2 = handle.cal_I(img2);
        cv::Mat I3 = handle.cal_I(img3);
        cv::Mat I4 = handle.cal_I(img4);
        cv::Mat I5 = handle.cal_I(img5);
        cv::Mat I6 = handle.cal_I(img6);
        cv::Mat I7 = handle.cal_I(img7);
        cv::Mat I8 = handle.cal_I(img8);
        cv::Mat I9 = handle.cal_I(img9);

        cv::Mat R1=handle.cal_R(img1);
        cv::Mat R2=handle.cal_R(img2);
        cv::Mat R3=handle.cal_R(img3);
        cv::Mat R4=handle.cal_R(img4);
        cv::Mat R5=handle.cal_R(img5);
        cv::Mat R6=handle.cal_R(img6);
        cv::Mat R7=handle.cal_R(img7);
        cv::Mat R8=handle.cal_R(img8);
        cv::Mat R9=handle.cal_R(img9);
        
        cv::Mat G1=handle.cal_G(img1);
        cv::Mat G2=handle.cal_G(img2);
        cv::Mat G3=handle.cal_G(img3);
        cv::Mat G4=handle.cal_G(img4);
        cv::Mat G5=handle.cal_G(img5);
        cv::Mat G6=handle.cal_G(img6);
        cv::Mat G7=handle.cal_G(img7);
        cv::Mat G8=handle.cal_G(img8);
        cv::Mat G9=handle.cal_G(img9);

        cv::Mat B1=handle.cal_B(img1);
        cv::Mat B2=handle.cal_B(img2);
        cv::Mat B3=handle.cal_B(img3);
        cv::Mat B4=handle.cal_B(img4);
        cv::Mat B5=handle.cal_B(img5);
        cv::Mat B6=handle.cal_B(img6);
        cv::Mat B7=handle.cal_B(img7);
        cv::Mat B8=handle.cal_B(img8);
        cv::Mat B9=handle.cal_B(img9);
        
        cv::Mat Y1=handle.cal_Y(img1);
        cv::Mat Y2=handle.cal_Y(img2);
        cv::Mat Y3=handle.cal_Y(img3);
        cv::Mat Y4=handle.cal_Y(img4);
        cv::Mat Y5=handle.cal_Y(img5);
        cv::Mat Y6=handle.cal_Y(img6);
        cv::Mat Y7=handle.cal_Y(img7);
        cv::Mat Y8=handle.cal_Y(img8);
        cv::Mat Y9=handle.cal_Y(img9);
        
        cv::Mat I_2_5=(handle.Ifeature_diff(I2,I5));
        cv::Mat I_2_6=(handle.Ifeature_diff(I2,I6));
        cv::Mat I_3_6=(handle.Ifeature_diff(I3,I6));
        cv::Mat I_3_7=(handle.Ifeature_diff(I3,I7));
        cv::Mat I_4_7=(handle.Ifeature_diff(I4,I7));
        cv::Mat I_4_8=(handle.Ifeature_diff(I4,I8));

        cv::Mat RG_2_5=(handle.RGBfeature_diff(R2,G2,R5,G5));
        cv::Mat RG_2_6=(handle.RGBfeature_diff(R2,G2,R6,G6));
        cv::Mat RG_3_6=(handle.RGBfeature_diff(R3,G3,R6,G6));
        cv::Mat RG_3_7=(handle.RGBfeature_diff(R3,G3,R7,G7));
        cv::Mat RG_4_7=(handle.RGBfeature_diff(R4,G4,R7,G7));
        cv::Mat RG_4_8=(handle.RGBfeature_diff(R4,G4,R8,G8));

        cv::Mat BY_2_5=(handle.RGBfeature_diff(B2,B2,Y5,Y5));
        cv::Mat BY_2_6=(handle.RGBfeature_diff(B2,B2,Y6,Y6));
        cv::Mat BY_3_6=(handle.RGBfeature_diff(B3,B3,Y6,Y6));
        cv::Mat BY_3_7=(handle.RGBfeature_diff(B3,B3,Y7,Y7));
        cv::Mat BY_4_7=(handle.RGBfeature_diff(B4,B4,Y7,Y7));
        cv::Mat BY_4_8=(handle.RGBfeature_diff(B4,B4,Y8,Y8));

        I_2_5=handle.normalizeImg(I_2_5);
        I_2_6=handle.normalizeImg(I_2_6);
        I_3_6=handle.normalizeImg(I_3_6);
        I_3_7=handle.normalizeImg(I_3_7);
        I_4_7=handle.normalizeImg(I_4_7);
        I_4_8=handle.normalizeImg(I_4_8);

        RG_2_5=handle.normalizeImg(RG_2_5);
        RG_2_6=handle.normalizeImg(RG_2_6);
        RG_3_6=handle.normalizeImg(RG_3_6);
        RG_3_7=handle.normalizeImg(RG_3_7);
        RG_4_7=handle.normalizeImg(RG_4_7);
        RG_4_8=handle.normalizeImg(RG_4_8);

        BY_2_5=handle.normalizeImg(BY_2_5);
        BY_2_6=handle.normalizeImg(BY_2_6);
        BY_3_6=handle.normalizeImg(BY_3_6);
        BY_3_7=handle.normalizeImg(BY_3_7);
        BY_4_7=handle.normalizeImg(BY_4_7);
        BY_4_8=handle.normalizeImg(BY_4_8);

        cv::Mat c2 = handle.mergeRGBYcomp(RG_2_5, BY_2_5, RG_2_6, BY_2_6);
        cv::Mat c3 = handle.mergeRGBYcomp(RG_3_6, BY_3_6, RG_3_7, BY_3_7);
        cv::Mat c4 = handle.mergeRGBYcomp(RG_4_7, BY_4_7, RG_4_8, BY_4_8);

        cv::resize(c3, c3, c2.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(c4, c4, c2.size(), 0, 0, cv::INTER_NEAREST);

        cv::Mat C = handle.UpGet(c2, c3, c4);
        cv::resize(C, C, img1.size(), 0, 0, cv::INTER_NEAREST);
        std::pair<double, double> threshold = handle.cal_th(C);
        double th = (threshold.first + threshold.second) * 1;

        cv::Mat Im_Itti_out = cv::Mat::zeros(C.size(), C.type());
        cv::threshold(C, Im_Itti_out, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        cv::Mat S_out;
        cv::add(binary_g / 2, Im_Itti_out / 2, S_out);

        cv::threshold(S_out, S_out, 200, 255, cv::THRESH_BINARY);
        #pragma endregion
        

        Eigen::Matrix4d lidar_installation;
        lidar_installation << 
        0.97761, 0.000630253,    0.210424,           0,
        0.000630253,    0.999982,  -0.0059232,           0,
        -0.210424,   0.0059232,    0.977592,           0,
            0,           0,           0,           1;

        Eigen::Matrix4d lidar_installation_inverse = lidar_installation.inverse();

        cv::Mat cameramat = cv::Mat::zeros(3, 3, CV_64F);
        double *data_1 = cameramat.ptr<double>(); 
        data_1[0] = 1452.65;
        data_1[2] = 969.05;
        data_1[4] = 1453.85;
        data_1[5] = 595.991;
        data_1[8] = 1;
        process.SetCameramat(cameramat); 

        cv::Mat distcoeff = cv::Mat::zeros(1, 4, CV_64F);
        double *data_2 = distcoeff.ptr<double>();
        data_2[0] = -0.087997;
        data_2[1] = 0.097785;
        data_2[2] = -0.002167;
        data_2[3] = 0.002477;
        process.SetDistcoeff(distcoeff);

        // transformMatrix，camera-->lidar ， lidar-->camera
        Eigen::Matrix4d transformMatrix_c2l = process.do_transformMatrix_c2l();
        Eigen::Matrix4d transformMatrix_l2c = transformMatrix_c2l.inverse();

        Eigen::Matrix4d lidar_xy;
        lidar_xy << 0, -1, 0, 0,
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
        Eigen::Matrix4d lidar_xy_inverse = lidar_xy.inverse();

        Eigen::Matrix4d transform = process.do_transform_imu(imu_data, cal);
        Eigen::Matrix4d transform_inverse = transform.inverse();
        
        
        // 2.2、FusSeg
        pcl::PointCloud<MyPointType>::Ptr cloud_roi(new pcl::PointCloud<MyPointType>);
        pcl::PassThrough<MyPointType> pass_xy;
        pass_xy.setInputCloud(cloud);
        pass_xy.setFilterFieldName("x");
        pass_xy.setFilterLimits(0.0, 10);
        pass_xy.setFilterLimitsNegative(false);
        pass_xy.filter(*cloud_roi);
        pass_xy.setInputCloud(cloud_roi);
        pass_xy.setFilterFieldName("y");
        pass_xy.setFilterLimits(-1.5, 1.5);
        pass_xy.setFilterLimitsNegative(false);
        pass_xy.filter(*cloud_roi);

        savepath;
        prefix = "../data/";
        suffix_name = "/cloud_roi.pcd";
        savepath = prefix + output_file_path.c_str() + suffix_name;
        // writer.writeASCII(savepath, *cloud_roi);

        pcl::PointCloud<MyPointType>::Ptr cloud_l2c(new pcl::PointCloud<MyPointType>);
        pcl::transformPointCloud(*cloud_roi, *cloud_l2c, transformMatrix_l2c);

        std::vector<Point> cloud_c;
        MyPointType tempPoint;
        pcl::PointCloud<MyPointType>::Ptr cloud_center(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_center2world(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_area_1(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_area_2(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_area_3(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_area_4(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_area_5(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr cloud_area_6(new pcl::PointCloud<MyPointType>);

        cv::Mat image_roi_after;
        pcl::PointCloud<MyPointType>::Ptr NoGround(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr NoGround_3456(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr all_area(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr area_135(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr area_246(new pcl::PointCloud<MyPointType>);

        pcl::PointCloud<MyPointType>::Ptr groundCloud(new pcl::PointCloud<MyPointType>);
        if (cloud_l2c->points.size())
        {
            process.seg_cloud_2_6areas(cloud_l2c, cloud_area_1, cloud_area_2, cloud_area_3, cloud_area_4, cloud_area_5, cloud_area_6);
            image_roi_after = brightened_image(rect2);
            
            pcl::transformPointCloud(*cloud_area_1, *cloud_area_1, transformMatrix_c2l);
            pcl::transformPointCloud(*cloud_area_1, *cloud_area_1, lidar_installation);
            pcl::transformPointCloud(*cloud_area_1, *cloud_area_1, lidar_xy);
            pcl::transformPointCloud(*cloud_area_1, *cloud_area_1, transform);
            pcl::transformPointCloud(*cloud_area_1, *cloud_area_1, lidar_xy_inverse);

            pcl::transformPointCloud(*cloud_area_2, *cloud_area_2, transformMatrix_c2l);
            pcl::transformPointCloud(*cloud_area_2, *cloud_area_2, lidar_installation);
            pcl::transformPointCloud(*cloud_area_2, *cloud_area_2, lidar_xy);
            pcl::transformPointCloud(*cloud_area_2, *cloud_area_2, transform);
            pcl::transformPointCloud(*cloud_area_2, *cloud_area_2, lidar_xy_inverse);

            pcl::transformPointCloud(*cloud_area_3, *cloud_area_3, transformMatrix_c2l);
            pcl::transformPointCloud(*cloud_area_3, *cloud_area_3, lidar_installation);
            pcl::transformPointCloud(*cloud_area_3, *cloud_area_3, lidar_xy);
            pcl::transformPointCloud(*cloud_area_3, *cloud_area_3, transform);
            pcl::transformPointCloud(*cloud_area_3, *cloud_area_3, lidar_xy_inverse);

            pcl::transformPointCloud(*cloud_area_4, *cloud_area_4, transformMatrix_c2l);
            pcl::transformPointCloud(*cloud_area_4, *cloud_area_4, lidar_installation);
            pcl::transformPointCloud(*cloud_area_4, *cloud_area_4, lidar_xy);
            pcl::transformPointCloud(*cloud_area_4, *cloud_area_4, transform);
            pcl::transformPointCloud(*cloud_area_4, *cloud_area_4, lidar_xy_inverse);

            pcl::transformPointCloud(*cloud_area_5, *cloud_area_5, transformMatrix_c2l);
            pcl::transformPointCloud(*cloud_area_5, *cloud_area_5, lidar_installation);
            pcl::transformPointCloud(*cloud_area_5, *cloud_area_5, lidar_xy);
            pcl::transformPointCloud(*cloud_area_5, *cloud_area_5, transform);
            pcl::transformPointCloud(*cloud_area_5, *cloud_area_5, lidar_xy_inverse);

            pcl::transformPointCloud(*cloud_area_6, *cloud_area_6, transformMatrix_c2l);
            pcl::transformPointCloud(*cloud_area_6, *cloud_area_6, lidar_installation);
            pcl::transformPointCloud(*cloud_area_6, *cloud_area_6, lidar_xy);
            pcl::transformPointCloud(*cloud_area_6, *cloud_area_6, transform);
            pcl::transformPointCloud(*cloud_area_6, *cloud_area_6, lidar_xy_inverse);

            pcl::PointCloud<MyPointType>::Ptr cloud_area_1_norm(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr cloud_area_2_norm(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr cloud_area_3_norm(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr cloud_area_4_norm(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr cloud_area_5_norm(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr cloud_area_6_norm(new pcl::PointCloud<MyPointType>);

            cloud_area_1_norm = process.pointsNormliaz(cloud_area_1);
            cloud_area_2_norm = process.pointsNormliaz(cloud_area_2);
            cloud_area_3_norm = process.pointsNormliaz(cloud_area_3);
            cloud_area_4_norm = process.pointsNormliaz(cloud_area_4);
            cloud_area_5_norm = process.pointsNormliaz(cloud_area_5);
            cloud_area_6_norm = process.pointsNormliaz(cloud_area_6);

            int thesh_1, thesh_2, thesh_3, thesh_4, thesh_5, thesh_6;
            thesh_1 = thesh_2 = thesh_3 = thesh_4 = thesh_5 = thesh_6 = 0;
            thesh_1 = process.cal_th(*cloud_area_1_norm);
            thesh_2 = process.cal_th(*cloud_area_2_norm);
            thesh_3 = process.cal_th(*cloud_area_3_norm);
            thesh_4 = process.cal_th(*cloud_area_4_norm);
            thesh_5 = process.cal_th(*cloud_area_5_norm);
            thesh_6 = process.cal_th(*cloud_area_6_norm);

            std::sort(cloud_area_1->points.begin(), cloud_area_1->points.end(), compareZ);
            std::sort(cloud_area_2->points.begin(), cloud_area_2->points.end(), compareZ);
            std::sort(cloud_area_3->points.begin(), cloud_area_3->points.end(), compareZ);
            std::sort(cloud_area_4->points.begin(), cloud_area_4->points.end(), compareZ);
            std::sort(cloud_area_5->points.begin(), cloud_area_5->points.end(), compareZ);
            std::sort(cloud_area_6->points.begin(), cloud_area_6->points.end(), compareZ);
            
            pcl::PointCloud<MyPointType>::Ptr filtered_cloud_area1(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filtered_cloud_area2(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filtered_cloud_area3(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filtered_cloud_area4(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filtered_cloud_area5(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filtered_cloud_area6(new pcl::PointCloud<MyPointType>);

            filtered_cloud_area1->points.assign(cloud_area_1->points.begin() + thesh_1, cloud_area_1->points.end());
            filtered_cloud_area1->width = filtered_cloud_area1->points.size();
            filtered_cloud_area1->height = 1;
            filtered_cloud_area1->is_dense = true;

            filtered_cloud_area2->points.assign(cloud_area_2->points.begin() + thesh_2, cloud_area_2->points.end());
            filtered_cloud_area2->width = filtered_cloud_area2->points.size();
            filtered_cloud_area2->height = 1;
            filtered_cloud_area2->is_dense = true;

            filtered_cloud_area3->points.assign(cloud_area_3->points.begin() + thesh_3, cloud_area_3->points.end());
            filtered_cloud_area3->width = filtered_cloud_area3->points.size();
            filtered_cloud_area3->height = 1;
            filtered_cloud_area3->is_dense = true;

            filtered_cloud_area4->points.assign(cloud_area_4->points.begin() + thesh_4, cloud_area_4->points.end());
            filtered_cloud_area4->width = filtered_cloud_area4->points.size();
            filtered_cloud_area4->height = 1;
            filtered_cloud_area4->is_dense = true;

            filtered_cloud_area5->points.assign(cloud_area_5->points.begin() + thesh_5, cloud_area_5->points.end());
            filtered_cloud_area5->width = filtered_cloud_area5->points.size();
            filtered_cloud_area5->height = 1;
            filtered_cloud_area5->is_dense = true;

            filtered_cloud_area6->points.assign(cloud_area_6->points.begin() + thesh_6, cloud_area_6->points.end());
            filtered_cloud_area6->width = filtered_cloud_area6->points.size();
            filtered_cloud_area6->height = 1;
            filtered_cloud_area6->is_dense = true;

            int mask_diameter = 11;
            filtered_cloud_area1 = process.secfilter(*filtered_cloud_area1, mask_diameter);
            filtered_cloud_area2 = process.secfilter(*filtered_cloud_area2, mask_diameter);
            filtered_cloud_area3 = process.secfilter(*filtered_cloud_area3, mask_diameter);
            filtered_cloud_area4 = process.secfilter(*filtered_cloud_area4, mask_diameter);
            filtered_cloud_area5 = process.secfilter(*filtered_cloud_area5, mask_diameter);
            filtered_cloud_area6 = process.secfilter(*filtered_cloud_area6, mask_diameter);

            *all_area = *cloud_area_1 + *cloud_area_2 + *cloud_area_3 + *cloud_area_4 + *cloud_area_5 + *cloud_area_6;
            *area_135 = *cloud_area_1 + *cloud_area_3 + *cloud_area_5;
            *area_246 = *cloud_area_2 + *cloud_area_4 + *cloud_area_6;

            pcl::PointCloud<MyPointType>::Ptr filter_area1_2(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filter_area3_4(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr filter_area5_6(new pcl::PointCloud<MyPointType>);

            *filter_area1_2 = *filtered_cloud_area1 + *filtered_cloud_area2;
            *filter_area3_4 = *filtered_cloud_area3 + *filtered_cloud_area4;
            *filter_area5_6 = *filtered_cloud_area5 + *filtered_cloud_area6;

            *NoGround = *filter_area1_2 + *filter_area3_4 + *filter_area5_6;
            *NoGround_3456 = *filter_area3_4 + *filter_area5_6;

            pcl::PointCloud<MyPointType>::Ptr ground_cloud_area1(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr ground_cloud_area2(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr ground_cloud_area3(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr ground_cloud_area4(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr ground_cloud_area5(new pcl::PointCloud<MyPointType>);
            pcl::PointCloud<MyPointType>::Ptr ground_cloud_area6(new pcl::PointCloud<MyPointType>);
            
            ground_cloud_area1->points.assign(cloud_area_1->points.begin(), cloud_area_1->points.begin() + thesh_1);
            ground_cloud_area1->width = ground_cloud_area1->points.size();
            ground_cloud_area1->height = 1;
            ground_cloud_area1->is_dense = true;

            ground_cloud_area2->points.assign(cloud_area_2->points.begin(), cloud_area_2->points.begin() + thesh_2);
            ground_cloud_area2->width = ground_cloud_area2->points.size();
            ground_cloud_area2->height = 1;
            ground_cloud_area2->is_dense = true;

            ground_cloud_area3->points.assign(cloud_area_3->points.begin(), cloud_area_3->points.begin() + thesh_3);
            ground_cloud_area3->width = ground_cloud_area3->points.size();
            ground_cloud_area3->height = 1;
            ground_cloud_area3->is_dense = true;

            ground_cloud_area4->points.assign(cloud_area_4->points.begin(), cloud_area_4->points.begin() + thesh_4);
            ground_cloud_area4->width = ground_cloud_area4->points.size();
            ground_cloud_area4->height = 1;
            ground_cloud_area4->is_dense = true;

            ground_cloud_area5->points.assign(cloud_area_5->points.begin(), cloud_area_5->points.begin() + thesh_5);
            ground_cloud_area5->width = ground_cloud_area5->points.size();
            ground_cloud_area5->height = 1;
            ground_cloud_area5->is_dense = true;

            ground_cloud_area6->points.assign(cloud_area_6->points.begin(), cloud_area_6->points.begin() + thesh_6);
            ground_cloud_area6->width = ground_cloud_area6->points.size();
            ground_cloud_area6->height = 1;
            ground_cloud_area6->is_dense = true;
            *groundCloud = *ground_cloud_area1 + *ground_cloud_area2 + *ground_cloud_area3 + *ground_cloud_area4 + *ground_cloud_area5 + *ground_cloud_area6;

            suffix_name = "/groundCloud.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *groundCloud);
            suffix_name = "/offGroundCloud.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *NoGround);

            pcl::PointCloud<MyPointType>::Ptr NoGround_tocamera(new pcl::PointCloud<MyPointType>);
            pcl::transformPointCloud(*NoGround, *NoGround_tocamera, lidar_xy);
            pcl::transformPointCloud(*NoGround_tocamera, *NoGround_tocamera, transform_inverse);
            pcl::transformPointCloud(*NoGround_tocamera, *NoGround_tocamera, lidar_xy_inverse);
            pcl::transformPointCloud(*NoGround_tocamera, *NoGround_tocamera, lidar_installation_inverse);
            pcl::transformPointCloud(*NoGround_tocamera, *NoGround_tocamera, transformMatrix_l2c);

            for (pcl::PointCloud<MyPointType>::const_iterator it = NoGround_tocamera->begin(); it != NoGround_tocamera->end(); it++)
            {
                double tmpx = it->x / it->z;
                double tmpy = it->y / it->z;
                double tmpz = it->z;

                cv::Point2d points2image;
                double r2 = tmpx * tmpx + tmpy * tmpy;
                double r1 = pow(r2, 0.5);
                double a0 = std::atan(r1);
                double a1;
                a1 = a0 * (1 + distcoeff.at<double>(0) * pow(a0, 2) + distcoeff.at<double>(1) * pow(a0, 4) + distcoeff.at<double>(2) * pow(a0, 6) + distcoeff.at<double>(3) * pow(a0, 8));
                points2image.x = (a1 / r1) * tmpx;
                points2image.y = (a1 / r1) * tmpy;

                points2image.x = cameramat.at<double>(0, 0) * points2image.x + cameramat.at<double>(0, 2);
                points2image.y = cameramat.at<double>(1, 1) * points2image.y + cameramat.at<double>(1, 2);

                int point_size = 2;
                if (points2image.y >= 500 and points2image.y < 980 and points2image.x >= 600 and points2image.x < 1320) 
                {
                    cv::circle(brightened_image, cv::Point(points2image.x, points2image.y), point_size, CV_RGB(255, 0, 0), -1);
                }
            }

            suffix_name = "/image_roi22.jpg";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // cv::imwrite(savepath, image_roi);
        }

        pcl::PointCloud<MyPointType>::Ptr posiCloud(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr posiCloud_2nd(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr negCloud(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr negCloud_2nd(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr NoGround_tozero(new pcl::PointCloud<MyPointType>);
        *NoGround_tozero = *NoGround;
        double center_x, center_y;
        if (NoGround->points.size())
        {
            MyPointType minPt_local_area, maxPt_local_area;
            pcl::getMinMax3D (*NoGround, minPt_local_area, maxPt_local_area);
            center_x = (maxPt_local_area.x - minPt_local_area.x) / 2 + minPt_local_area.x;
            center_y = maxPt_local_area.y + minPt_local_area.y;
            for (int i = 0; i < NoGround->points.size(); i++)
            {
                NoGround->points[i].x -= center_x;
            }
            for (int i = 0; i < NoGround_tozero->points.size(); i++)
            {
                NoGround_tozero->points[i].x -= center_x;
            }

            suffix_name = "/NoGround_tozero.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *NoGround_tozero);
            
            double angle_interval = 10.0; 
            double min_angle = 0.0; 
            double max_angle = 180.0; 
            int area_number = max_angle / angle_interval;

            std::vector<std::vector<int>> container_posi(area_number);
            std::vector<std::vector<int>> container_neg(area_number);

            for (int i = 0; i < NoGround->size(); i++)
            {
                double point_angle = atan2(NoGround->points[i].y, NoGround->points[i].x) * 180 / M_PI;
                int segment_index = std::floor(point_angle / angle_interval);

                if(segment_index >= 0)
                {
                    container_posi[segment_index].push_back(i);
                }
                else
                {
                    segment_index = -segment_index - 1;
                    container_neg[segment_index].push_back(i);
                }
            }

            for (size_t i = 0; i < container_posi.size(); i++)
            {
                pcl::IndicesPtr pIndexs_new(new std::vector<int>(container_posi[i]));
                pcl::PointCloud<MyPointType>::Ptr segment_cloud(new pcl::PointCloud<MyPointType>);
                pcl::PointCloud<MyPointType>::Ptr temp_cloud(new pcl::PointCloud<MyPointType>);
                pcl::ExtractIndices<MyPointType> extract;
                extract.setInputCloud(NoGround);
                extract.setIndices(pIndexs_new);
                extract.setNegative(false);
                extract.filter(*segment_cloud);

                double thresh_distXY = process.getmeanXY(segment_cloud);

                std::vector<int> posi_in;
                for (size_t j = 0; j < segment_cloud->points.size(); j++)
                {
                    double distXY = pow(segment_cloud->points[j].x, 2) + pow(segment_cloud->points[j].y, 2);
                    if (distXY  < thresh_distXY)
                    {
                        posi_in.push_back(j);
                    }
                }
                pcl::IndicesPtr pIndexs_in(new std::vector<int>(posi_in));
                pcl::PointCloud<MyPointType>::Ptr temp_in(new pcl::PointCloud<MyPointType>);
                extract.setInputCloud(segment_cloud);
                extract.setIndices(pIndexs_in);
                extract.setNegative(false);
                extract.filter(*temp_in);
                *posiCloud += *temp_in;
                
            }
            for (size_t i = 0; i < container_neg.size(); i++)
            {
                pcl::IndicesPtr pIndexs_new(new std::vector<int>(container_neg[i]));
                pcl::PointCloud<MyPointType>::Ptr segment_cloud(new pcl::PointCloud<MyPointType>);
                pcl::PointCloud<MyPointType>::Ptr temp_cloud(new pcl::PointCloud<MyPointType>);
                pcl::ExtractIndices<MyPointType> extract;
                extract.setInputCloud(NoGround);
                extract.setIndices(pIndexs_new);
                extract.setNegative(false);
                extract.filter(*segment_cloud);

                double thresh_distXY = process.getmeanXY(segment_cloud);
                std::vector<int> neg_in;
                for (size_t j = 0; j < segment_cloud->points.size(); j++)
                {
                    double distXY = pow(segment_cloud->points[j].x, 2) + pow(segment_cloud->points[j].y, 2);
                    if (distXY  < thresh_distXY)
                    {
                        neg_in.push_back(j);
                    }
                }
                pcl::IndicesPtr pIndexs_in(new std::vector<int>(neg_in));
                pcl::PointCloud<MyPointType>::Ptr temp_in(new pcl::PointCloud<MyPointType>);
                extract.setInputCloud(segment_cloud);
                extract.setIndices(pIndexs_in);
                extract.setNegative(false);
                extract.filter(*temp_in);
                *negCloud += *temp_in;
            }
            
            suffix_name = "/posiCloud2.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *posiCloud);
            suffix_name = "/negCloud2.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *negCloud);
        }

        pcl::PointCloud<MyPointType>::Ptr leftCloud(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr rightCloud(new pcl::PointCloud<MyPointType>);

        MyPointType p1, p2, p3, p4;
        double save_thresh = 0.1;
        double left_distance = 0.6; 
        double right_distance = 0.6; 
        Vector2f Cloud_line_1, Cloud_line_2;
        if (posiCloud->points.size() > 0 || negCloud->points.size() > 0)
        {
            if (posiCloud->points.size() > 0)
            {
                pcl::SACSegmentation<MyPointType> seg;
                seg.setOptimizeCoefficients(true);
                seg.setInputCloud(posiCloud);
                seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
                seg.setMethodType(pcl::SAC_RANSAC);
                const Eigen::Vector3f axis(1,0,0);
                const double eps = 5 * DEG_TO_RAD;
                seg.setAxis(axis);
                seg.setEpsAngle(eps);
                seg.setMaxIterations(1000);
                seg.setDistanceThreshold(0.20);

                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                seg.segment(*inliers, *coefficients);

                Cloud_line_1[0] = coefficients->values[3];
                Cloud_line_1[1] = coefficients->values[4];

                Eigen::Vector4f PofL(coefficients->values[0], coefficients->values[1], 0, 0); 
                Eigen::Vector4f VofL(coefficients->values[3], coefficients->values[4], 0, 0); 

                std::sort(posiCloud->points.begin(), posiCloud->points.end(), compareX);
                Eigen::Vector4f Pt(0, 0, 0, 0);

                double distance1 = pcl::sqrPointToLineDistance(Pt, PofL, VofL);
                distance1 = sqrt(distance1);
                
                if (distance1 > left_distance)
                {
                    for (size_t i = 0; i < posiCloud->points.size(); i++)
                    {
                        Eigen::Vector4f Point(posiCloud->points[i].x, posiCloud->points[i].y, 0, 0);
                        double ddis = pcl::sqrPointToLineDistance(Point, PofL, VofL);
                        ddis = sqrt(ddis);

                        if (ddis > save_thresh)
                        {
                            posiCloud_2nd->points.push_back(posiCloud->points[i]);
                            posiCloud_2nd->width = posiCloud_2nd->points.size();
                            posiCloud_2nd->height = 1;
                            posiCloud_2nd->is_dense = true;
                        }
                    }

                    seg.setOptimizeCoefficients(true);
                    seg.setInputCloud(posiCloud_2nd);
                    seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
                    seg.setMethodType(pcl::SAC_RANSAC);
                    seg.setAxis(axis);
                    seg.setEpsAngle(eps);
                    seg.setMaxIterations(1000);
                    seg.setDistanceThreshold(0.10);

                    pcl::ModelCoefficients::Ptr coefficients_2(new pcl::ModelCoefficients);
                    pcl::PointIndices::Ptr inliers_2(new pcl::PointIndices);
                    seg.segment(*inliers, *coefficients_2);
                    PofL << coefficients_2->values[0], coefficients_2->values[1], 0, 0;
                    VofL << coefficients_2->values[3], coefficients_2->values[4], 0, 0;

                    for (size_t i = 0; i < NoGround_tozero->points.size(); i++)
                    {
                        Eigen::Vector4f Point(NoGround_tozero->points[i].x, NoGround_tozero->points[i].y, 0, 0);
                        double ddis = pcl::sqrPointToLineDistance(Point, PofL, VofL);
                        ddis = sqrt(ddis);
                        p3.x = coefficients->values[0] - 5 * coefficients->values[3];
                        p3.y = coefficients->values[1] - 5 * coefficients->values[4];
                        p3.z = coefficients->values[2];
                        p4.x = coefficients->values[0] + 5 * coefficients->values[3];
                        p4.y = coefficients->values[1] + 5 * coefficients->values[4];
                        p4.z = coefficients->values[2];
                        if (ddis < save_thresh)
                        {
                            leftCloud->points.push_back(NoGround_tozero->points[i]);
                            leftCloud->width = leftCloud->points.size();
                            leftCloud->height = 1;
                            leftCloud->is_dense = true;
                        }
                    }
                }

                else if (distance1 < left_distance)
                {
                    for (size_t i = 0; i < NoGround_tozero->points.size(); i++)
                    {
                        Eigen::Vector4f Point(NoGround_tozero->points[i].x, NoGround_tozero->points[i].y, 0, 0);
                        Eigen::Vector4f PofL(coefficients->values[0], coefficients->values[1], 0, 0);
                        Eigen::Vector4f V(coefficients->values[3], coefficients->values[4], 0, 0);
                        double ddis = pcl::sqrPointToLineDistance(Point, PofL, V);
                        ddis = sqrt(ddis);
                        
                        p1.x = coefficients->values[0] - 5 * coefficients->values[3];
                        p1.y = coefficients->values[1] - 5 * coefficients->values[4];
                        p1.z = coefficients->values[2];
                        p2.x = coefficients->values[0] + 5 * coefficients->values[3];
                        p2.y = coefficients->values[1] + 5 * coefficients->values[4];
                        p2.z = coefficients->values[2];
                        if (ddis < save_thresh)
                        {
                            leftCloud->points.push_back(NoGround_tozero->points[i]);
                            leftCloud->width = leftCloud->points.size();
                            leftCloud->height = 1;
                            leftCloud->is_dense = true;
                            
                        }
                    }
                }
                
            }

            if (negCloud->points.size() > 0)
            {
                MyPointType minPt_neg, maxPt_neg;
                pcl::getMinMax3D (*negCloud, minPt_neg, maxPt_neg);

                pcl::SACSegmentation<MyPointType> seg;
                seg.setOptimizeCoefficients(true);
                seg.setInputCloud(negCloud);
                seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
                seg.setMethodType(pcl::SAC_RANSAC);
                const Eigen::Vector3f axis(1,0,0);
                const double eps = 5 * DEG_TO_RAD;
                seg.setAxis(axis);
                seg.setEpsAngle(eps);
                seg.setMaxIterations(1000);
                seg.setDistanceThreshold(0.10);
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                seg.segment(*inliers, *coefficients);

                Cloud_line_2[0] = coefficients->values[3];
                Cloud_line_2[1] = coefficients->values[4];
                
                Eigen::Vector4f PofR(coefficients->values[0], coefficients->values[1], 0, 0); 
                Eigen::Vector4f VofR(coefficients->values[3], coefficients->values[4], 0, 0); 

                std::sort(negCloud->points.begin(), negCloud->points.end(), compareX);
                Eigen::Vector4f Pt(0, 0, 0, 0);

                double distance1 = pcl::sqrPointToLineDistance(Pt, PofR, VofR);
                distance1 = sqrt(distance1);

                if (distance1 >= right_distance)
                {
                    for (size_t i = 0; i < negCloud->points.size(); i++)
                    {
                        Eigen::Vector4f Point(negCloud->points[i].x, negCloud->points[i].y, 0, 0);
                        double ddis = pcl::sqrPointToLineDistance(Point, PofR, VofR);
                        ddis = sqrt(ddis);

                        if (ddis > save_thresh)
                        {
                            negCloud_2nd->points.push_back(negCloud->points[i]);
                            negCloud_2nd->width = negCloud_2nd->points.size();
                            negCloud_2nd->height = 1;
                            negCloud_2nd->is_dense = true;
                        }
                    }

                    seg.setOptimizeCoefficients(true);
                    seg.setInputCloud(negCloud_2nd);
                    seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
                    seg.setMethodType(pcl::SAC_RANSAC);
                    seg.setAxis(axis);
                    seg.setEpsAngle(eps);
                    seg.setMaxIterations(1000);
                    seg.setDistanceThreshold(0.10);
                    
                    pcl::ModelCoefficients::Ptr coefficients_2(new pcl::ModelCoefficients);
                    pcl::PointIndices::Ptr inliers_2(new pcl::PointIndices);
                    seg.segment(*inliers, *coefficients_2);
                    PofR <<coefficients_2->values[0], coefficients_2->values[1], 0, 0;
                    VofR << coefficients_2->values[3], coefficients_2->values[4], 0, 0;

                    for (size_t i = 0; i < NoGround_tozero->points.size(); i++)
                    {
                        Eigen::Vector4f Point(NoGround_tozero->points[i].x, NoGround_tozero->points[i].y, 0, 0);
                        double ddis = pcl::sqrPointToLineDistance(Point, PofR, VofR);
                        ddis = sqrt(ddis);
                        p3.x = coefficients->values[0] - 5 * coefficients->values[3];
                        p3.y = coefficients->values[1] - 5 * coefficients->values[4];
                        p3.z = coefficients->values[2];
                        p4.x = coefficients->values[0] + 5 * coefficients->values[3];
                        p4.y = coefficients->values[1] + 5 * coefficients->values[4];
                        p4.z = coefficients->values[2];
                        if (ddis < save_thresh)
                        {
                            rightCloud->points.push_back(NoGround_tozero->points[i]);
                            rightCloud->width = rightCloud->points.size();
                            rightCloud->height = 1;
                            rightCloud->is_dense = true;
                        }
                    }
                }
                
                else if (distance1 < right_distance)
                {
                    for (size_t i = 0; i < NoGround_tozero->points.size(); i++)
                    {
                        Eigen::Vector4f Point(NoGround_tozero->points[i].x, NoGround_tozero->points[i].y, 0, 0);
                        double ddis = pcl::sqrPointToLineDistance(Point, PofR, VofR);
                        ddis = sqrt(ddis);
                        p3.x = coefficients->values[0] - 5 * coefficients->values[3];
                        p3.y = coefficients->values[1] - 5 * coefficients->values[4];
                        p3.z = coefficients->values[2];
                        p4.x = coefficients->values[0] + 5 * coefficients->values[3];
                        p4.y = coefficients->values[1] + 5 * coefficients->values[4];
                        p4.z = coefficients->values[2];
                        if (ddis < save_thresh)
                        {
                            rightCloud->points.push_back(NoGround_tozero->points[i]);
                            rightCloud->width = rightCloud->points.size();
                            rightCloud->height = 1;
                            rightCloud->is_dense = true;
                        }
                    }
                }
                 
            }
            suffix_name = "/leftCloud.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *leftCloud);

            suffix_name = "/rightCloud.pcd";
            savepath = prefix + output_file_path.c_str() + suffix_name;
            // writer.writeASCII(savepath, *rightCloud);
        }
        
        for (int m = 0; m < leftCloud->points.size(); m++)
        {
            leftCloud->points[m].x += center_x;
        }
        for (int m = 0; m < rightCloud->points.size(); m++)
        {
            rightCloud->points[m].x += center_x;
        }

        pcl::PointCloud<MyPointType>::Ptr leftCloud_view(new pcl::PointCloud<MyPointType>);
        pcl::PointCloud<MyPointType>::Ptr rightCloud_view(new pcl::PointCloud<MyPointType>);
        pcl::transformPointCloud(*leftCloud, *leftCloud_view, lidar_xy);
        pcl::transformPointCloud(*leftCloud_view, *leftCloud_view, transform_inverse);
        pcl::transformPointCloud(*leftCloud_view, *leftCloud_view, lidar_xy_inverse);
        pcl::transformPointCloud(*leftCloud_view, *leftCloud_view, lidar_installation_inverse);
        pcl::transformPointCloud(*leftCloud_view, *leftCloud_view, transformMatrix_l2c);

        std::vector<cv::Point2f> left_pixel;
        std::vector<cv::Point2f> right_pixel;
        for (pcl::PointCloud<MyPointType>::const_iterator it = leftCloud_view->begin(); it != leftCloud_view->end(); it++)
        {
            double tmpx = it->x / it->z;
            double tmpy = it->y / it->z;
            double tmpz = it->z;

            cv::Point2d points2image;
            double r2 = tmpx * tmpx + tmpy * tmpy;
            double r1 = pow(r2, 0.5);
            double a0 = std::atan(r1);
            double a1;
            a1 = a0 * (1 + distcoeff.at<double>(0) * pow(a0, 2) + distcoeff.at<double>(1) * pow(a0, 4) +
                        distcoeff.at<double>(2) * pow(a0, 6) + distcoeff.at<double>(3) * pow(a0, 8));
            points2image.x = (a1 / r1) * tmpx;
            points2image.y = (a1 / r1) * tmpy;

            points2image.x = cameramat.at<double>(0, 0) * points2image.x + cameramat.at<double>(0, 2);
            points2image.y = cameramat.at<double>(1, 1) * points2image.y + cameramat.at<double>(1, 2);

            int point_size = 3;
            if (points2image.y >= 500 and points2image.y < 980 and points2image.x >= 600 and points2image.x < 1320) 
            {
                cv::circle(brightened_image, cv::Point(points2image.x, points2image.y), point_size, CV_RGB(0, 0, 255), -1);
                left_pixel.push_back(points2image);
            }
        }

        pcl::transformPointCloud(*rightCloud, *rightCloud_view, lidar_xy);
        pcl::transformPointCloud(*rightCloud_view, *rightCloud_view, transform_inverse);
        pcl::transformPointCloud(*rightCloud_view, *rightCloud_view, lidar_xy_inverse);
        pcl::transformPointCloud(*rightCloud_view, *rightCloud_view, lidar_installation_inverse);
        pcl::transformPointCloud(*rightCloud_view, *rightCloud_view, transformMatrix_l2c);

        for (pcl::PointCloud<MyPointType>::const_iterator it = rightCloud_view->begin(); it != rightCloud_view->end(); it++)
        {
            double tmpx = it->x / it->z;
            double tmpy = it->y / it->z;
            double tmpz = it->z;

            cv::Point2d points2image;

            double r2 = tmpx * tmpx + tmpy * tmpy;
            double r1 = pow(r2, 0.5);
            double a0 = std::atan(r1);
            double a1;
            a1 = a0 * (1 + distcoeff.at<double>(0) * pow(a0, 2) + distcoeff.at<double>(1) * pow(a0, 4) +
                        distcoeff.at<double>(2) * pow(a0, 6) + distcoeff.at<double>(3) * pow(a0, 8));
            points2image.x = (a1 / r1) * tmpx;
            points2image.y = (a1 / r1) * tmpy;

            points2image.x = cameramat.at<double>(0, 0) * points2image.x + cameramat.at<double>(0, 2);
            points2image.y = cameramat.at<double>(1, 1) * points2image.y + cameramat.at<double>(1, 2);

            int point_size = 3;
            if (points2image.y >= 500 and points2image.y < 980 and points2image.x >= 600 and points2image.x < 1320) 
            {
                cv::circle(brightened_image, cv::Point(points2image.x, points2image.y), point_size, CV_RGB(0, 0, 255), -1);
                right_pixel.push_back(points2image);
            }
        }

        // 2.3 Fitting the polygon box
        std::vector<cv::Point2f> hull_left;
        cv::convexHull(left_pixel, hull_left);
        std::vector<cv::Point2i> hull_left_int;
        for (int i = 0; i < hull_left.size(); i++) 
        {
            hull_left_int.push_back(cv::Point2i(static_cast<int>(hull_left[i].x - 600), static_cast<int>(hull_left[i].y - 500)));
        }
        // cv::polylines(image_roi, hull_left_int, true, cv::Scalar(0, 255, 255), 2, LINE_AA, 0);

        cv::Mat mask_left(image_roi.size(), CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(mask_left, hull_left_int.data(), hull_left_int.size(), cv::Scalar(255));

        std::vector<cv::Point2f> hull_right;
        cv::convexHull(right_pixel, hull_right);
        std::vector<cv::Point2i> hull_right_int;
        for (int i = 0; i < hull_right.size(); i++) 
        {
            hull_right_int.push_back(cv::Point2i(static_cast<int>(hull_right[i].x -600), static_cast<int>(hull_right[i].y - 500)));
        }
        cv::polylines(image_roi, hull_right_int, true, cv::Scalar(0, 255, 255), 2, LINE_AA, 0);

        cv::Mat mask_right(image_roi.size(), CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(mask_right, hull_right_int.data(), hull_right_int.size(), cv::Scalar(255));

        for (auto& point : right_pixel) 
        {
            point.x -= 600;
            point.y -= 500;
        }
        for (auto& point : left_pixel) 
        {
            point.x -= 600;
            point.y -= 500;
        }

        Vec4f linePara_left, linePara_right; 
        fitLine(right_pixel, linePara_right, DIST_L2, 0, 0.01, 0.01);
        fitLine(left_pixel, linePara_left, DIST_L2, 0, 0.01, 0.01);

        int x1 = 0;
        int y1 = (-linePara_right[2] * linePara_right[1] / linePara_right[0]) + linePara_right[3];
        int x2 = image_roi_after.cols;
        int y2 = ((image_roi_after.cols - 1 - linePara_right[2]) * linePara_right[1] / linePara_right[0]) + linePara_right[3];
 
        int x3 = 0;
        int y3 = (-linePara_left[2] * linePara_left[1] / linePara_left[0]) + linePara_left[3];
        int x4 = image_roi_after.cols;
        int y4 = ((image_roi_after.cols - 1 - linePara_left[2]) * linePara_left[1] / linePara_left[0]) + linePara_left[3];

        cv::Mat out_left, out_right, out;
        S_out.copyTo(out_left, mask_left);
        S_out.copyTo(out_right, mask_right);
        out = out_left + out_right;

        vector<vector<Point>> contours_left;
        vector<Vec4i> hierarchy;
        findContours(out_left, contours_left, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat contourImage = Mat::zeros(out.size(), CV_8UC3);
        for (size_t i = 0; i < contours_left.size(); i++) 
        {
            if (contours_left[i].size() > 15)
            {
                Scalar color = Scalar(0, 255, 0); 
                drawContours(contourImage, contours_left, i, color, 1, LINE_8, hierarchy, 0);
            }
        }

        vector<Point> allcontoursPoints_left;
        for (size_t i = 0; i < contours_left.size(); i++) 
        {
            if (contours_left[i].size() >= 15) 
            { 
                allcontoursPoints_left.insert(allcontoursPoints_left.end(), contours_left[i].begin(), contours_left[i].end());
            }
        }

        Vec4f lineParams_left;
        fitLine(allcontoursPoints_left, lineParams_left, DIST_L2, 0, 0.01, 0.01);

        Point startPoint, endPoint;

        startPoint.x = (((-lineParams_left[3] / lineParams_left[1]) * lineParams_left[0] + lineParams_left[2]));
        startPoint.y = 0;
        endPoint.x = (((480 - lineParams_left[3]) / lineParams_left[1]) * lineParams_left[0] + lineParams_left[2]) ;
        endPoint.y = 480;
        
        Scalar color = Scalar(0, 0, 255); 
        line(contourImage, startPoint, endPoint, color, 2, LINE_AA);
        line(image_roi, startPoint, endPoint, color, 2, LINE_AA);

        vector<vector<Point>> contours_right;
        vector<Vec4i> hierarchy_right;
        findContours(out_right, contours_right, hierarchy_right, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (size_t i = 0; i < contours_right.size(); i++) 
        {
            if (contours_right[i].size() > 15)
            {
                Scalar color = Scalar(0, 255, 0); 
                drawContours(contourImage, contours_right, i, color, 1, LINE_8, hierarchy_right, 0);
            }
        }

        vector<Point> allcontoursPoints_right;
        for (size_t i = 0; i < contours_right.size(); i++) 
        {
            if (contours_right[i].size() >= 15) 
            { 
                allcontoursPoints_right.insert(allcontoursPoints_right.end(), contours_right[i].begin(), contours_right[i].end());
            }
        }

        Vec4f lineParams_right;
        fitLine(allcontoursPoints_right, lineParams_right, DIST_L2, 0, 0.01, 0.01);

        Point startPoint_r, endPoint_r;
        
        startPoint_r.x = (((-lineParams_right[3] / lineParams_right[1]) * lineParams_right[0] + lineParams_right[2]));
        startPoint_r.y = 0;
        endPoint_r.x = (((480 - lineParams_right[3]) / lineParams_right[1]) * lineParams_right[0] + lineParams_right[2]);
        endPoint_r.y = 480;

        color = Scalar(0, 0, 255); 
        line(contourImage, startPoint_r, endPoint_r, color, 2, LINE_AA);
        line(image_roi, startPoint_r, endPoint_r, color, 2, LINE_AA);
            
        // 3、Calculate the angle of the navigation line
        lineParams_left = handle.inverse(lineParams_left);
        lineParams_right = handle.inverse(lineParams_right);

        Vec4f lineParams_nav;
        lineParams_nav[0] = lineParams_left[0] + lineParams_right[0];
        lineParams_nav[1] = lineParams_left[1] + lineParams_right[1]; 
        lineParams_nav[2] = handle.LinesIntersection(lineParams_left,lineParams_right).first;
        lineParams_nav[3] = handle.LinesIntersection(lineParams_left,lineParams_right).second;

        // 4、Navigation line visualization
        Point startPoint_m, endPoint_m;
        startPoint_m.x = cvRound(lineParams_nav[2] - lineParams_nav[0] * 1000);
        startPoint_m.y = cvRound(lineParams_nav[3] - lineParams_nav[1] * 1000);
        endPoint_m.x = cvRound(lineParams_nav[2] + lineParams_nav[0] * 1000);
        endPoint_m.y = cvRound(lineParams_nav[3] + lineParams_nav[1] * 1000);

        Point nav_lidar_up, nav_lidar_down;
        nav_lidar_up.x = (((-linePara_left[3] / linePara_left[1]) * linePara_left[0] + linePara_left[2]) + ((-linePara_right[3] / linePara_right[1]) * linePara_right[0] + linePara_right[2])) / 2;
        nav_lidar_up.y = 0;
        nav_lidar_down.x = (((480 - linePara_left[3]) / linePara_left[1]) * linePara_left[0] + linePara_left[2] + ((480 - linePara_right[3]) / linePara_right[1]) * linePara_right[0] + linePara_right[2]) / 2;
        nav_lidar_down.y = 480;

        Point nav_fusion_up, nav_fusion_down;
        nav_fusion_up.x = (startPoint_r.x + startPoint.x) / 2;
        nav_fusion_up.y = 0;
        nav_fusion_down.x = (endPoint.x + endPoint_r.x) / 2;
        nav_fusion_down.y = 480;
        handle.draw_dotted_line2(image_roi, nav_fusion_up, nav_fusion_down, Scalar(0, 0, 255), 2);

        Vector2f line_left, line_right, line_nav, line_middle;
        line_left[0] = lineParams_left[0];
        line_left[1] = lineParams_left[1];
        line_right[0] = lineParams_right[0];
        line_right[1] = lineParams_right[1];
        line_nav[0] = lineParams_nav[0];
        line_nav[1] = lineParams_nav[1];
        line_middle[0] = 0;
        line_middle[1] = 1;

        Cloud_line_1[0] = x3-x4;
        Cloud_line_1[1] = y3-y4;
        Cloud_line_2[0] = x2-x1;
        Cloud_line_2[1] = y2-y1;
        float lidar_posi = handle.angleBetweenVectors(line_middle, Cloud_line_1); 
        float lidar_neg = handle.angleBetweenVectors(line_middle, Cloud_line_2);
        
        float angle_left = handle.angleBetweenVectors(line_middle, line_left);
        float angle_right = handle.angleBetweenVectors(line_middle, line_right);
        float angle_nav = handle.angleBetweenVectors(line_middle, line_nav);
        float navigation_angle = (angle_left + angle_right) / 2;
        endtime = clock(); 
        double useTime = (double)(endtime - starttime) / CLOCKS_PER_SEC;
        
        ofs << cal << " , " << angle_left << " , " << angle_nav << " , " << angle_right << " ! " << lidar_posi << " , " << (lidar_posi + lidar_neg) / 2 << " , "<< lidar_neg<< " , "<<useTime* 1000 << endl;

        // 5、View processing results online
        cout << "总耗时:" << useTime * 1000 << "ms" << endl;

        imgName = "/out_";
        suffix_name = ".jpg";
        savepath = prefix + output_file_path.c_str() + imgName + to_string(cal) + suffix_name;
        cv::imwrite(savepath, image_roi_after);
        if (look > 0)
        {
            if(look == 1)
            {
                cout << "查看图像image" << endl;

                imshow("image_roi_after Image", image_roi_after);
                cv::imwrite("../data/out/contourImage.jpg", contourImage);
                imshow("contourImage Image", contourImage);
                imshow("S_out Image", S_out);
                imshow("brightened_image Image", brightened_image);
                waitKey(0);
            }
            else if (look == 2)
            {
                cout <<"查看点云pcd" <<  endl;
                process.visual_cloud(NoGround_tozero, leftCloud, NoGround_tozero, rightCloud);
                
            }
        }
    }

    return 0;
}
