/*
 * Copyright © 2019 AutonomouStuff, LLC
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies
 * or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef FLIR_BOSON_USB_BOSONCAMERA_H
#define FLIR_BOSON_USB_BOSONCAMERA_H

// C++ Includes
#include <string>
#include <thread> // NOLINT [build/c++11]
#include <mutex> // NOLINT [build/c++11]

// Linux system includes
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <asm/types.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <linux/videodev2.h>

// OpenCV Includes
#include <opencv2/opencv.hpp>

// ROS Includes
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Temperature.h>

// Dynamic Reconfigure
#include <dynamic_reconfigure/server.h>
#include "flir_boson_usb/BosonCameraConfig.h"

namespace flir_boson_usb
{

enum Encoding
{
  YUV = 0,
  RAW16 = 1
};

enum SensorTypes
{
  Boson320,
  Boson640
};

class BosonCamera : public nodelet::Nodelet
{
  public:
    BosonCamera();
    ~BosonCamera();

  private:
    virtual void onInit();
    void agcBasicLinear(const cv::Mat& input_16,
                        cv::Mat* output_8,
                        cv::Mat* output_16,
                        const int& height,
                        const int& width,
                        double* max_temp,
                        double* min_temp);
    bool openCamera();
    bool closeCamera();
    void captureAndPublish(const ros::TimerEvent& evt);

    ros::NodeHandle nh, pnh;
    std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info;
    std::shared_ptr<image_transport::ImageTransport> it;
    image_transport::CameraPublisher image_pub, image_pub_8, image_pub_8_norm, image_pub_heatmap, image_pub_temp;
    cv_bridge::CvImage cv_img;
    sensor_msgs::ImagePtr pub_image, pub_image_8, pub_image_8_norm, pub_image_heatmap, pub_image_temp;
    ros::Publisher max_temp_pub, min_temp_pub, ptr_temp_pub;
    sensor_msgs::Temperature max_temp_msg, min_temp_msg, ptr_temp_msg;
    ros::Timer capture_timer;
    int32_t width, height;
    double max_temp, min_temp, ptr_temp;
    int32_t fd;
    int32_t i;
    struct v4l2_capability cap;
    int32_t frame = 0;                // First frame number enumeration
    int8_t thermal_sensor_name[20];  // To store the sensor name
    struct v4l2_buffer bufferinfo;
    void* buffer_start;

    cv::Mat thermal16, thermal16_linear, thermal8_linear, thermal8_heatmap, thermal8_temp,
            thermal8_norm, thermal16_linear_zoom, thermal_rgb_zoom, thermal_luma, thermal_rgb;
    cv::Point temp_ptr;

    // Default Program options
    std::string frame_id, dev_path, camera_info_url,
      video_mode_str, sensor_type_str;
    float frame_rate;
    Encoding video_mode;
    bool zoom_enable;
    SensorTypes sensor_type;

    dynamic_reconfigure::Server<flir_boson_usb::BosonCameraConfig> reconfigure_server;
    std::mutex mutex;
    void reconfigureCallback(const flir_boson_usb::BosonCameraConfig& config);
    // Dynamic reconfigure parameters
    int point_x, point_y;
    int max_temp_limit, min_temp_limit;
    double norm_margin;
};

}  // namespace flir_boson_usb

#endif  // FLIR_BOSON_USB_BOSONCAMERA_H
