/*
 * Copyright © 2019 AutonomouStuff, LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "flir_boson_usb/BosonCamera.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(flir_boson_usb::BosonCamera, nodelet::Nodelet)

using namespace cv;
using namespace flir_boson_usb;

BosonCamera::BosonCamera() : cv_img() {}

BosonCamera::~BosonCamera() { closeCamera(); }

void BosonCamera::onInit()
{
  nh = getNodeHandle();
  pnh = getPrivateNodeHandle();
  camera_info = std::shared_ptr<camera_info_manager::CameraInfoManager>(
      new camera_info_manager::CameraInfoManager(nh));
  it = std::shared_ptr<image_transport::ImageTransport>(
      new image_transport::ImageTransport(nh));
  image_pub = it->advertiseCamera("image_raw", 1);
  image_pub_8 = it->advertiseCamera("image8", 1);
  image_pub_8_norm = it->advertiseCamera("image8_norm", 1);
  image_pub_heatmap = it->advertiseCamera("image_heatmap", 1);
  image_pub_temp = it->advertiseCamera("image_temp", 1);
  max_temp_pub = nh.advertise<sensor_msgs::Temperature>("max_temp", 1);
  min_temp_pub = nh.advertise<sensor_msgs::Temperature>("min_temp", 1);
  ptr_temp_pub = nh.advertise<sensor_msgs::Temperature>("ptr_temp", 1);

  bool exit = false;
  reconfigure_server.setCallback(
      [this](flir_boson_usb::BosonCameraConfig &config, uint32_t /* level */)
      {
        reconfigureCallback(config);
      }); // NOLINT

  pnh.param<std::string>("frame_id", frame_id, "boson_camera");
  pnh.param<std::string>("dev", dev_path, "/dev/video0");
  pnh.param<float>("frame_rate", frame_rate, 60.0);
  pnh.param<std::string>("video_mode", video_mode_str, "RAW16");
  pnh.param<bool>("zoom_enable", zoom_enable, false);
  pnh.param<std::string>("sensor_type", sensor_type_str, "Boson_640");
  pnh.param<std::string>("camera_info_url", camera_info_url, "");

  ROS_INFO("flir_boson_usb - Got frame_id: %s.", frame_id.c_str());
  ROS_INFO("flir_boson_usb - Got dev: %s.", dev_path.c_str());
  ROS_INFO("flir_boson_usb - Got frame rate: %f.", frame_rate);
  ROS_INFO("flir_boson_usb - Got video mode: %s.", video_mode_str.c_str());
  ROS_INFO("flir_boson_usb - Got zoom enable: %s.",
           (zoom_enable ? "true" : "false"));
  ROS_INFO("flir_boson_usb - Got sensor type: %s.", sensor_type_str.c_str());
  ROS_INFO("flir_boson_usb - Got camera_info_url: %s.",
           camera_info_url.c_str());

  if (video_mode_str == "RAW16")
  {
    video_mode = RAW16;
  }
  else if (video_mode_str == "YUV")
  {
    video_mode = YUV;
  }
  else
  {
    exit = true;
    ROS_ERROR("flir_boson_usb - Invalid video_mode value provided. Exiting.");
  }

  if (sensor_type_str == "Boson_320" || sensor_type_str == "boson_320")
  {
    sensor_type = Boson320;
    camera_info->setCameraName("Boson320");
  }
  else if (sensor_type_str == "Boson_640" || sensor_type_str == "boson_640")
  {
    sensor_type = Boson640;
    camera_info->setCameraName("Boson640");
  }
  else
  {
    exit = true;
    ROS_ERROR("flir_boson_usb - Invalid sensor_type value provided. Exiting.");
  }

  if (camera_info->validateURL(camera_info_url))
  {
    camera_info->loadCameraInfo(camera_info_url);
  }
  else
  {
    ROS_INFO("flir_boson_usb - camera_info_url could not be validated. "
             "Publishing with unconfigured camera.");
  }

  if (!exit)
  {
    exit = openCamera() ? exit : true;
  }

  if (exit)
  {
    ros::shutdown();
    return;
  }
  else
  {
    capture_timer =
        nh.createTimer(ros::Duration(1.0 / frame_rate),
                       boost::bind(&BosonCamera::captureAndPublish, this, _1));
  }
}

// AGC Sample ONE: Linear from min to max.
// Input is a MATRIX (height x width) of 16bits. (OpenCV mat)
// Output is a MATRIX (height x width) of 8 bits (OpenCV mat)
void BosonCamera::agcBasicLinear(const Mat &input_16, Mat *output_8,
                                 Mat *output_16, const int &height,
                                 const int &width, double *max_temp,
                                 double *min_temp)
{
  int i, j;  // aux variables

  // auxiliary variables for AGC calcultion
  unsigned int max1 = 0;       // 16 bits
  unsigned int min1 = 0xFFFF;  // 16 bits
  unsigned int value1, value2, value3, value4, value5;

  // RUN a super basic AGC
  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      value1 = input_16.at<uchar>(i, j * 2 + 1) & 0xFF;  // High Byte
      value2 = input_16.at<uchar>(i, j * 2) & 0xFF;      // Low Byte
      value3 = (value1 << 8) + value2;

      if (value3 <= min1)
        min1 = value3;

      if (value3 >= max1)
        max1 = value3;
    }
  }
  *max_temp = max1 / 100. - 273.15;
  *min_temp = min1 / 100. - 273.15;

  if (max_temp_limit < min_temp_limit)
  {
    std::stringstream err_msg_ss;
    err_msg_ss << "max_temp_limit should be larger than min_temp_limit ";
    err_msg_ss << "(max_temp_limit: " << max_temp_limit
               << ", min_temp_limit: " << min_temp_limit << ")";
    throw std::range_error(err_msg_ss.str());
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    max1 = (max_temp_limit + 273.15) * 100;
    min1 = (min_temp_limit + 273.15) * 100;
  }

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      value1 = input_16.at<uchar>(i, j * 2 + 1) & 0xFF;  // High Byte
      value2 = input_16.at<uchar>(i, j * 2) & 0xFF;      // Low Byte
      value3 = (value1 << 8) + value2;
      value5 = value3;

      if (value5 > max1)
      {
        value5 = max1;
      }
      if (value5 < min1)
      {
        value5 = min1;
      }

      value4 = ((255 * (value5 - min1))) / (max1 - min1);
      output_8->at<uchar>(i, j) = static_cast<uint8_t>(value4);
      // output raw 16 bit image
      output_16->at<uint16_t>(i, j) = static_cast<uint16_t>(value3);
    }
  }
}

bool BosonCamera::openCamera()
{
  // Open the Video device
  if ((fd = open(dev_path.c_str(), O_RDWR)) < 0)
  {
    ROS_ERROR("flir_boson_usb - ERROR : OPEN. Invalid Video Device.");
    return false;
  }

  // Check VideoCapture mode is available
  if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0)
  {
    ROS_ERROR("flir_boson_usb - ERROR : VIDIOC_QUERYCAP. Video Capture is not "
              "available.");
    return false;
  }

  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
  {
    ROS_ERROR("flir_boson_usb - The device does not handle single-planar video "
              "capture.");
    return false;
  }

  struct v4l2_format format;

  // Two different FORMAT modes, 8 bits vs RAW16
  if (video_mode == RAW16)
  {
    // I am requiring thermal 16 bits mode
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_Y16;

    // Select the frame SIZE (will depend on the type of sensor)
    switch (sensor_type)
    {
    case Boson320:  // Boson320
      width = 320;
      height = 256;
      break;
    case Boson640:  // Boson640
      width = 640;
      height = 512;
      break;
    default:  // Boson320
      width = 320;
      height = 256;
      break;
    }
  }
  else  // 8- bits is always 640x512 (even for a Boson 320)
  {
    format.fmt.pix.pixelformat =
        V4L2_PIX_FMT_YVU420;  // thermal, works   LUMA, full Cr, full Cb
    width = 640;
    height = 512;
  }

  // Common varibles
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = width;
  format.fmt.pix.height = height;

  // request desired FORMAT
  if (ioctl(fd, VIDIOC_S_FMT, &format) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_S_FMT error. The camera does not "
              "support the requested video format.");
    return false;
  }

  // we need to inform the device about buffers to use.
  // and we need to allocate them.
  // we'll use a single buffer, and map our memory using mmap.
  // All this information is sent using the VIDIOC_REQBUFS call and a
  // v4l2_requestbuffers structure:
  struct v4l2_requestbuffers bufrequest;
  bufrequest.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  bufrequest.memory = V4L2_MEMORY_MMAP;
  bufrequest.count = 1;  // we are asking for one buffer

  if (ioctl(fd, VIDIOC_REQBUFS, &bufrequest) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_REQBUFS error. The camera failed to "
              "allocate a buffer.");
    return false;
  }

  // Now that the device knows how to provide its data,
  // we need to ask it about the amount of memory it needs,
  // and allocate it. This information is retrieved using the VIDIOC_QUERYBUF
  // call, and its v4l2_buffer structure.

  memset(&bufferinfo, 0, sizeof(bufferinfo));

  bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  bufferinfo.memory = V4L2_MEMORY_MMAP;
  bufferinfo.index = 0;

  if (ioctl(fd, VIDIOC_QUERYBUF, &bufferinfo) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_QUERYBUF error. Failed to retreive "
              "buffer information.");
    return false;
  }

  // map fd+offset into a process location (kernel will decide due to our NULL).
  // length and properties are also passed
  buffer_start = mmap(NULL, bufferinfo.length, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, bufferinfo.m.offset);

  if (buffer_start == MAP_FAILED)
  {
    ROS_ERROR("flir_boson_usb - mmap error. Failed to create a memory map for "
              "buffer.");
    return false;
  }

  // Fill this buffer with ceros. Initialization. Optional but nice to do
  memset(buffer_start, 0, bufferinfo.length);

  // Activate streaming
  int type = bufferinfo.type;
  if (ioctl(fd, VIDIOC_STREAMON, &type) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_STREAMON error. Failed to activate "
              "streaming on the camera.");
    return false;
  }

  // Declarations for RAW16 representation
  // Will be used in case we are reading RAW16 format
  // Boson320 , Boson 640
  // OpenCV input buffer  : Asking for all info: two bytes per pixel (RAW16)
  // RAW16 mode`
  thermal16 = Mat(height, width, CV_16U, buffer_start);
  // OpenCV output buffer : Data used to display the video
  thermal8_linear = Mat(height, width, CV_8U, 1);
  thermal8_norm = Mat(height, width, CV_8U, 1);
  thermal8_heatmap = Mat(height, width, CV_8UC3, 1);
  thermal16_linear = Mat(height, width, CV_16U, 1);

  // Declarations for 8bits YCbCr mode
  // Will be used in case we are reading YUV format
  // Boson320, 640 :  4:2:0
  int luma_height = height + height / 2;
  int luma_width = width;
  int color_space = CV_8UC1;

  // Declarations for Zoom representation
  // Will be used or not depending on program arguments
  thermal_luma = Mat(luma_height, luma_width, color_space,
                     buffer_start);  // OpenCV input buffer
  // OpenCV output buffer , BGR -> Three color spaces :
  // (640 - 640 - 640 : p11 p21 p31 .... / p12 p22 p32 ..../ p13 p23 p33 ...)
  thermal_rgb = Mat(height, width, CV_8UC3, 1);

  return true;
}

bool BosonCamera::closeCamera()
{
  // Finish loop. Exiting.
  // Deactivate streaming
  int type = bufferinfo.type;
  if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_STREAMOFF error. Failed to disable "
              "streaming on the camera.");
    return false;
  };

  close(fd);

  return true;
}

void BosonCamera::captureAndPublish(const ros::TimerEvent &evt)
{
  Size size(640, 512);

  sensor_msgs::CameraInfoPtr ci(
      new sensor_msgs::CameraInfo(camera_info->getCameraInfo()));

  ci->header.frame_id = frame_id;

  // Put the buffer in the incoming queue.
  if (ioctl(fd, VIDIOC_QBUF, &bufferinfo) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_QBUF error. Failed to queue the image "
              "buffer.");
    return;
  }

  // The buffer's waiting in the outgoing queue.
  if (ioctl(fd, VIDIOC_DQBUF, &bufferinfo) < 0)
  {
    ROS_ERROR("flir_boson_usb - VIDIOC_DQBUF error. Failed to dequeue the "
              "image buffer.");
    return;
  }

  if (video_mode == RAW16)
  {
    // -----------------------------
    // RAW16 DATA
    try
    {
      agcBasicLinear(thermal16, &thermal8_linear, &thermal16_linear, height,
                     width, &max_temp, &min_temp);
    }
    catch (std::range_error &e)
    {
      ROS_ERROR_THROTTLE(1, "%s", e.what());
    }

    // Display thermal after 16-bits AGC... will display an image
    if (!zoom_enable)
    {
      bool use_filter = false;
      if (use_filter)
      {
        // Threshold using Otsu's method, then use the result as a mask on the
        // original image
        Mat mask_mat, masked_img;
        threshold(thermal8_linear, mask_mat, 0, 255,
                  CV_THRESH_BINARY | CV_THRESH_OTSU);
        thermal8_linear.copyTo(masked_img, mask_mat);

        // Normalize the pixel values to the range [0, 1] then raise to power
        // (gamma). Then convert back for display.
        Mat d_out_img, norm_image, d_norm_image, gamma_corrected_image,
            d_gamma_corrected_image;
        double gamma = 0.8;
        masked_img.convertTo(d_out_img, CV_64FC1);
        normalize(d_out_img, d_norm_image, 0, 1, NORM_MINMAX, CV_64FC1);
        pow(d_out_img, gamma, d_gamma_corrected_image);
        d_gamma_corrected_image.convertTo(gamma_corrected_image, CV_8UC1);
        normalize(gamma_corrected_image, gamma_corrected_image, 0, 255,
                  NORM_MINMAX, CV_8UC1);

        // Apply top hat filter
        int erosion_size = 5;
        Mat top_hat_img, kernel = getStructuringElement(
                             MORPH_ELLIPSE,
                             Size(2 * erosion_size + 1, 2 * erosion_size + 1));
        morphologyEx(gamma_corrected_image, top_hat_img, MORPH_TOPHAT, kernel);
      }

      ros::Time now = ros::Time::now();

      // 16bit image
      cv_img.image = thermal16_linear;
      cv_img.header.stamp = now;
      cv_img.header.frame_id = frame_id;
      cv_img.encoding = "16UC1";
      pub_image = cv_img.toImageMsg();

      ci->header.stamp = pub_image->header.stamp;
      image_pub.publish(pub_image, ci);

      // 8bit image
      cv_img.image = thermal8_linear;
      cv_img.header.stamp = now;
      cv_img.header.frame_id = frame_id;
      cv_img.encoding = "mono8";
      pub_image_8 = cv_img.toImageMsg();

      ci->header.stamp = pub_image_8->header.stamp;
      image_pub_8.publish(pub_image_8, ci);

      // 8bit image (auto range)
      double min, max;
      minMaxLoc(thermal8_linear, &min, &max);
      double min_threshold = std::min(min - norm_margin, 0.0);
      double max_threshold = std::max(max + norm_margin, 255.0);
      if ((max_threshold - min_threshold) != 0)
      {
        thermal8_norm = (thermal8_linear - min_threshold) * (255 - 0) / (max_threshold - min_threshold);
      }
      else
      {
        thermal8_norm = thermal8_linear;
      }
      cv_img.image = thermal8_norm;
      cv_img.header.stamp = now;
      cv_img.header.frame_id = frame_id;
      cv_img.encoding = "mono8";
      pub_image_8_norm = cv_img.toImageMsg();

      ci->header.stamp = pub_image_8_norm->header.stamp;
      image_pub_8_norm.publish(pub_image_8_norm, ci);

      cv::applyColorMap(thermal8_linear, thermal8_heatmap, cv::COLORMAP_JET);
      // 8bit heatmap image
      cv_img.image = thermal8_heatmap;
      cv_img.header.stamp = now;
      cv_img.header.frame_id = frame_id;
      cv_img.encoding = "bgr8";
      pub_image_heatmap = cv_img.toImageMsg();

      ci->header.stamp = pub_image_heatmap->header.stamp;
      image_pub_heatmap.publish(pub_image_heatmap, ci);

      // 8bit heatmap image with temperature info
      thermal8_temp = thermal8_heatmap.clone();
      std::stringstream max_temp_ss, min_temp_ss, ptr_temp_ss;
      max_temp_ss << std::fixed << std::setprecision(2) << max_temp;
      min_temp_ss << std::fixed << std::setprecision(2) << min_temp;

      std::string disp_max_temp = "Max: " + max_temp_ss.str() + " deg";
      std::string disp_min_temp = "Min: " + min_temp_ss.str() + " deg";
      cv::putText(thermal8_temp, disp_max_temp, cv::Point(15, 15),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
      cv::putText(thermal8_temp, disp_min_temp, cv::Point(15, 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

      // pointer temperature
      {
        std::lock_guard<std::mutex> lock(mutex);
        temp_ptr = cv::Point(point_x, point_y);
        ptr_temp =
            thermal16_linear.at<uint16_t>(point_y, point_x) / 100.0 - 273.15;
      }
      ptr_temp_ss << std::fixed << std::setprecision(2) << ptr_temp;
      std::string disp_ptr_temp = "Ptr: " + ptr_temp_ss.str() + " deg";
      cv::putText(thermal8_temp, disp_ptr_temp, cv::Point(15, 45),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
      cv::circle(thermal8_temp, temp_ptr, 3, cv::Scalar(0, 0, 0), 1,
                 cv::LINE_AA);
      cv::circle(thermal8_temp, temp_ptr, 2, cv::Scalar(255, 255, 255), -1,
                 cv::LINE_AA);

      cv_img.image = thermal8_temp;
      cv_img.header.stamp = now;
      cv_img.header.frame_id = frame_id;
      cv_img.encoding = "bgr8";
      pub_image_temp = cv_img.toImageMsg();

      ci->header.stamp = pub_image_temp->header.stamp;
      image_pub_temp.publish(pub_image_temp, ci);

      max_temp_msg.header.stamp = now;
      min_temp_msg.header.stamp = now;
      ptr_temp_msg.header.stamp = now;
      max_temp_msg.temperature = max_temp;
      min_temp_msg.temperature = min_temp;
      ptr_temp_msg.temperature = ptr_temp;

      max_temp_pub.publish(max_temp_msg);
      min_temp_pub.publish(min_temp_msg);
      ptr_temp_pub.publish(ptr_temp_msg);
    }
    else
    {
      resize(thermal16_linear, thermal16_linear_zoom, size);

      cv_img.image = thermal16_linear_zoom;
      cv_img.header.stamp = ros::Time::now();
      cv_img.header.frame_id = frame_id;
      cv_img.encoding = "mono8";
      pub_image = cv_img.toImageMsg();

      ci->header.stamp = pub_image->header.stamp;
      image_pub.publish(pub_image, ci);
    }
  }
  else  // Video is in 8 bits YUV
  {
    // ---------------------------------
    // DATA in YUV
    cvtColor(thermal_luma, thermal_rgb, COLOR_YUV2GRAY_I420, 0);

    cv_img.image = thermal_rgb;
    cv_img.encoding = "mono8";
    cv_img.header.stamp = ros::Time::now();
    cv_img.header.frame_id = frame_id;
    pub_image = cv_img.toImageMsg();

    ci->header.stamp = pub_image->header.stamp;
    image_pub.publish(pub_image, ci);
  }
}

void BosonCamera::reconfigureCallback(
    const flir_boson_usb::BosonCameraConfig &config)
{
  std::lock_guard<std::mutex> lock(mutex);
  point_x = config.point_x;
  point_y = config.point_y;
  max_temp_limit = config.max_temp_limit;
  min_temp_limit = config.min_temp_limit;
  norm_margin = config.norm_margin;
}
