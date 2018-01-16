#include <kinect2_registration/kinect2_color_registration.h>
#include <kinect2_registration/kinect2_console.h>

#ifdef DEPTH_REG_CPU
#include "color_registration_cpu.h"
#endif

#ifdef DEPTH_REG_OPENCL
#include "color_registration_cpu.h"
#endif

ColorRegistration::ColorRegistration()
{
}

ColorRegistration::~ColorRegistration()
{
}

bool ColorRegistration::init(const cv::Mat &cameraMatrixDepth, const cv::Size &sizeDepth, const cv::Mat &cameraMatrixColor, const cv::Size &sizeColor,
                             const cv::Mat &distortionColor, const cv::Mat &rotation, const cv::Mat &translation,
                             const float zNear, const float zFar, const int deviceId)
{
  this->cameraMatrixColor = cameraMatrixColor;
  this->cameraMatrixDepth = cameraMatrixDepth;
  this->rotation = rotation;
  this->translation = translation;
  this->sizeColor = sizeColor;
  this->sizeDepth = sizeDepth;
  this->zNear = zNear;
  this->zFar = zFar;


  cv::initUndistortRectifyMap(cameraMatrixColor, distortionColor, cv::Mat(), cameraMatrixDepth, sizeDepth, CV_32FC1, mapX, mapY);

  return init(deviceId);
}

ColorRegistration *ColorRegistration::New(Method method)
{
  if(method == DEFAULT)
  {
#ifdef DEPTH_REG_OPENCL
    method = CPU;
#elif defined DEPTH_REG_CPU
    method = CPU;
#endif
  }

  switch(method)
  {
  case DEFAULT:
    OUT_ERROR("No default registration method available!");
    break;
  case CPU:
#ifdef DEPTH_REG_CPU
    OUT_INFO("Using CPU registration method!");
    return new ColorRegistrationCPU();
#else
    OUT_ERROR("CPU registration method not available!");
    break;
#endif
  }
  return NULL;
}
