#pragma once
#ifndef __KINECT2_COLOR_REGISTRATION_H__
#define __KINECT2_COLOR_REGISTRATION_H__

#include <vector>

#include <opencv2/opencv.hpp>

class ColorRegistration
{
public:
  enum Method
  {
    DEFAULT = 0,
    CPU
  };

protected:
  cv::Mat cameraMatrixColor, cameraMatrixDepth, rotation, translation, mapX, mapY;
  cv::Size sizeColor, sizeDepth;
  float zNear, zFar;

  ColorRegistration();

  virtual bool init(const int deviceId) = 0;

public:
  virtual ~ColorRegistration();

  bool init(const cv::Mat &cameraMatrixDepth, const cv::Size &sizeDepth, const cv::Mat &cameraMatrixColor, const cv::Size &sizeColor,
            const cv::Mat &distortionColor, const cv::Mat &rotation, const cv::Mat &translation,
            const float zNear = 0.5f, const float zFar = 12.0f, const int deviceId = -1);

  virtual bool registerColor(const cv::Mat &depth, const cv::Mat& color, cv::Mat& colorRegistered) = 0;

  static ColorRegistration *New(Method method = DEFAULT);
};

#endif //__KINECT2_COLOR_REGISTRATION_H__
