#pragma once
#ifndef __COLOR_REGISTRATION_CPU_H__
#define __COLOR_REGISTRATION_CPU_H__

#include <Eigen/Geometry>

#include <kinect2_registration/kinect2_color_registration.h>

class ColorRegistrationCPU : public ColorRegistration
{
private:
  cv::Mat lookupX, lookupY;
  Eigen::Matrix4d proj;
  double fx, fy, cx, cy;

public:
  ColorRegistrationCPU();

  ~ColorRegistrationCPU();

  bool init(const int deviceId);

  bool registerColor(const cv::Mat &depth, const cv::Mat& color, cv::Mat& colorRegistered) override;

private:
  void createLookup();
  void remapColor(const cv::Mat &color, cv::Mat &colorScaled) const;
  void projectColor(const cv::Mat& depth, const cv::Mat &colorScaled, cv::Mat &colorRegistered) const;
};

#endif //__COLOR_REGISTRATION_CPU_H__
