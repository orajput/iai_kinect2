#include "color_registration_cpu.h"
#include <kinect2_registration/kinect2_console.h>

ColorRegistrationCPU::ColorRegistrationCPU()
  : ColorRegistration()
{
}

ColorRegistrationCPU::~ColorRegistrationCPU()
{
}

bool ColorRegistrationCPU::init(const int deviceId)
{
  createLookup();

  proj(0, 0) = rotation.at<double>(0, 0);
  proj(0, 1) = rotation.at<double>(0, 1);
  proj(0, 2) = rotation.at<double>(0, 2);
  proj(0, 3) = translation.at<double>(0, 0);
  proj(1, 0) = rotation.at<double>(1, 0);
  proj(1, 1) = rotation.at<double>(1, 1);
  proj(1, 2) = rotation.at<double>(1, 2);
  proj(1, 3) = translation.at<double>(1, 0);
  proj(2, 0) = rotation.at<double>(2, 0);
  proj(2, 1) = rotation.at<double>(2, 1);
  proj(2, 2) = rotation.at<double>(2, 2);
  proj(2, 3) = translation.at<double>(2, 0);
  proj(3, 0) = 0;
  proj(3, 1) = 0;
  proj(3, 2) = 0;
  proj(3, 3) = 1;

  fx = cameraMatrixDepth.at<double>(0, 0);
  fy = cameraMatrixDepth.at<double>(1, 1);
  cx = cameraMatrixDepth.at<double>(0, 2) + 0.5;
  cy = cameraMatrixDepth.at<double>(1, 2) + 0.5;

  return true;
}

void ColorRegistrationCPU::createLookup()
{
  const double fx = 1.0 / cameraMatrixDepth.at<double>(0, 0);
  const double fy = 1.0 / cameraMatrixDepth.at<double>(1, 1);
  const double cx = cameraMatrixDepth.at<double>(0, 2);
  const double cy = cameraMatrixDepth.at<double>(1, 2);
  double *it;

  lookupY = cv::Mat(1, sizeDepth.height, CV_64F);
  it = lookupY.ptr<double>();
  for(size_t r = 0; r < (size_t)sizeDepth.height; ++r, ++it)
  {
    *it = (r - cy) * fy;
  }

  lookupX = cv::Mat(1, sizeDepth.width, CV_64F);
  it = lookupX.ptr<double>();
  for(size_t c = 0; c < (size_t)sizeDepth.width; ++c, ++it)
  {
    *it = (c - cx) * fx;
  }
}

void ColorRegistrationCPU::remapColor(const cv::Mat &color, cv::Mat &colorScaled) const
{
  colorScaled.create(sizeDepth, CV_8UC3);

  cv::remap(color, colorScaled, mapX, mapY, cv::INTER_LINEAR);
}

void ColorRegistrationCPU::projectColor(const cv::Mat& depth, const cv::Mat &colorScaled, cv::Mat &colorRegistered) const
{
    cv::Mat colorIdx = cv::Mat::zeros(sizeDepth, CV_16UC3);

    const double fxColor = cameraMatrixDepth.at<double>(0, 0);
    const double fyColor = cameraMatrixDepth.at<double>(1, 1);
    const double cxColor = cameraMatrixDepth.at<double>(0, 2) + 0.5;
    const double cyColor = cameraMatrixDepth.at<double>(1, 2) + 0.5;

    #pragma omp parallel for
    for(size_t r = 0; r < (size_t)sizeDepth.height; ++r)
    {
      const uint16_t *itD = depth.ptr<uint16_t>(r);
      const double y = lookupY.at<double>(0, r);
      const double *itX = lookupX.ptr<double>();

      for(size_t c = 0; c < (size_t)sizeDepth.width; ++c, ++itD, ++itX)
      {
        const double depthValue = *itD / 1000.0;

        if(depthValue < zNear || depthValue > zFar)
        {
          continue;
        }

        Eigen::Vector4d pointD(*itX * depthValue, y * depthValue, depthValue, 1);
        Eigen::Vector4d pointP = proj * pointD;

        const double z = pointP[2];

        const double invZ = 1 / z;
        const int xP = (fxColor * pointP[0]) * invZ + cxColor;
        const int yP = (fyColor * pointP[1]) * invZ + cyColor;

        if(xP >= 0 && xP < sizeDepth.width && yP >= 0 && yP < sizeDepth.height)
        {
          cv::Vec3b& cReg = colorRegistered.at<cv::Vec3b>(r, c);

          if (colorIdx.at<cv::Vec3s>(yP,xP)[2] == 0)
          {
            // no corresponding depth pixel assigned yet

            colorIdx.at<cv::Vec3s>(yP,xP)[0] = r;
            colorIdx.at<cv::Vec3s>(yP,xP)[1] = c;
            colorIdx.at<cv::Vec3s>(yP,xP)[2] = *itD;
            cReg = colorScaled.at<cv::Vec3b>(yP,xP);
          }
          else if (abs(*itD - colorIdx.at<cv::Vec3s>(yP,xP)[2]) < 50)
          {
            // pixel is within distance of 5 cm to currently assigned depth pixel
            // -> allow reuse of color pixel

            cReg = colorScaled.at<cv::Vec3b>(yP,xP);
          }
          else if (*itD < colorIdx.at<cv::Vec3s>(yP,xP)[2])
          {
            // pixel is closer to camera than currently assigned pixel
            // replace pixel and clear previously used pixel

              cReg = colorScaled.at<cv::Vec3b>(yP,xP);
              colorRegistered.at<cv::Vec3b>(colorIdx.at<cv::Vec3s>(yP,xP)[0],colorIdx.at<cv::Vec3s>(yP,xP)[1])[0] = 0;
              colorRegistered.at<cv::Vec3b>(colorIdx.at<cv::Vec3s>(yP,xP)[0],colorIdx.at<cv::Vec3s>(yP,xP)[1])[1] = 0;
              colorRegistered.at<cv::Vec3b>(colorIdx.at<cv::Vec3s>(yP,xP)[0],colorIdx.at<cv::Vec3s>(yP,xP)[1])[2] = 0;
              colorIdx.at<cv::Vec3s>(yP,xP)[0] = r;
              colorIdx.at<cv::Vec3s>(yP,xP)[1] = c;
              colorIdx.at<cv::Vec3s>(yP,xP)[2] = *itD;
          }

        }
      }
    }
}


bool ColorRegistrationCPU::registerColor(const cv::Mat &depth, const cv::Mat& color, cv::Mat& colorRegistered)
{
    colorRegistered = cv::Mat::zeros(sizeDepth, CV_8UC3);
    cv::Mat colorScaled = cv::Mat::zeros(sizeDepth, CV_8UC3);

    remapColor(color, colorScaled);
    projectColor(depth, colorScaled, colorRegistered);

    return true;
}
