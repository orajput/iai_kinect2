#include "color_registration_cpu.h"
#include <kinect2_registration/kinect2_console.h>

inline bool isWithinMat(int x, int y, const cv::Mat& image)
{
  return (x >= 0 && x < image.cols && y >= 0 && y < image.rows);
}

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

void ColorRegistrationCPU::projectColor(const cv::Mat& depth, const cv::Mat &color, cv::Mat &colorRegistered) const
{
  const double fxColor = cameraMatrixColor.at<double>(0, 0);
  const double fyColor = cameraMatrixColor.at<double>(1, 1);
  const double cxColor = cameraMatrixColor.at<double>(0, 2) + 0.5;
  const double cyColor = cameraMatrixColor.at<double>(1, 2) + 0.5;

  // depth filter size in x direction
  const size_t filter_width = 5;

  // depth filter size in y direction
  const size_t filter_height = 3;

  // relative depth tolerance for duplicate pixels
  const float filterTolerance = 0.03;

  // holds corresponding color pixel coordinate in depth image (x)
  // initialized with invalid pixel cordinate
  cv::Mat depthToColorPixelMapX = cv::Mat(sizeDepth, CV_16UC1, cv::Scalar(std::numeric_limits<unsigned short>::max()));

  // holds corresponding color pixel coordinate in depth image (x)
  // initialized with invalid pixel cordinate
  cv::Mat depthToColorPixelMapY = cv::Mat(sizeDepth, CV_16UC1, cv::Scalar(std::numeric_limits<unsigned short>::max()));

  // holds minimal depth for pixels in neighbood for color image
  cv::Mat minimalDepthMap = cv::Mat(sizeColor, CV_16UC1, cv::Scalar(std::numeric_limits<unsigned short>::max()));

  // collect depth to color pixel mapping
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

      // transform points from depth frame to color frame
      Eigen::Vector4d pointD(*itX * depthValue, y * depthValue, depthValue, 1);
      Eigen::Vector4d pointP = proj * pointD;
      const double z = pointP[2];
      const double invZ = 1 / z;

      // calculate color pixel coordinates from projection onto image plane
      const int xColor = (fxColor * pointP[0]) * invZ + cxColor;
      const int yColor = (fyColor * pointP[1]) * invZ + cyColor;

      if (isWithinMat(xColor, yColor, color))
      {
        // store minimal depth for color pixel
        if ((*itD < minimalDepthMap.at<uint16_t>(yColor,xColor)))
        {
            minimalDepthMap.at<uint16_t>(yColor, xColor) = *itD;
        }

        // keep track of which color pixel is mapped to which depth pixel
        depthToColorPixelMapX.at<uint16_t>(r,c) = xColor;
        depthToColorPixelMapY.at<uint16_t>(r,c) = yColor;
      }
    }
  }

  // minimum filter: spread minimal depth values in neighborhood
  cv::Mat element(filter_height, filter_width, CV_16UC1,cv::Scalar(1));
  cv::erode(minimalDepthMap, minimalDepthMap, element);

  // filter duplicate depth values to avoid ghosting/shadow effects in background regions
  // -> only closest point (in terms of depth) and neighbors within a tolerance are kept
  for(size_t r = 0; r < (size_t)sizeDepth.height; ++r)
  {
    const uint16_t *itD = depth.ptr<uint16_t>(r);

    for(size_t c = 0; c < (size_t)sizeDepth.width; ++c, ++itD)
    {
      const int depthToColorPixelX = depthToColorPixelMapX.at<uint16_t>(r,c);
      const int depthToColorPixelY = depthToColorPixelMapY.at<uint16_t>(r,c);

      if (isWithinMat(depthToColorPixelX, depthToColorPixelY, color) && (*itD > 0))
      {
        const uint16_t minDepth = minimalDepthMap.at<uint16_t>(depthToColorPixelY, depthToColorPixelX);
        const uint16_t depth = *itD;

        float depth_error = ((float)(depth - minDepth) / (float)depth);

        // allow duplicate pixels within a relative depth tolerance
        if ((depth_error < filterTolerance))
        {
          colorRegistered.at<cv::Vec3b>(r, c) = color.at<cv::Vec3b>(depthToColorPixelY, depthToColorPixelX);
        }
      }
    }
  }
}


bool ColorRegistrationCPU::registerColor(const cv::Mat &depth, const cv::Mat& color, cv::Mat& colorRegistered)
{
  colorRegistered = cv::Mat::zeros(sizeDepth, CV_8UC3);

  projectColor(depth, color, colorRegistered);

  return true;
}
