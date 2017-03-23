/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <thread>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <std_msgs/Float64.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <kinect2_calibration/kinect2_calibration_definitions.h>
#include <kinect2_bridge/kinect2_definitions.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <tf/transform_broadcaster.h>
#include "rt_msgs/TransformRTStamped.h"

enum Source
{
	COLOR,
	IR,
	SYNC
};

class CalibBoardPoseEst
{
	typedef Eigen::Isometry3d RigidTransform;

private:
	const bool circleBoard;
	int circleFlags;

	const cv::Size boardDims;
	const double boardSize;
	const Source mode;

	const std::string path, calib_path, base_name_tf;
	const std::string topicColor, topicIr, topicDepth, rt_tf_topic;
	std::mutex lock;

	std::thread displayThread;

	bool running;
	bool update;
	bool foundColor, foundIr;
	cv::Mat color, ir, irGrey, depth;
	ros::Time colorStamp, irStamp, depthStamp;

	size_t frame;
	std::vector<int> params;

	std::vector<cv::Point3f> board;
	std::vector<cv::Point2f> pointsColor, pointsIr;

	RigidTransform K_T_C;
	bool doPlot;

	typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> ColorIrDepthSyncPolicy;
	ros::NodeHandle nh;
	ros::AsyncSpinner spinner;
	image_transport::ImageTransport it;
	image_transport::SubscriberFilter *subImageColor, *subImageIr, *subImageDepth;
	message_filters::Synchronizer<ColorIrDepthSyncPolicy> *sync;

	ros::Publisher rot3dfit_error_publisher, rt_tf_pub;

	int minIr, maxIr;
	cv::Ptr<cv::CLAHE> clahe;

	cv::Mat cameraMatrix, distortion;
	double fx, fy, cx, cy;

	tf::TransformBroadcaster broadcaster;

public:
	CalibBoardPoseEst(const std::string &path, const std::string &calib_path, const std::string &topicColor, const std::string &topicIr, const std::string &topicDepth, const Source mode, const bool circleBoard, const bool symmetric, const cv::Size &boardDims, const double boardSize, const std::string &base_name_tf, bool doPlot)
		: circleBoard(circleBoard), boardDims(boardDims), boardSize(boardSize), mode(mode), path(path), calib_path(calib_path), topicColor(topicColor), topicIr(topicIr),
		  topicDepth(topicDepth), update(false), foundColor(false), foundIr(false), frame(0), nh("~"), spinner(0), it(nh), minIr(0), maxIr(0x7FFF), base_name_tf(base_name_tf), running(true), doPlot(doPlot), rt_tf_topic(std::string("/trk_rt_tf"))
	{
		if(symmetric)
		{
			circleFlags = cv::CALIB_CB_SYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
		}
		else
		{
			circleFlags = cv::CALIB_CB_ASYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
		}

		params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		params.push_back(9);

		board.resize(boardDims.width * boardDims.height);
		for(size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
		{
			for(size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
			{
				board[i] = cv::Point3f(c * boardSize, r * boardSize, 0);
			}
		}

		clahe = cv::createCLAHE(1.5, cv::Size(32, 32));

		rot3dfit_error_publisher = nh.advertise<std_msgs::Float64>("/rot3dfit_error", 1);
		rt_tf_pub = nh.advertise<rt_msgs::TransformRTStamped>(rt_tf_topic, 1);
	}

	~CalibBoardPoseEst()
	{
	}

	void run()
	{
		startAcq();

		if (doPlot)
		{
			displayThread = std::thread(&CalibBoardPoseEst::display, this);
		}

		poseEstimation();

		stopAcq();
	}

private:
	void startAcq()
	{
		OUT_INFO("Controls:" << std::endl
				 << FG_YELLOW "   [ESC, q]" NO_COLOR " - Exit" << std::endl
				 << FG_YELLOW " [SPACE, s]" NO_COLOR " - Save current frame" << std::endl
				 << FG_YELLOW "        [l]" NO_COLOR " - decrease min and max value for IR value range" << std::endl
				 << FG_YELLOW "        [h]" NO_COLOR " - increase min and max value for IR value range" << std::endl
				 << FG_YELLOW "        [1]" NO_COLOR " - decrease min value for IR value range" << std::endl
				 << FG_YELLOW "        [2]" NO_COLOR " - increase min value for IR value range" << std::endl
				 << FG_YELLOW "        [3]" NO_COLOR " - decrease max value for IR value range" << std::endl
				 << FG_YELLOW "        [4]" NO_COLOR " - increase max value for IR value range");

		image_transport::TransportHints hints("compressed");
		subImageColor = new image_transport::SubscriberFilter(it, topicColor, 4, hints);
		subImageIr = new image_transport::SubscriberFilter(it, topicIr, 4, hints);
		subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, 4, hints);

		sync = new message_filters::Synchronizer<ColorIrDepthSyncPolicy>(ColorIrDepthSyncPolicy(4), *subImageColor, *subImageIr, *subImageDepth);
		sync->registerCallback(boost::bind(&CalibBoardPoseEst::callback, this, _1, _2, _3));

		spinner.start();

		bool ret = loadCalibration();

		if(ret)
		{
			fx = cameraMatrix.at<double>(0, 0);
			fy = cameraMatrix.at<double>(1, 1);
			cx = cameraMatrix.at<double>(0, 2);
			cy = cameraMatrix.at<double>(1, 2);
		}
	}

	void stopAcq()
	{
		if (doPlot)
		{
			displayThread.join();
		}
		spinner.stop();

		delete sync;
		delete subImageColor;
		delete subImageIr;
		delete subImageDepth;
	}

	void convertIr(const cv::Mat &ir, cv::Mat &grey)
	{
		const double factor = 255.0f / (maxIr - minIr);
		grey.create(ir.rows, ir.cols, CV_8U);

#pragma omp parallel for
		for(size_t r = 0; r < (size_t)ir.rows; ++r)
		{
			const uint16_t *itI = ir.ptr<uint16_t>(r);
			uint8_t *itO = grey.ptr<uint8_t>(r);

			for(size_t c = 0; c < (size_t)ir.cols; ++c, ++itI, ++itO)
			{
				*itO = std::min(std::max(*itI - minIr, 0) * factor, 255.0);
			}
		}
		clahe->apply(grey, grey);
	}

	void findMinMax(const cv::Mat &ir, const std::vector<cv::Point2f> &pointsIr)
	{
		minIr = 0xFFFF;
		maxIr = 0;
		for(size_t i = 0; i < pointsIr.size(); ++i)
		{
			const cv::Point2f &p = pointsIr[i];
			cv::Rect roi(std::max(0, (int)p.x - 2), std::max(0, (int)p.y - 2), 9, 9);
			roi.width = std::min(roi.width, ir.cols - roi.x);
			roi.height = std::min(roi.height, ir.rows - roi.y);

			findMinMax(ir(roi));
		}
	}

	void findMinMax(const cv::Mat &ir)
	{
		for(size_t r = 0; r < (size_t)ir.rows; ++r)
		{
			const uint16_t *it = ir.ptr<uint16_t>(r);

			for(size_t c = 0; c < (size_t)ir.cols; ++c, ++it)
			{
				minIr = std::min(minIr, (int) * it);
				maxIr = std::max(maxIr, (int) * it);
			}
		}
	}

	void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageIr, const sensor_msgs::Image::ConstPtr imageDepth)
	{
		std::vector<cv::Point2f> pointsColor, pointsIr;
		cv::Mat color, ir, irGrey, irScaled, depth;
		ros::Time colorStamp, irStamp, depthStamp;

		bool foundColor = false;
		bool foundIr = false;

		if(mode == COLOR || mode == SYNC)
		{
			readImage(imageColor, color);
			colorStamp = imageColor->header.stamp;
		}
		if(mode == IR || mode == SYNC)
		{
			readImage(imageIr, ir);
			readImage(imageDepth, depth);
			irStamp = imageIr->header.stamp;
			depthStamp = imageDepth->header.stamp;
			cv::resize(ir, irScaled, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

			convertIr(irScaled, irGrey);
		}

		if(circleBoard)
		{
			switch(mode)
			{
			case COLOR:
				foundColor = cv::findCirclesGrid(color, boardDims, pointsColor, circleFlags);
				break;
			case IR:
				foundIr = cv::findCirclesGrid(irGrey, boardDims, pointsIr, circleFlags);
				break;
			case SYNC:
				foundColor = cv::findCirclesGrid(color, boardDims, pointsColor, circleFlags);
				foundIr = cv::findCirclesGrid(irGrey, boardDims, pointsIr, circleFlags);
				break;
			}
		}
		else
		{
			const cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::COUNT, 100, DBL_EPSILON);
			switch(mode)
			{
			case COLOR:
				foundColor = cv::findChessboardCorners(color, boardDims, pointsColor, cv::CALIB_CB_FAST_CHECK);
				break;
			case IR:
				foundIr = cv::findChessboardCorners(irGrey, boardDims, pointsIr, cv::CALIB_CB_ADAPTIVE_THRESH);
				break;
			case SYNC:
				foundColor = cv::findChessboardCorners(color, boardDims, pointsColor, cv::CALIB_CB_FAST_CHECK);
				foundIr = cv::findChessboardCorners(irGrey, boardDims, pointsIr, cv::CALIB_CB_ADAPTIVE_THRESH);
				break;
			}
			if(foundColor)
			{
				cv::cornerSubPix(color, pointsColor, cv::Size(11, 11), cv::Size(-1, -1), termCriteria);
			}
			if(foundIr)
			{
				cv::cornerSubPix(irGrey, pointsIr, cv::Size(11, 11), cv::Size(-1, -1), termCriteria);
			}
		}

		if(foundIr)
		{
			// Update min and max ir value based on checkerboard values
			findMinMax(irScaled, pointsIr);
		}

		lock.lock();
		this->color = color;
		this->ir = ir;
		this->irGrey = irGrey;
		this->depth = depth;
		this->foundColor = foundColor;
		this->foundIr = foundIr;
		this->pointsColor = pointsColor;
		this->pointsIr = pointsIr;
		this->colorStamp = colorStamp;
		this->irStamp = irStamp;
		this->depthStamp = depthStamp;
		update = true;
		lock.unlock();
	}

	void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
	{
		double *itC = cameraMatrix.ptr<double>(0, 0);
		for(size_t i = 0; i < 9; ++i, ++itC)
		{
			*itC = cameraInfo->K[i];
		}
	}

	bool loadCalibration()
	{
		cv::FileStorage fs;

		if(fs.open(calib_path + K2_CALIB_IR, cv::FileStorage::READ))
		{
			fs[K2_CALIB_CAMERA_MATRIX] >> cameraMatrix;
			fs[K2_CALIB_DISTORTION] >> distortion;
			fs.release();
		}
		else
		{
			OUT_ERROR("couldn't read calibration '" << calib_path + K2_CALIB_IR << "'!");
			return false;
		}

		return true;
	}
	bool estimateCalBoardPose(const cv::Mat & cornersDepth, const std::vector<cv::Point2f> & pointsIr, RigidTransform & K_T_C, double & errRms)
	{
		int type = cornersDepth.type();
		std::vector<cv::Point3d> corners3D;
		int cnt = 0;
		std::vector<int> selCornerInds;
		for (int i = 0; i < boardDims.width*boardDims.height; i++)
		{
			double currCornerDepth = cornersDepth.at<float>(i,0);
			if ((currCornerDepth != 0) && !(std::isnan(currCornerDepth)))
			{
				currCornerDepth /= 1000.0;
				double x = (pointsIr[i].x/2.0 - cx)*currCornerDepth/fx;
				double y = (pointsIr[i].y/2.0 - cy)*currCornerDepth/fy;
				double z = currCornerDepth;
				cv::Point3d pt(x,y,z);
				corners3D.push_back(pt);
				selCornerInds.push_back(i);
				cnt = cnt + 1;
			}
		}
		bool visFlag;
		if (cnt < 3)
		{
			visFlag = false;
			return visFlag;
		}
		else
		{
			visFlag = true;
			Eigen::MatrixX3d corners3DMat, boardSelMat;
			corners3DMat.resize(cnt, Eigen::NoChange);
			boardSelMat.resize(cnt, Eigen::NoChange);
			for (int i = 0; i < cnt; i++)
			{
				corners3DMat(i, 0) = corners3D[i].x;
				corners3DMat(i, 1) = corners3D[i].y;
				corners3DMat(i, 2) = corners3D[i].z;

				boardSelMat(i, 0) = board[selCornerInds[i]].x;
				boardSelMat(i, 1) = board[selCornerInds[i]].y;
				boardSelMat(i, 2) = board[selCornerInds[i]].z;
			}
			errRms = rot3dfit(boardSelMat, corners3DMat, K_T_C);
			double det = K_T_C.matrix().topLeftCorner<3,3>().determinant();
			if (det<0)
			{
				OUT_WARN("rot3dfit returned LH basis!");
				visFlag = false;
			}
			return visFlag;
		}
	}

	double rot3dfit(const Eigen::MatrixX3d & X, const Eigen::MatrixX3d & Y, RigidTransform & T)
	{
		// mean correct
		Eigen::RowVector3d Xm = X.colwise().mean();
		Eigen::RowVector3d Ym = Y.colwise().mean();

		Eigen::MatrixX3d X1 = X - Eigen::MatrixXd::Ones(X.rows(), 1)*Xm;
		Eigen::MatrixX3d Y1 = Y - Eigen::MatrixXd::Ones(Y.rows(), 1)*Ym;
		// calculate best rotation using algorithm 12.4.1 from
		// G. H. Golub and C. F. van Loan, "Matrix Computations"
		// 2nd Edition, Baltimore: Johns Hopkins, 1989, p. 582.
		Eigen::Matrix3d XtY = (X1.transpose())*Y1;
		Eigen::JacobiSVD<Eigen::Matrix3d> svdXtY(XtY, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d R = svdXtY.matrixU() * svdXtY.matrixV().transpose();

		// solve for the translation vector
		Eigen::RowVector3d t = Ym - Xm*R;

		T.matrix().topLeftCorner<3,3>() = R.transpose();
		T.matrix().topRightCorner<3,1>() = t.transpose();
		T(3,0) = 0.; T(3,1) = 0.; T(3,2) = 0.; T(3,3) = 1.;

		// calculate fit points
		Eigen::MatrixX3d Yf = X*R + Eigen::MatrixXd::Ones(X.rows(), 1)*t;

		// calculate the error
		Eigen::MatrixX3d dY = Y - Yf;
		double errRMS = std::sqrt(dY.rowwise().squaredNorm().mean()); // RMS#
		return errRMS;
	}

	void poseEstimation()
	{
		int winSz = 10;
		std::vector<cv::Point2f> /*pointsColor, */pointsIr;
		std::vector< std::vector<cv::Point2f> > pointsIrBuffer(winSz, std::vector<cv::Point2f>(boardDims.width*boardDims.height));

		cv::Mat /*color, ir, irGrey,*/ depth, depthF;
		std::vector<cv::Mat> depthBuff(winSz, cv::Mat(424,512,CV_32FC1));
		ros::Time /*colorStamp, */irStamp, depthStamp;
		ros::Duration irDepthTimeDiff;
		cv::Mat cornersDepth(boardDims.width*boardDims.height, 1, CV_32FC1);
		cv::Mat mapPointsIrX(boardDims.width*boardDims.height, 1, CV_32FC1);
		cv::Mat mapPointsIrY(boardDims.width*boardDims.height, 1, CV_32FC1);
		// bool foundColor = false;
		bool foundIr = false;
		bool running = true;

		double errRms;

		bool visFlag;
		RigidTransform K_T_C;
		tf::StampedTransform mrkTf;
		tf::Matrix3x3 rot;
		tf::Vector3 trans;

		tf::Quaternion qZero;
		qZero.setRPY(0, 0, 0);
		tf::Vector3 vZero(0, 0, 0);
		tf::Transform tZero(qZero, vZero);
		mrkTf = tf::StampedTransform(tZero, ros::Time::now(), base_name_tf + K2_TF_IR_OPT_FRAME, base_name_tf + "_calboard");

		rt_msgs::TransformRTStamped rt_tf_msg;
		geometry_msgs::TransformStamped msgtf;

		std::chrono::milliseconds duration(1);

		while(!update && ros::ok())
		{
			std::this_thread::sleep_for(duration);
		}

		double fNaN = std::numeric_limits<float>::quiet_NaN();
		int winCnt = 0;
		bool filledOnce = false;
		for(; ros::ok() && running;)
		{
			lock.lock();
			running = this->running;
			lock.unlock();
			if(update)
			{
				if(mode == IR || mode == SYNC)
				{
					lock.lock();
					depth = this->depth;
					foundIr = this->foundIr;
					pointsIr = this->pointsIr;
					irStamp = this->irStamp;
					depthStamp = this->depthStamp;
					update = false;
					lock.unlock();

					if (foundIr)
					{
						pointsIrBuffer[winCnt] = pointsIr;
						depth.convertTo(depthF, CV_32FC1);
						depthF.setTo(fNaN, depthF == 0);
						depthBuff[winCnt] = depthF;
						if (filledOnce)
						{
							std::vector<cv::Point2f> pointsIrAvg(boardDims.width*boardDims.height, cv::Point2f(0.,0.));
							// averaging corners
							for (int k = 0; k < boardDims.width*boardDims.height; k++)
							{
								for (int i = 0; i < winSz; i++)
								{
									pointsIrAvg[k].x += pointsIrBuffer[i][k].x / winSz;
									pointsIrAvg[k].y += pointsIrBuffer[i][k].y / winSz;
								}
							}
							cv::Mat depthAvg(424, 512, CV_32FC1, cv::Scalar(0.0f));
							for (int i = 0; i < winSz; i++)
							{
								depthAvg += depthBuff[i] / winSz;
							}
							for (int i = 0; i < boardDims.width*boardDims.height; i++)
							{
								mapPointsIrX.at<float>(i,0) = pointsIrAvg[i].x/2.0;
								mapPointsIrY.at<float>(i,0) = pointsIrAvg[i].y/2.0;
							}
							cv::remap(depthAvg, cornersDepth, mapPointsIrX, mapPointsIrY, cv::INTER_NEAREST);
							visFlag = estimateCalBoardPose(cornersDepth, pointsIr, K_T_C, errRms);
							if (visFlag)
							{
								lock.lock();
								this->K_T_C = K_T_C;
								lock.unlock();
								rot.setValue(	K_T_C.matrix()(0,0), K_T_C.matrix()(0,1), K_T_C.matrix()(0,2),
												K_T_C.matrix()(1,0), K_T_C.matrix()(1,1), K_T_C.matrix()(1,2),
												K_T_C.matrix()(2,0), K_T_C.matrix()(2,1), K_T_C.matrix()(2,2)
												);
								trans.setX(K_T_C.matrix()(0,3)); trans.setY(K_T_C.matrix()(1,3)); trans.setZ(K_T_C.matrix()(2,3));
								irDepthTimeDiff = irStamp - depthStamp;
								if (irDepthTimeDiff.toSec() > 0.01)
									OUT_WARN("IR and depth frames' timestamps differ by " << irDepthTimeDiff.toSec() << " s.");
								mrkTf.stamp_ = depthStamp;
								mrkTf.setBasis(rot);
								mrkTf.setOrigin(trans);
								broadcaster.sendTransform(mrkTf);

								transformStampedTFToMsg(mrkTf, msgtf);
								// RT TF publish
								rt_tf_msg.rt_stamp = irStamp;
								rt_tf_msg.transform_stamped = msgtf;
								rt_tf_pub.publish(rt_tf_msg);

								std_msgs::Float64 err;
								err.data = errRms;
								rot3dfit_error_publisher.publish(err);
							}
						}
						if (!filledOnce && (winCnt == (winSz - 1)))
							filledOnce = true;

						winCnt = (winCnt + 1) % winSz;
					}
					//cv::cvtColor(irGrey, irDisp, CV_GRAY2BGR);
					//cv::drawChessboardCorners(irDisp, boardDims, pointsIr, foundIr);
					//cv::resize(irDisp, irDisp, cv::Size(), 0.5, 0.5);
					//cv::flip(irDisp, irDisp, 1);
				}

				//				if(mode == COLOR || mode == SYNC)
				//				{
				//					lock.lock();
				//					color = this->color;
				//					foundColor = this->foundColor;
				//					pointsColor = this->pointsColor;
				//					colorStamp = this->colorStamp;
				//					update = false;
				//					lock.unlock();

				//					cv::cvtColor(color, colorDisp, CV_GRAY2BGR);
				//					cv::drawChessboardCorners(colorDisp, boardDims, pointsColor, foundColor);
				//					//cv::resize(colorDisp, colorDisp, cv::Size(), 0.5, 0.5);
				//					//cv::flip(colorDisp, colorDisp, 1);
				//				}
			}
		}
	}

	void display()
	{
		std::vector<cv::Point2f> pointsColor, pointsIr, pointsIrReproj;
		cv::Mat color, ir, irGrey, depth;
		cv::Mat colorDisp, irDisp;
		bool foundColor = false;
		bool foundIr = false;
		bool save = false;
		bool running = true;
		RigidTransform K_T_C;

		std::chrono::milliseconds duration(1);

		while(!update && ros::ok())
		{
			std::this_thread::sleep_for(duration);
		}

		cv::Mat rvec, rotation, translation, distCoeffs;

		for(; ros::ok() && running;)
		{
			if(update)
			{
				if(mode == IR || mode == SYNC)
				{
					lock.lock();
					ir = this->ir;
					irGrey = this->irGrey;
					depth = this->depth;
					foundIr = this->foundIr;
					pointsIr = this->pointsIr;
					irStamp = this->irStamp;
					depthStamp = this->depthStamp;
					K_T_C = this->K_T_C;
					update = false;
					lock.unlock();
					Eigen::Matrix3d R = K_T_C.matrix().topLeftCorner<3,3>();
					Eigen::Vector3d t = K_T_C.matrix().topRightCorner<3,1>();
					eigen2cv(R,rotation);
					eigen2cv(t,translation);
					Eigen::Vector3d rvecEig, tvecEig;
					cv::Rodrigues(rotation, rvec);
					cv2eigen(rvec,rvecEig);
					cv2eigen(translation, tvecEig);
					cv::projectPoints(board, rvec, translation, cameraMatrix, distCoeffs, pointsIrReproj);
					for (int i = 0; i<boardDims.height*boardDims.width; i++)
					{
						pointsIrReproj[i].x = pointsIrReproj[i].x*2.;
						pointsIrReproj[i].y = pointsIrReproj[i].y*2.;
					}
					cv::cvtColor(irGrey, irDisp, CV_GRAY2BGR);
					cv::drawChessboardCorners(irDisp, boardDims, pointsIr, foundIr);
					cv::drawChessboardCorners(irDisp, boardDims, pointsIrReproj, foundIr);
					//cv::resize(irDisp, irDisp, cv::Size(), 0.5, 0.5);
					//cv::flip(irDisp, irDisp, 1);
				}

				if(mode == COLOR || mode == SYNC)
				{
					lock.lock();
					color = this->color;
					foundColor = this->foundColor;
					pointsColor = this->pointsColor;
					colorStamp = this->colorStamp;
					update = false;
					lock.unlock();
					cv::cvtColor(color, colorDisp, CV_GRAY2BGR);
					cv::drawChessboardCorners(colorDisp, boardDims, pointsColor, foundColor);
					//cv::resize(colorDisp, colorDisp, cv::Size(), 0.5, 0.5);
					//cv::flip(colorDisp, colorDisp, 1);
				}
			}

			switch(mode)
			{
			case COLOR:
				cv::imshow("color", colorDisp);
				break;
			case IR:
				cv::imshow("ir", irDisp);
				break;
			case SYNC:
				cv::imshow("color", colorDisp);
				cv::imshow("ir", irDisp);
				break;
			}

			int key = cv::waitKey(10);
			switch(key & 0xFF)
			{
			case ' ':
			case 's':
				save = true;
				break;
			case 27:
			case 'q':
				running = false;
				lock.lock();
				this->running = running;
				lock.unlock();
				break;
			case '1':
				minIr = std::max(0, minIr - 100);
				break;
			case '2':
				minIr = std::min(maxIr - 1, minIr + 100);
				break;
			case '3':
				maxIr = std::max(minIr + 1, maxIr - 100);
				break;
			case '4':
				maxIr = std::min(0xFFFF, maxIr + 100);
				break;
			case 'l':
				minIr = std::max(0, minIr - 100);
				maxIr = std::max(minIr + 1, maxIr - 100);
				break;
			case 'h':
				maxIr = std::min(0x7FFF, maxIr + 100);
				minIr = std::min(maxIr - 1, minIr + 100);
				break;
			}

			if(save && ((mode == COLOR && foundColor) || (mode == IR && foundIr) || (mode == SYNC && foundColor && foundIr)))
			{
				store(color, ir, irGrey, depth, pointsColor, pointsIr);
				save = false;
			}
		}
		cv::destroyAllWindows();
		cv::waitKey(100);
	}

	void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
	{
		cv_bridge::CvImageConstPtr pCvImage;
		pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
		pCvImage->image.copyTo(image);
	}

	void store(const cv::Mat &color, const cv::Mat &ir, const cv::Mat &irGrey, const cv::Mat &depth, const std::vector<cv::Point2f> &pointsColor, std::vector<cv::Point2f> &pointsIr)
	{
		std::ostringstream oss;
		oss << std::setfill('0') << std::setw(4) << frame++;
		const std::string frameNumber(oss.str());
		OUT_INFO("storing frame: " << frameNumber);
		std::string base = path + frameNumber;

		for(size_t i = 0; i < pointsIr.size(); ++i)
		{
			pointsIr[i].x /= 2.0;
			pointsIr[i].y /= 2.0;
		}

		if(mode == SYNC)
		{
			base += CALIB_SYNC;
		}

		if(mode == COLOR || mode == SYNC)
		{
			cv::imwrite(base + CALIB_FILE_COLOR, color, params);

			cv::FileStorage file(base + CALIB_POINTS_COLOR, cv::FileStorage::WRITE);
			file << "points" << pointsColor;
		}

		if(mode == IR || mode == SYNC)
		{
			cv::imwrite(base + CALIB_FILE_IR, ir, params);
			cv::imwrite(base + CALIB_FILE_IR_GREY, irGrey, params);
			cv::imwrite(base + CALIB_FILE_DEPTH, depth, params);

			cv::FileStorage file(base + CALIB_POINTS_IR, cv::FileStorage::WRITE);
			file << "points" << pointsIr;
		}
	}
};

void help(const std::string &path)
{
	std::cout << path << FG_BLUE " [options]" << std::endl
			  << FG_GREEN "  name" NO_COLOR ": " FG_YELLOW "'any string'" NO_COLOR " equals to the kinect2_bridge topic base name" << std::endl
			  << FG_GREEN "  source" NO_COLOR ": " FG_YELLOW "'color'" NO_COLOR ", " FG_YELLOW "'ir'" NO_COLOR ", " FG_YELLOW "'sync'" << std::endl
			  << FG_GREEN "  board" NO_COLOR ":" << std::endl
			  << FG_YELLOW "    'circle<WIDTH>x<HEIGHT>x<SIZE>'  " NO_COLOR "for symmetric circle grid" << std::endl
			  << FG_YELLOW "    'acircle<WIDTH>x<HEIGHT>x<SIZE>' " NO_COLOR "for asymmetric circle grid" << std::endl
			  << FG_YELLOW "    'chess<WIDTH>x<HEIGHT>x<SIZE>'   " NO_COLOR "for chessboard pattern" << std::endl
			  << FG_GREEN "  distortion model" NO_COLOR ": " FG_YELLOW "'rational'" NO_COLOR " for using model with 8 instead of 5 coefficients" << std::endl
			  << FG_GREEN "  output path" NO_COLOR ": " FG_YELLOW "'-path <PATH>'" NO_COLOR << std::endl
			  << FG_GREEN "  calib path" NO_COLOR ": " FG_YELLOW "'-calib_path <PATH>'" NO_COLOR << std::endl
			  << FG_GREEN "  base name for the tf" NO_COLOR ": " FG_YELLOW "'-base_name_tf <base name>'" NO_COLOR << std::endl
			  << FG_GREEN "  to show calib board detection" NO_COLOR ": " FG_YELLOW "'plot'" NO_COLOR << std::endl;
}

int main(int argc, char **argv)
{
#if EXTENDED_OUTPUT
	ROSCONSOLE_AUTOINIT;
	if(!getenv("ROSCONSOLE_FORMAT"))
	{
		ros::console::g_formatter.tokens_.clear();
		ros::console::g_formatter.init("[${severity}] ${message}");
	}
#endif

	Source source = IR;

	bool circleBoard = false;
	bool symmetric = true;
	cv::Size boardDims = cv::Size(8, 5);
	double boardSize = 0.04;
	std::string ns = K2_DEFAULT_NS;
	std::string path = "./";
	std::string calib_path = "/home/omer/catkin_ws/src/iai_kinect2/kinect2_bridge/data/502442443142/";
	std::string base_name_tf = std::string(K2_DEFAULT_NS);
	bool doPlot = false;

	ros::init(argc, argv, "kinect2_calib", ros::init_options::AnonymousName);

	if(!ros::ok())
	{
		return 0;
	}

	for(int argI = 1; argI < argc; ++ argI)
	{
		std::string arg(argv[argI]);

		if(arg == "--help" || arg == "--h" || arg == "-h" || arg == "-?" || arg == "--?")
		{
			help(argv[0]);
			ros::shutdown();
			return 0;
		}
		else if(arg == "color")
		{
			source = COLOR;
		}
		else if(arg == "ir")
		{
			source = IR;
		}
		else if(arg == "sync")
		{
			source = SYNC;
		}
		else if(arg == "plot")
		{
			doPlot = true;
		}
		else if(arg.find("circle") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
		{
			circleBoard = true;
			const size_t start = 6;
			const size_t leftX = arg.find('x');
			const size_t rightX = arg.rfind('x');
			const size_t end = arg.size();

			int width = atoi(arg.substr(start, leftX - start).c_str());
			int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
			boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
			boardDims = cv::Size(width, height);
		}
		else if((arg.find("circle") == 0 || arg.find("acircle") == 0) && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
		{
			symmetric = arg.find("circle") == 0;
			circleBoard = true;
			const size_t start = 6 + (symmetric ? 0 : 1);
			const size_t leftX = arg.find('x');
			const size_t rightX = arg.rfind('x');
			const size_t end = arg.size();

			int width = atoi(arg.substr(start, leftX - start).c_str());
			int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
			boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
			boardDims = cv::Size(width, height);
		}
		else if(arg.find("chess") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
		{
			circleBoard = false;
			const size_t start = 5;
			const size_t leftX = arg.find('x');
			const size_t rightX = arg.rfind('x');
			const size_t end = arg.size();

			int width = atoi(arg.substr(start, leftX - start).c_str());
			int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
			boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
			boardDims = cv::Size(width, height);
		}
		else if(arg == "-path" && ++argI < argc)
		{
			arg = argv[argI];
			struct stat fileStat;
			if(stat(arg.c_str(), &fileStat) == 0 && S_ISDIR(fileStat.st_mode))
			{
				path = arg;
			}
			else
			{
				OUT_ERROR("Unknown path: " << arg);
				help(argv[0]);
				ros::shutdown();
				return 0;
			}
		}
		else if(arg == "-calib_path" && ++argI < argc)
		{
			arg = argv[argI];
			struct stat fileStat;
			if(stat(arg.c_str(), &fileStat) == 0 && S_ISDIR(fileStat.st_mode))
			{
				calib_path = arg;
			}
			else
			{
				OUT_ERROR("Unknown calib_path: " << arg);
				help(argv[0]);
				ros::shutdown();
				return 0;
			}
		}
		else
		{
			ns = arg;
		}
	}

	std::string topicColor = "/" + ns + K2_TOPIC_HD + K2_TOPIC_IMAGE_MONO + K2_TOPIC_IMAGE_RECT;
	std::string topicIr = "/" + ns + K2_TOPIC_SD + K2_TOPIC_IMAGE_IR + K2_TOPIC_IMAGE_RECT;
	std::string topicDepth = "/" + ns + K2_TOPIC_SD + K2_TOPIC_IMAGE_DEPTH + K2_TOPIC_IMAGE_RECT;
	OUT_INFO("Start settings:" << std::endl
			 << "     Source: " FG_CYAN << (source == COLOR ? "color" : (source == IR ? "ir" : "sync")) << NO_COLOR << std::endl
			 << "      Board: " FG_CYAN << (circleBoard ? "circles" : "chess") << NO_COLOR << std::endl
			 << " Dimensions: " FG_CYAN << boardDims.width << " x " << boardDims.height << NO_COLOR << std::endl
			 << " Field size: " FG_CYAN << boardSize << NO_COLOR << std::endl
			 << "Topic color: " FG_CYAN << topicColor << NO_COLOR << std::endl
			 << "   Topic ir: " FG_CYAN << topicIr << NO_COLOR << std::endl
			 << "Topic depth: " FG_CYAN << topicDepth << NO_COLOR << std::endl
			 << "       Path: " FG_CYAN << path << NO_COLOR << std::endl
			 << "  CalibPath: " FG_CYAN << calib_path << NO_COLOR << std::endl
			 << " BaseNameTF: " FG_CYAN << base_name_tf << NO_COLOR << std::endl);

	if(!ros::master::check())
	{
		OUT_ERROR("checking ros master failed.");
		return -1;
	}

	CalibBoardPoseEst poseEst(path, calib_path, topicColor, topicIr, topicDepth, source, circleBoard, symmetric, boardDims, boardSize, base_name_tf, doPlot);

	OUT_INFO("starting CalibBoardPoseEst...");
	poseEst.run();

	OUT_INFO("stopped CalibBoardPoseEst...");


	return 0;
}
