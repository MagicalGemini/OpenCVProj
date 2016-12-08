#ifndef __OBJ_MATCHER_H__
#define __OBJ_MATCHER_H__

#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace xfeatures2d;

std::string& detectKeyPoints(Mat& img);

int objMatchWithSerialData(Mat& newImg, std::string& serialData);

const char* saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors);

void loadKeyPoints(std::string& buff, std::vector<KeyPoint>& kps, Mat& descriptors);

#endif//__OBJ_MATCHER_H__