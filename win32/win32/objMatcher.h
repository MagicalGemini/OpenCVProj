#ifndef __OBJ_MATCHER_H__
#define __OBJ_MATCHER_H__

#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace xfeatures2d;

std::string detectKeyPoints(char* buffer, int bufferSize);

int objMatchWithSerialData(char* imgBuffer, int imgBufferSize, char* serialBuffer, int serialBufferSize);

std::string saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors);

void loadKeyPoints(std::string& buff, std::vector<KeyPoint>& kps, Mat& descriptors);

#endif//__OBJ_MATCHER_H__