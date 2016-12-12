#ifndef __OBJ_MATCHER_H__
#define __OBJ_MATCHER_H__

#include <stdio.h>
#include <stdlib.h>

#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/flann.hpp>
#include<opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace xfeatures2d;

#ifdef WIN32
#define EXPORT_DLL extern "C" __declspec(dllexport)
#else
#define EXPORT_DLL
#endif

EXPORT_DLL std::string detectKeyPoints(char* buffer, int bufferSize);

EXPORT_DLL int objMatchWithSerialData(char* imgBuffer, int imgBufferSize, char* serialBuffer, int serialBufferSize);

#endif//__OBJ_MATCHER_H__