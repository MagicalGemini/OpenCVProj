#ifndef __OBJ_MATCHER_H__
#define __OBJ_MATCHER_H__

#include <stdio.h>
#include <stdlib.h>

#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/flann.hpp>
#include<opencv2/xfeatures2d.hpp>

#ifdef WIN32
#define EXPORT_DLL __declspec(dllexport)
#else
#define EXPORT_DLL
#endif

#ifdef __cplusplus
extern "C" 
{
#endif

	using namespace cv;
	using namespace xfeatures2d;

	EXPORT_DLL std::string detectKeyPoints(char* buffer, int bufferSize);

	EXPORT_DLL int objMatchWithSerialData(char* imgBuffer, int imgBufferSize, char* serialBuffer, int serialBufferSize);

#ifdef __cplusplus
}//__cplusplus
#endif

#endif//__OBJ_MATCHER_H__

