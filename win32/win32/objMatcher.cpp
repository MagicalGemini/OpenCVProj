#include"objMatcher.h"

#include <stdio.h>

#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/flann.hpp>
#include<opencv2/xfeatures2d.hpp>

#define SERIAL2FILE 0
#define ENABLE_TEST 0
#define SHOW_PROCESS_IMG 0
#define MATCH_CONF 0.4f

#if SHOW_PROCESS_IMG
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#endif

#define MIN_KEYPOINT_NUM 500
#define MIN_MATCH_POINT_NUM 10

using namespace cv;
using namespace xfeatures2d;

std::string saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors);
void loadKeyPoints(std::string& buff, std::vector<KeyPoint>& kps, Mat& descriptors);

const char* detectKeyPoints(char* buffer, int bufferSize/*, int algorithm*/)
{
	Mat rawData(1, bufferSize, CV_8UC1, buffer);
	Mat img = imdecode(rawData, IMREAD_GRAYSCALE);

	Ptr<Feature2D> detector = SIFT::create();

	std::vector<KeyPoint> keypointsVec;
	detector->detect(img, keypointsVec);
	std::string retData;

	if (keypointsVec.size() < MIN_KEYPOINT_NUM)
	{
		retData.append("0 | not enough key points: ");

		char buf[16] = { 0 };
		sprintf(buf, "%d", keypointsVec.size());
		retData.append(buf);
	}
	else
	{
		Mat descriptors;
		detector->compute(img, keypointsVec, descriptors);

		retData.append("1|");
		retData.append(saveKeyPoints(keypointsVec, descriptors));
	}

	const char* strData = retData.c_str();
	char* data = (char*)malloc(strlen(strData) + 1);
	strcpy(data, strData);

	return data;
}

int objMatchWithSerialData(char* imgBuffer, int imgBufferSize, char* serialBuffer, int serialBufferSize)
{
	std::vector<KeyPoint> oriKpVec, newImgKpVec;
	Mat oriDes, newImgDes;

	std::string serialData;
	serialData.append(serialBuffer, serialBufferSize);
	loadKeyPoints(serialData, oriKpVec, oriDes);

	Mat rawData(1, imgBufferSize, CV_8UC1, imgBuffer);

#if SHOW_PROCESS_IMG
	Mat image = imdecode(rawData, IMREAD_UNCHANGED);
	Mat newImg;
	cvtColor(image, newImg, COLOR_BGR2GRAY);

	Mat oriImage = imread("object.png", IMREAD_UNCHANGED);
#else
	Mat newImg = imdecode(rawData, IMREAD_GRAYSCALE);
#endif // SHOW_PROCESS_IMG

	Ptr<Feature2D> detector = SIFT::create();
	detector->detect(newImg, newImgKpVec);
	if (newImgKpVec.size() < MIN_KEYPOINT_NUM)
	{
		return 0;
	}

	detector->compute(newImg, newImgKpVec, newImgDes);

	Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>(5);
	Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);
	Ptr<cv::DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

	if (oriDes.type() != CV_32F)
	{
		oriDes.convertTo(oriDes, CV_32F);
	}
	if (newImgDes.type() != CV_32F)
	{
		newImgDes.convertTo(newImgDes, CV_32F);
	}

	std::vector<DMatch> matcheVec;
	std::vector< std::vector<DMatch> > pair_matches;
	matcher->knnMatch(oriDes, newImgDes, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m = pair_matches[i][0];
		const DMatch& n= pair_matches[i][1];
		if (m.distance < MATCH_CONF * n.distance)
		{
			matcheVec.push_back(m);
		}
	}
	
#if SHOW_PROCESS_IMG
	Mat outImg;
	drawMatches(oriImage, oriKpVec, image, newImgKpVec, matcheVec, outImg);
	imshow("matches", outImg);
#endif // SHOW_PROCESS_IMG

	if (matcheVec.size() >= MIN_MATCH_POINT_NUM)
	{
		return 1;
	}

	return 0;
}

std::string saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors)
{
#if SERIAL2FILE
	FileStorage fs("kps.xml", FileStorage::WRITE);
#else
	FileStorage fs("kps.xml", FileStorage::WRITE + FileStorage::MEMORY);
#endif

	fs << "kps" << kps << "des" << descriptors;
	cv::String cvStr = fs.releaseAndGetString();
	std::string retStr;
	retStr.append(cvStr.c_str(), cvStr.length());
	return retStr;
}

void loadKeyPoints(std::string& buff, std::vector<KeyPoint>& kps, Mat& descriptors)
{
#if SERIAL2FILE
	cv::FileStorage fs("kps.xml", FileStorage::READ);
#else
	cv::FileStorage fs(buff, FileStorage::READ + FileStorage::MEMORY);
#endif
	fs["kps"] >> kps;
	fs["des"] >> descriptors;
	fs.release();
}

#if ENABLE_TEST
void TestDetectKeyPoints()
{
	FILE* f = fopen("object.png", "rb");
	fseek(f, 0, SEEK_END);
	int dataSize = ftell(f);
	char* buffer = new char[dataSize];
	memset(buffer, 0, dataSize);

	fseek(f, 0, SEEK_SET);
	fread(buffer, 1, dataSize, f);
	fclose(f);

	const char* retData = detectKeyPoints(buffer, dataSize);
	free((void*)retData);
	delete[] buffer;
	buffer = nullptr;
}

void TestObjMatchWithSerialData()
{
	FILE* f = fopen("kps.xml", "rb");
	fseek(f, 0, SEEK_END);
	int serialBufferSize = ftell(f);
	char* serialBuffer = new char[serialBufferSize];
	memset(serialBuffer, 0, serialBufferSize);

	fseek(f, 0, SEEK_SET);
	fread(serialBuffer, 1, serialBufferSize, f);
	fclose(f);

	//f = fopen("scene5.png", "rb");
	f = fopen("pct_3.png", "rb");
	fseek(f, 0, SEEK_END);
	int imgBufferSize = ftell(f);
	char* imgBuffer = new char[imgBufferSize];
	memset(imgBuffer, 0, imgBufferSize);

	fseek(f, 0, SEEK_SET);
	fread(imgBuffer, 1, imgBufferSize, f);
	fclose(f);

	int ret = objMatchWithSerialData(imgBuffer, imgBufferSize, serialBuffer, serialBufferSize);
	printf("match result = %d", ret);
}

int main(int argc, char** argv)
{	
	//TestDetectKeyPoints();

	TestObjMatchWithSerialData();

#if SHOW_PROCESS_IMG
	waitKey();
#endif // SHOW_PROCESS_IMG

	
	return 0;
}
#endif//ENABLE_TEST