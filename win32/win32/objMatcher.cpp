#include"objMatcher.h"

#define SERIAL2FILE 0
#define ENABLE_TEST 0

#define MIN_KEYPOINT_NUM 1000
#define MIN_MATCH_POINT_NUM 20


std::string saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors);
void loadKeyPoints(std::string& buff, std::vector<KeyPoint>& kps, Mat& descriptors);

std::string detectKeyPoints(char* buffer, int bufferSize)
{
	Mat rawData(1, bufferSize, CV_8UC1, buffer);
	Mat img = imdecode(rawData, IMREAD_GRAYSCALE);
	Ptr<SIFT> detector = SIFT::create();
	std::vector<KeyPoint> keypointsVec;
	detector->detect(img, keypointsVec);
	std::string retData;
	if (keypointsVec.size() < MIN_KEYPOINT_NUM)
	{
		retData.append("0|not enough key points");
	}
	else
	{
		Mat descriptors;
		detector->compute(img, keypointsVec, descriptors);

		retData.append("1|");
		retData.append(saveKeyPoints(keypointsVec, descriptors));
	}

	return retData;
}

int objMatchWithSerialData(char* imgBuffer, int imgBufferSize, char* serialBuffer, int serialBufferSize)
{
	std::vector<KeyPoint> oriKpVec, newImgKpVec;
	Mat oriDes, newImgDes;

	std::string serialData;
	serialData.append(serialBuffer, serialBufferSize);
	loadKeyPoints(serialData, oriKpVec, oriDes);

	Mat rawData(1, imgBufferSize, CV_8UC1, imgBuffer);
	Mat newImg = imdecode(rawData, IMREAD_GRAYSCALE);

	Ptr<SIFT> detector = SIFT::create();
	detector->detectAndCompute(newImg, Mat(), newImgKpVec, newImgDes);

	Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>(5);
	Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);
	Ptr<cv::DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

	std::vector<DMatch> matcheVec;
	std::vector< std::vector<DMatch> > pair_matches;
	matcher->knnMatch(oriDes, newImgDes, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m = pair_matches[i][0];
		const DMatch& n= pair_matches[i][1];
		if (m.distance < 0.7f * n.distance)
		{
			matcheVec.push_back(m);
		}
	}
	
	if (matcheVec.size() >= MIN_MATCH_POINT_NUM)
	{
		return 1;
	}

	return 0;
}

std::string saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors)
{
#if SERIAL2FILE
	FileStorage fs("E:/kps.xml", FileStorage::WRITE_BASE64);
#else
	FileStorage fs("kps.xml", FileStorage::WRITE_BASE64 + FileStorage::MEMORY);
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
	cv::FileStorage fs("E:/kps.xml", FileStorage::READ);
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

	std::string retData = detectKeyPoints(buffer, dataSize);
	printf(retData.c_str());

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

	f = fopen("scene5.png", "rb");
	fseek(f, 0, SEEK_END);
	int imgBufferSize = ftell(f);
	char* imgBuffer = new char[imgBufferSize];
	memset(imgBuffer, 0, imgBufferSize);

	fseek(f, 0, SEEK_SET);
	fread(imgBuffer, 1, imgBufferSize, f);
	fclose(f);

	int ret = objMatchWithSerialData(imgBuffer, imgBufferSize, serialBuffer, serialBufferSize);
	printf("ret = %d", ret);
}

int main(int argc, char** argv)
{
	TestDetectKeyPoints();

	TestObjMatchWithSerialData();

	return 0;
}
#endif//ENABLE_TEST