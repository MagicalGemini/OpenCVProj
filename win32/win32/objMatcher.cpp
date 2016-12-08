#include "stdafx.h"
#include"objMatcher.h"


std::string& detectKeyPoints(Mat& img)
{
	Ptr<SIFT> detector = SIFT::create();
	std::vector<KeyPoint> keypointsVec;
	detector->detect(img, keypointsVec);
	std::string retData;
	if (keypointsVec.size() < 1000)
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

int objMatchWithSerialData(Mat& newImg, std::string& serialData)
{
	std::vector<KeyPoint> oriKpVec, newImgKpVec;
	Mat oriDes, newImgDes;

	loadKeyPoints(serialData, oriKpVec, oriDes);

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

	if (matcheVec.size() >= 1000)
	{
		return 1;
	}

	return 0;
}

const char* saveKeyPoints(std::vector<KeyPoint>& kps, Mat& descriptors)
{
	FileStorage fs("kps.xml", FileStorage::WRITE_BASE64 + FileStorage::MEMORY);
	fs << "kps" << kps << "des" << descriptors;
	return fs.releaseAndGetString().c_str();
}

void loadKeyPoints(std::string& buff, std::vector<KeyPoint>& kps, Mat& descriptors)
{
	cv::FileStorage fs(buff, FileStorage::READ + FileStorage::MEMORY);
	fs["kps"] >> kps;
	fs["des"] >> descriptors;
	fs.release();
}

int main(int argc, char** argv)
{
	return 0;
}