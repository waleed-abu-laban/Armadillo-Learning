#include "ConvolutionalNetworkManager.h"
#include "Manager.h"
#include <assert.h>
#include "ActivationFunction.h"
#include "NeuralNetworkData.h"
#include "opencv2/opencv.hpp"

using namespace arma;

//===========================================================
ConvolutionalNetworkManager::ConvolutionalNetworkManager(std::string operation)
	:NeuralNetworkManager(operation)
{
}

//===========================================================
ConvolutionalNetworkManager::~ConvolutionalNetworkManager()
{
}

//===========================================================
//void ConvolutionalNetworkManager::FillInputMatrix(arma::mat& X, std::string matrixPath, std::string dataSet)
//{
//	//std::vector<cv::Mat> images;
//	//std::vector<std::string> subPaths;
//	//subPaths.push_back("/cats");
//	//subPaths.push_back("/dogs");
//	//double isCat;
//	//std::vector< double > vgrady;
//	//for (size_t j = 0; j < subPaths.size(); j++)
//	//{
//	//	if (j == 0)
//	//	{
//	//		isCat = 1.0;
//	//	}
//	//	else
//	//	{
//	//		isCat = 0.0;
//	//	}
//	//	std::vector<cv::String> png;
//	//	cv::glob(imagesPath + subPaths[j] + "/*.png", png, false);
//	//	for (size_t i = 0; i < png.size(); i++)
//	//	{
//	//		images.push_back(cv::imread(png[i]));
//	//		vgrady.push_back(isCat);
//	//		//imshow("Display window", images[i]);
//	//		//int k = cv::waitKey(0); // Wait for a keystroke in the window
//	//	}
//
//	//	std::vector<cv::String> jpg;
//	//	cv::glob(imagesPath + subPaths[j] + "/*.jpg", jpg, false);
//	//	//for (size_t i = 0; i < jpg.size(); i++)
//	//	for (size_t i = 0; i < jpg.size(); i++)
//	//	{
//	//		images.push_back(cv::imread(jpg[i]));
//	//		vgrady.push_back(isCat);
//	//		//imshow("Display window", images[images.size() - 1 + i]);
//	//		//int k = cv::waitKey(0); // Wait for a keystroke in the window
//	//	}
//	//}
//	//m_neuralNetworkData->Y = mat(vgrady).t();
//
//	//if (calcSize)
//	//{
//	//	int minRowCount = INT_MAX;
//	//	int minColCount = INT_MAX;
//
//	//	for (size_t i = 0; i < images.size(); i++)
//	//	{
//	//		minRowCount = std::min(minRowCount, images[i].rows);
//	//		minColCount = std::min(minColCount, images[i].cols);
//	//	}
//	//	m_rowsNum = minRowCount;
//	//	m_colsNum = minColCount;
//	//}
//
//	//for (size_t i = 0; i < images.size(); i++)
//	//{
//	//	cv::resize(images[i], images[i], cv::Size(m_colsNum, m_rowsNum));//resize image
//	//}
//
//	//std::vector<mat>tempMatVec;
//	//for (size_t i = 0; i < images.size(); i++)
//	//{
//	//	std::vector< double > vgradx;
//	//	cv::Mat vec = images[i].reshape(1, 1);
//	//	vgradx.insert(vgradx.end(), vec.data, vec.data + vec.cols);
//	//	mat tempMat(vgradx);
//	//	tempMatVec.push_back(tempMat);
//	//}
//
//	//for (size_t i = 0; i < tempMatVec.size(); i++)
//	//{
//	//	m_neuralNetworkData->X.insert_cols(i, tempMatVec[i]);
//	//}
//}