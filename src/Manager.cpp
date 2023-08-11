#include "Manager.h"
#include <assert.h>
#include "H5Cpp.h"
#include "NeuralNetworkManager.h"
#include "NeuralNetworkData.h"

using namespace arma;
using namespace H5;

//===========================================================
Manager* Manager::TheManager = 0;

//===========================================================
Manager* Manager::GetInstance()
{
	if (!TheManager)
	{
		TheManager = new Manager();
	}

	return TheManager;
}

//===========================================================
Manager::Manager()
{
}

//===========================================================
Manager::~Manager()
{
	delete TheManager;
	delete pNeuralNetworkManager;
}

//===========================================================
void Manager::Initiate(std::string operation)
{
	pNeuralNetworkManager = new NeuralNetworkManager(operation);
	if (operation == "-Train")
	{
		Train();
	}
	else if (operation == "-Predict")
	{
		Predict();
	}
}

//===========================================================
void Manager::Predict()
{
	// Data structure that contains all data
	NeuralNetworkData* pData = new NeuralNetworkData();

	// Initialize the data
	Initialize(pData, Operation::PredictOperation);

	// Predict!
	arma::mat predictionMat;
	Inspect(pData, predictionMat);
	std::cout << "The result is " << endl << predictionMat << endl;
}

//===========================================================
void Manager::Train()
{
	// Data structure that contains all data
	NeuralNetworkData* pData = new NeuralNetworkData();

	// Initialize the data
	Initialize(pData, Operation::TrainOperation);

	// Train the data
	std::vector<double> costVec;
	std::pair<arma::mat, double> dwb;
	Optimize(pData, costVec, dwb);

	// Validate using training data
	pData->TrainedOutputMatrix()->clear();
	arma::mat predictionMat0;
	Inspect(pData, predictionMat0);
	std::cout << "train accuracy : " << "%" << 100 - arma::mean(arma::abs(predictionMat0 - (*pData->TrueOutputMatrix())), 1) * 100 << endl;

	// Validate using testing data
	pData->TrainedOutputMatrix()->clear();
	pData->InputMatrix()->clear();
	pData->TrueOutputMatrix()->clear();
	Initialize(pData, Operation::TestOperation);
	arma::mat predictionMat;
	Inspect(pData, predictionMat);
	std::cout << "test accuracy : " << "%" << 100 - arma::mean(arma::abs(predictionMat - (*pData->TrueOutputMatrix())), 1) * 100 << endl;

	std::vector<arma::mat> matrices2Save;
	matrices2Save.push_back(*pData->WeightsMatrix());
	matrices2Save.push_back(*pData->Bias());
	SaveMatrices(Manager::GetInstance()->OUTPUT_FILE(), matrices2Save);

	std::cin.get();
}

//===========================================================
void Manager::SaveMatrices(std::string outputFile, std::vector<arma::mat> matrices2Save)
{
	for (size_t i = 0; i < matrices2Save.size(); i++)
	{
		matrices2Save[i].save(csv_name(outputFile + "\\Output" + std::to_string(i) + ".csv"));
	}
}

//===========================================================
void Manager::Inspect(NeuralNetworkData* pData, arma::mat& predictionMat)
{
	pNeuralNetworkManager->ForwardPropagation(pData);
	predictionMat.set_size(pData->TrainedOutputMatrix()->n_rows, pData->TrainedOutputMatrix()->n_cols);
	for (size_t i = 0; i < pData->TrainedOutputMatrix()->n_cols; i++)
	{
		if ((*pData->TrainedOutputMatrix())(0, i) > 0.5)
		{
			predictionMat(0, i) = 1;
		}
		else
		{
			predictionMat(0, i) = 0;
		}
		//cout << predictionMat(0, i) << "++++++" << (*pData->TrueOutputMatrix())(0, i) << endl;
	}
}

//===========================================================
void Manager::Optimize(NeuralNetworkData* pData, std::vector<double>& costVec, std::pair<arma::mat, double>& dwb)
{
	std::string LearningRate = Manager::GetInstance()->LEARNING_RATE();
	double learningRate = std::stod(LearningRate);
	std::string batchesNumS = Manager::GetInstance()->BATCHES_NUMBER();
	int batchesNum = std::stoi(batchesNumS);
	for (size_t i = 0; i < batchesNum; i++)
	{
		double outCost;
		pNeuralNetworkManager->Propagate(pData, outCost, dwb);
		*pData->WeightsMatrix() += -1 * learningRate * dwb.first;
		*pData->Bias() += -1 * learningRate * dwb.second;

		if (i % 100 == 0)
		{
			costVec.push_back(outCost);
			std::cout << "Cost after iteration " << i << ": " << outCost << endl;
		}
	}
}

//===========================================================
void Manager::Initialize(NeuralNetworkData* pData, Manager::Operation operationOnSet)
{
	if (operationOnSet == Manager::Operation::TrainOperation)
	{
		FillInputMatrix(*pData->InputMatrix(), Manager::GetInstance()->TRAINING_DATA(), "train_set_x");
		FillTrueOutputMatrix(*pData->TrueOutputMatrix(), Manager::GetInstance()->TRAINING_DATA(), "train_set_y");
		FillWeightsBiasMatrix(*pData->WeightsMatrix(), *pData->Bias(), *pData->InputMatrix());
	}
	else if (operationOnSet == Manager::Operation::TestOperation)
	{
		FillInputMatrix(*pData->InputMatrix(), Manager::GetInstance()->TESTING_DATA(), "test_set_x");
		FillTrueOutputMatrix(*pData->TrueOutputMatrix(), Manager::GetInstance()->TESTING_DATA(), "test_set_y");
	}
	else  if (operationOnSet == Manager::Operation::PredictOperation)
	{
		FillInputMatrix(*pData->InputMatrix(), Manager::GetInstance()->TESTING_DATA());
		FillWeightsBiasMatrix(*pData->WeightsMatrix(), *pData->Bias(), Manager::GetInstance()->INPUT_MODEL_WEIGHTS(), Manager::GetInstance()->INPUT_MODEL_BIASES());
	}
}

//===========================================================
void Manager::FillInputMatrix(arma::mat& X, std::string matrixPath, std::string DataSet)
{
	X = (ReadH5(matrixPath, DataSet));
	X /= 255; // Standarize
}

//===========================================================
void Manager::FillInputMatrix(arma::mat& X, std::string imagePath)
{
	//X.load(imagePath, ppm_binary);

	//cube RGB(173, 232, 3, fill::randu);
	////RGB *= 100;
	//RGB.slice(0).submat(span(20, 51), span(20, 51)) =
	//	/*250 **/ ones(32, 32); // Red box
	//RGB.slice(1).submat(span(80, 111), span(70, 101)) =
	//	/*250 **/ ones(32, 32); // Green box
	//RGB.slice(2).submat(span(20, 51), span(110, 141)) =
	//	/*250 **/ ones(32, 32); // Blue box

	//arma::mat R = vectorise(RGB.slice(0));
	//arma::mat G = vectorise(RGB.slice(1));
	//arma::mat B = vectorise(RGB.slice(2));

	//X = arma::join_cols(R, G, B);

	cube RGB;
	RGB.load(imagePath, ppm_binary);
	X = vectorise(RGB.slice(0));
}

//===========================================================
void Manager::FillTrueOutputMatrix(arma::mat& Y, std::string matrixPath, std::string DataSet)
{
	Y = (ReadH5(matrixPath, DataSet));
}

//===========================================================
void Manager::FillWeightsBiasMatrix(arma::mat& W, arma::mat& b, arma::mat X)
{
	W = arma::mat(X.n_rows, 1, arma::fill::zeros);
	b = arma::mat(1, X.n_cols, fill::zeros);
}

//===========================================================
void Manager::FillWeightsBiasMatrix(arma::mat& W, arma::mat& b, std::string wightsPath, std::string biasesPath)
{
	W = ReadCSV(wightsPath);
	b = ReadCSV(biasesPath);
}

//===========================================================
arma::mat Manager::ReadCSV(std::string fileName)
{
	mat A;
	A.load(fileName);
	return A;
}

//===========================================================
arma::mat Manager::ReadH5(std::string fileName, std::string dataSetName)
{
	const H5std_string FILE_NAME(fileName);
	const H5std_string DATASET_NAME(dataSetName);

	H5File file(FILE_NAME, H5F_ACC_RDONLY); //Open the specified file and the specified dataset in the file.
	DataSet dataset = file.openDataSet(DATASET_NAME);

	DataSpace dataspace = dataset.getSpace(); //Get dataspace of the dataset.

	int rank = dataspace.getSimpleExtentNdims(); //Get the number of dimensions in the dataspace.
	hsize_t* dims_out = new hsize_t[rank]; // Get the dimension size of each dimension in the dataspaceand display them.
	int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);

	hsize_t dataOutRawSize = 1;
	hsize_t dataOutColomSize = 1;
	std::cout << "rank " << rank << ", dimensions ";
	for (size_t i = 0; i < rank; i++)
	{
		hsize_t singleData = dims_out[i];
		if (i == 0)
		{
			dataOutColomSize *= singleData;
		}
		else
		{
			dataOutRawSize *= singleData;
		}
		std::cout << singleData;
		if (i < rank - 1.0)
		{
			std::cout << " x ";
		}
	}
	std::cout << endl;

	/*
	* Define hyperslab in the dataset; implicitly giving strike and
	* block NULL.
	*/
	//hsize_t* offset = new hsize_t[rank];   // hyperslab offset in the file
	//hsize_t* count = new hsize_t[rank];   // hyperslab offset in the file
	//for (size_t i = 0; i < rank; i++)
	//{
	//	offset[i] = 0;
	//	count[i] = dims_out[i];
	//}
	//count[0] = 1;

	hsize_t offset[4];   // hyperslab offset in the file
	hsize_t count[4];   // hyperslab offset in the file
	for (size_t i = 0; i < 4; i++)
	{
		offset[i] = 0;
		count[i] = dims_out[i];
	}
	count[0] = 1;

	/*
	* Define the memory dataspace.
	*/
	hsize_t dimsm[2];
	dimsm[0] = dataOutColomSize;
	dimsm[1] = dataOutRawSize;
	DataSpace memspace(2, dimsm);

	/*
	 * Define memory hyperslab.
	 */
	hsize_t offset_out[2];       // hyperslab offset in memory
	hsize_t count_out[2];        // size of the hyperslab in memory
	offset_out[1] = 0;
	count_out[0] = 1;
	count_out[1] = dataOutRawSize;

	//std::vector<int> data_out(dataOutColomSize * dataOutRawSize, 0);
	//dataset.read(&data_out[0], PredType::NATIVE_INT, memspace, dataspace);

	arma::mat A(dataOutRawSize, dataOutColomSize, fill::zeros); // Output buffer initialization
	for (size_t i = 0; i < dataOutColomSize; i++)
	{
		offset_out[0] = i;
		offset[0] = i;
		memspace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);
		dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
		dataset.read(A.memptr(), PredType::NATIVE_DOUBLE, memspace, dataspace);
	}

	return A;
}

//===========================================================
void Manager::PrintMatrix(std::string name, arma::mat matrix)
{
	std::cout << name << " : " << endl;
	std::cout << matrix << endl;
}

//===========================================================
bool Manager::FillUserInputData(std::string operation, std::vector<std::string> arguments)
{
	if (operation == "-Train")
	{
		short i = 0;
		mNETWORK_TYPE = (arguments[i++]);
		mTRAINING_DATA = (arguments[i++]);
		mTESTING_DATA = (arguments[i++]);
		mACTIVATION_FUNCTION = (arguments[i++]);
		mCOST_FUNCTION = (arguments[i++]);
		mLEARNING_RATE = (arguments[i++]);
		mBATCHES_NUMBER = (arguments[i++]);
		mOUTPUT_FILE = (arguments[i++]);
		return true;
	}
	else if (operation == "-Predict")
	{
		short i = 0;
		mINPUT_MODEL_WEIGHTS = (arguments[i++]);
		mINPUT_MODEL_BIASES = (arguments[i++]);
		mTESTING_DATA = (arguments[i++]);
		mACTIVATION_FUNCTION = (arguments[i++]);
		return true;
	}
	return false;
}