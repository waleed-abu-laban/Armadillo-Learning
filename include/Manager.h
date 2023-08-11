#pragma once

#include <armadillo>
#include <vector>
#include <string>

class NeuralNetworkData;
class NeuralNetworkManager;
class Manager
{
public:
	enum class Operation { TrainOperation, TestOperation, PredictOperation };

	static Manager* GetInstance();
	~Manager();

	bool FillUserInputData(std::string operation, std::vector<std::string>);

	virtual void Train();
	virtual void Predict();

	virtual void Initiate(std::string operation);
	virtual void Initialize(NeuralNetworkData* pData, Manager::Operation operationOnSet);
	virtual void Optimize(NeuralNetworkData* pData, std::vector<double>& costVec, std::pair<arma::mat, double>& dwbVec);
	virtual void Inspect(NeuralNetworkData* pData, arma::mat& predictionMat);

	virtual void FillInputMatrix(arma::mat& X, std::string matrixPath, std::string DataSet);
	virtual void FillInputMatrix(arma::mat& X, std::string matrixPath);
	virtual void FillTrueOutputMatrix(arma::mat& Y, std::string matrixPath, std::string DataSet);
	virtual void FillWeightsBiasMatrix(arma::mat& W, arma::mat& b, arma::mat X);
	virtual void FillWeightsBiasMatrix(arma::mat& W, arma::mat& b, std::string wightsPath, std::string biasesPath);

	arma::mat ReadCSV(std::string fileName);
	arma::mat ReadH5(std::string fileName, std::string dataSetName);
	void SaveMatrices(std::string outputFile, std::vector<arma::mat> matrices2Save);
	void PrintMatrix(std::string name, arma::mat matrix);

	inline const std::string NETWORK_TYPE() const { return mNETWORK_TYPE; }
	inline const std::string TRAINING_DATA() const { return mTRAINING_DATA; }
	inline const std::string TESTING_DATA() const { return mTESTING_DATA; }
	inline const std::string ACTIVATION_FUNCTION() const { return mACTIVATION_FUNCTION; }
	inline const std::string COST_FUNCTION() const { return mCOST_FUNCTION; }
	inline const std::string LEARNING_RATE() const { return mLEARNING_RATE; }
	inline const std::string BATCHES_NUMBER() const { return mBATCHES_NUMBER; }
	inline const std::string OUTPUT_FILE() const { return mOUTPUT_FILE; }
	inline const std::string INPUT_MODEL_WEIGHTS() const { return mINPUT_MODEL_WEIGHTS; }
	inline const std::string INPUT_MODEL_BIASES() const { return mINPUT_MODEL_BIASES; }

private:
	Manager();
	static Manager* TheManager;

	NeuralNetworkManager* pNeuralNetworkManager;

	std::string mNETWORK_TYPE;
	std::string mTRAINING_DATA;
	std::string mTESTING_DATA;
	std::string mACTIVATION_FUNCTION;
	std::string mCOST_FUNCTION;
	std::string mLEARNING_RATE;
	std::string mBATCHES_NUMBER;
	std::string mOUTPUT_FILE;
	std::string mINPUT_MODEL_WEIGHTS;
	std::string mINPUT_MODEL_BIASES;
};