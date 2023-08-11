#include "NeuralNetworkManager.h"
#include "ActivationFunction.h"
#include "CostFunction.h"
#include <assert.h>

using namespace arma;

//===========================================================
NeuralNetworkManager::NeuralNetworkManager(std::string operation)
{
	m_activationFunction = new ActivationFunction();
	if (operation == "-Train")
	{
		m_costFunction = new CostFunction();
	}
	else if (operation == "-Predict")
	{
	}
}

//===========================================================
NeuralNetworkManager::~NeuralNetworkManager()
{
	delete m_activationFunction;
	delete m_costFunction;
}

//===========================================================
void NeuralNetworkManager::Propagate(NeuralNetworkData* pData, double& cost, std::pair<arma::mat, double>& dwb)
{
	ForwardPropagation(pData);
	BackwardPropagation(pData, cost, dwb);
}

//===========================================================
void NeuralNetworkManager::ForwardPropagation(NeuralNetworkData* pData)
{
	m_activationFunction->Calculate(pData);
}

//===========================================================
void NeuralNetworkManager::BackwardPropagation(NeuralNetworkData* pData, double& cost, std::pair<arma::mat, double>& dwb)
{
	m_costFunction->Calculate(pData, cost, dwb);
}