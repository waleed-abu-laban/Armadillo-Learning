#pragma once

#include <armadillo>
#include <vector>
#include <string>

class NeuralNetworkData;
class ActivationFunction;
class CostFunction;
class NeuralNetworkManager
{
public:
	NeuralNetworkManager(std::string operation);
	~NeuralNetworkManager();

	virtual void Propagate(NeuralNetworkData* pData, double& cost, std::pair<arma::mat, double>& dwb);
	
	virtual void ForwardPropagation(NeuralNetworkData* pData);
	virtual void BackwardPropagation(NeuralNetworkData* pData, double& cost, std::pair<arma::mat, double>& dwb);
	
private:
	ActivationFunction* m_activationFunction;
	CostFunction* m_costFunction;
};