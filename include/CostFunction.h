#pragma once

#include <string>
#include <armadillo>

class NeuralNetworkData;
class CostFunction
{
public:
	enum class Type { Log };

	CostFunction();
	~CostFunction();

	void SetType(CostFunction::Type);
	CostFunction::Type GetType();

	void Calculate(NeuralNetworkData* pData, double &cost, std::pair<arma::mat, double>& dwb);

private:
	CostFunction::Type m_type;
};