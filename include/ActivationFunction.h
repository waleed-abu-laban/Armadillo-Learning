#pragma once

#include <string>
#include <armadillo>

class NeuralNetworkData;
class ActivationFunction
{
public:
	enum class Type { ReLU, Sigmoid };

	ActivationFunction();
	~ActivationFunction();

	void SetType(ActivationFunction::Type);
	ActivationFunction::Type GetType();

	void Calculate(NeuralNetworkData* pData);

private:
	ActivationFunction::Type m_type;
};