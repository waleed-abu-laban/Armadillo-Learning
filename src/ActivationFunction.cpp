#include "ActivationFunction.h"
#include "Manager.h"
#include "NeuralNetworkData.h"

using namespace arma;

//===========================================================
ActivationFunction::ActivationFunction()
{
	std::string activationFunctionType = Manager::GetInstance()->ACTIVATION_FUNCTION();
	for (size_t i = 0; i < activationFunctionType.size(); i++)
	{
		activationFunctionType[i] = std::tolower(activationFunctionType[i]);
	}

	if (activationFunctionType == "sigmoid")
	{
		SetType(ActivationFunction::Type::Sigmoid);
	}
	else
	{
		SetType(ActivationFunction::Type::ReLU);
		if (activationFunctionType != "relu")
		{
			cerr << "The passed activation function is not defined so ReLU is used instead" << endl;
		}
	}
}

//===========================================================
ActivationFunction::~ActivationFunction()
{
}

//===========================================================
void ActivationFunction::SetType(ActivationFunction::Type type)
{
	m_type = type;
}

//===========================================================
ActivationFunction::Type ActivationFunction::GetType()
{
	return m_type;
}

//===========================================================
void ActivationFunction::Calculate(NeuralNetworkData* pData)
{
	pData->InputMatrix()->reshape(pData->WeightsMatrix()->n_rows, pData->InputMatrix()->n_cols);
	mat Z = pData->WeightsMatrix()->t() * *(pData->InputMatrix()) + (*pData->Bias())(0, 0);
	if (GetType() == ActivationFunction::Type::Sigmoid)
	{
		*pData->TrainedOutputMatrix() = 1.0 / (1.0 + arma::exp(-1.0 * Z));
	}
	else
	{
		*pData->TrainedOutputMatrix() = arma::max(Z, 0);
		if (GetType() != ActivationFunction::Type::ReLU)
		{
			cerr << "The passed activation function is not defined so ReLU is used instead" << endl;
		}
	}
}