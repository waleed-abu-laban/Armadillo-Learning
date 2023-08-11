#include "CostFunction.h"
#include "Manager.h"
#include "NeuralNetworkData.h"

using namespace arma;

//===========================================================
CostFunction::CostFunction()
{
	std::string CostFunctionType = Manager::GetInstance()->COST_FUNCTION();
	for (size_t i = 0; i < CostFunctionType.size(); i++)
	{
		CostFunctionType[i] = std::tolower(CostFunctionType[i]);
	}

	SetType(CostFunction::Type::Log);
	if (CostFunctionType != "log")
	{
		cerr << "The passed cost function is not defined so log is used instead" << endl;
	}
}

//===========================================================
CostFunction::~CostFunction()
{
}

//===========================================================
void CostFunction::SetType(CostFunction::Type type)
{
	m_type = type;
}

//===========================================================
CostFunction::Type CostFunction::GetType()
{
	return m_type;
}

//===========================================================
void CostFunction::Calculate(NeuralNetworkData* pData, double &cost, std::pair<arma::mat, double>& dwb)
{
	//if (GetType() == CostFunction::Type::Log)
	{
		arma::mat Y = *pData->TrueOutputMatrix();
		arma::mat A = *pData->TrainedOutputMatrix();
		mat dZ = A - Y;
		double oneOverM = (1.0 / A.n_cols);
		mat lossFunction = -1 * (Y % arma::log(A) + (1 - Y) % arma::log(1 - A));
		cost = arma::accu(lossFunction) * oneOverM;
		dwb.first = *pData->InputMatrix() * dZ.t() * oneOverM;
		dwb.second = arma::accu(dZ) * oneOverM;
	}
	if (GetType() != CostFunction::Type::Log)
	{
		cerr << "The passed cost function is not defined so log is used instead" << endl;
	}
}