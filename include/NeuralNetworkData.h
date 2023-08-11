#pragma once

#include <armadillo>

class NeuralNetworkData
{
public:
	NeuralNetworkData()
	{ 
		X = arma::mat();
		W = arma::mat();
		b = arma::mat();
		Y = arma::mat();
		A = arma::mat();
	}

	inline arma::mat* InputMatrix() { return &X; }
	inline arma::mat* WeightsMatrix() { return &W; }
	inline arma::mat* Bias() { return &b; }
	inline arma::mat* TrueOutputMatrix() { return &Y; }
	inline arma::mat* TrainedOutputMatrix() { return &A; }

private:
	arma::mat X; //Inputs Matrix
	arma::mat W; //Weights Matrix
	arma::mat b;		 //Bias
	arma::mat Y; //Output Matrix (True values)
	arma::mat A; //Output Matrix (Trained values)
};