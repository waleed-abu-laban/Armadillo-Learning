#pragma once

#include "NeuralNetworkManager.h"

class ConvolutionalNetworkManager : public NeuralNetworkManager
{
public:
	ConvolutionalNetworkManager(std::string operation);
	~ConvolutionalNetworkManager();
};