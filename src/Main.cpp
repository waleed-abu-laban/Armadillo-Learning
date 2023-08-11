#include <iostream>
#include <vector>
#include <string>
#include "Manager.h"

using namespace std;

int main(int argc, char** argv)
{
	if (argc > 2)
	{
		std::vector<std::string> Arguments;
		for (size_t i = 2; i < argc; i++)
		{
			Arguments.push_back(argv[i]);
		}
		Manager* manager = Manager::GetInstance();
		manager->FillUserInputData(argv[1], Arguments);
		manager->Initiate(argv[1]); // Start the process;
	}
	else
	{
		cout << "No Arguments are passed";
	}

	return 0;
}