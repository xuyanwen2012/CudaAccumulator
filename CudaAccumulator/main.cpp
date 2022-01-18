#include <array>
#include <iostream>
#include <random>

#include "cu_nbody.cuh"

double my_rand()
{
	static thread_local std::mt19937 generator; // NOLINT(cert-msc51-cpp)
	const std::uniform_real_distribution<double> distribution(0.0, 1.0);
	return distribution(generator);
}

int main(int argc, char* argv[])
{
	constexpr int num_bodies = 1024;

	//// Inputs
	//std::array<double, num_bodies> xs{};
	//std::array<double, num_bodies> ys{};
	//std::array<double, num_bodies> masses{};

	//for (int i = 0; i < num_bodies; ++i)
	//{
	//	xs[i] = my_rand();
	//	ys[i] = my_rand();
	//	masses[i] = my_rand() * 1.5;
	//}

	// Outputs
	//std::array<double2, num_bodies> us{};

	// Compute
	compute_with_cuda(num_bodies);

	//// Print result
	//for (int i = 0; i < 10; i+=2)
	//{
	//	std::cout << '(' << us[i] << ", " << us[i + 1]<< ')' << std::endl;
	//}

	return EXIT_SUCCESS;
}
