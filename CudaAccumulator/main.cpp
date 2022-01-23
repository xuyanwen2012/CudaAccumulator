#include <array>
#include <iostream>
#include <random>

#include <vector_types.h>
#include "accumulator.h"

#ifdef _WIN32
float my_rand()
{
	static thread_local std::mt19937 generator; // NOLINT(cert-msc51-cpp)
	const std::uniform_real_distribution<float> distribution(0.0, 1.0);
	return distribution(generator);
}
#endif

#ifdef __linux__
float my_rand(const float f_min = 0.0, const float f_max = 1.0)
{
	const float f = static_cast<float>(rand()) / RAND_MAX;
	return f_min + f * (f_max - f_min);
}
#endif


int main(int argc, char* argv[])
{
	constexpr int num_bodies = 1024;

	// Inputs
	std::array<float, num_bodies> xs{};
	std::array<float, num_bodies> ys{};
	std::array<float, num_bodies> masses{};

	for (int i = 0; i < num_bodies; ++i)
	{
		xs[i] = my_rand();
		ys[i] = my_rand();
		masses[i] = my_rand() * 1.5;
	}

	// Outputs
	std::array<float2, num_bodies> us{};

	// Compute
	accumulator_handle* acc = get_accumulator();

	for (int i = 0; i < num_bodies; ++i)
	{
		accumulator_set_constants_and_result_address(xs[i], ys[i], &us[i].x, acc);

		for (int j = 0; j < num_bodies; ++j)
		{
			accumulator_accumulate(xs[j], ys[j], masses[j], acc);
		}
	}

	release_accumulator(acc);


	// Print result
	for (int i = 0; i < 10; ++i)
	{
		std::cout << '(' << us[i].x << ", " << us[i].y << ')' << std::endl;
	}

	return EXIT_SUCCESS;
}
