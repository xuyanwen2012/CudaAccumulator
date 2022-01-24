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


void print_ground_truth(const float* xs, const float* ys, const float* masses, const unsigned n)
{
	constexpr unsigned n_to_print = 10;
	std::array<std::pair<float, float>, n_to_print> us{};

	for (unsigned i = 0; i < n_to_print; ++i)
	{
		us[i] = {0.0f, 0.0f};
		for (unsigned j = 0; j < n; ++j)
		{
			const float dx = xs[i] - xs[j];
			const float dy = ys[i] - ys[j];

			const float dist_sqr = dx * dx + dy * dy + 1e-9f;
			const float inv_dist = 1.0f / sqrtf(dist_sqr);
			const float inv_dist3 = inv_dist * inv_dist * inv_dist;
			const float with_mass = inv_dist3 * masses[j]; // z is the mass in this case

			us[i].first += dx * with_mass;
			us[i].second += dy * with_mass;
		}
	}

	std::cout << "==================" << std::endl;
	for (unsigned i = 0; i < n_to_print; ++i)
	{
		std::cout << '('
			<< us[i].first << ", "
			<< us[i].second << ')' << std::endl;
	}
}

int main(int argc, char* argv[])
{
	constexpr int num_bodies = 1024 * 2;

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

	print_ground_truth(xs.data(), ys.data(), masses.data(), num_bodies);

	return EXIT_SUCCESS;
}
