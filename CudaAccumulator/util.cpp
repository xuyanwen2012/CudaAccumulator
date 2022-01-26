#include "util.h"

#include <array>
#include <cstdlib>
#include <iostream>

#ifdef _WIN32

#include <random>

float my_rand()
{
	static thread_local std::mt19937 generator; // NOLINT(cert-msc51-cpp)
	const std::uniform_real_distribution<float> distribution(0.0, 1.0);
	return distribution(generator);
}

#endif

#ifdef __linux__

float my_rand()
{
	constexpr float f_min = 0.0f;
	constexpr float f_max = 1.0f;
	const float f = static_cast<float>(rand()) / RAND_MAX;
	return f_min + f * (f_max - f_min);
}
#endif

void print_ground_truth(const std::vector<std::shared_ptr<body<float>>>& bodies)
{
	constexpr unsigned n_to_print = 10;
	std::array<std::pair<float, float>, n_to_print> us{};

	for (unsigned i = 0; i < n_to_print; ++i)
	{
		us[i] = {0.0f, 0.0f};
		for (unsigned j = 0; j < bodies.size(); ++j)
		{
			const float dx = bodies[i]->x - bodies[j]->x;
			const float dy = bodies[i]->y - bodies[j]->y;

			const float dist_sqr = dx * dx + dy * dy + 1e-9f;
			const float inv_dist = 1.0f / sqrtf(dist_sqr);
			const float inv_dist3 = inv_dist * inv_dist * inv_dist;
			const float with_mass = inv_dist3 * bodies[j]->mass; // z is the mass in this case

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
