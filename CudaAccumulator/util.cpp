#include "util.h"

#include <array>
#include <cstdlib>
#include <iostream>

float my_rand(const float f_min, const float f_max)
{
	const float f = static_cast<float>(rand()) / RAND_MAX;
	return f_min + f * (f_max - f_min);
}

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
