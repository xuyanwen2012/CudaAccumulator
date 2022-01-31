#include "util.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <omp.h>

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
#include "cstdlib"
float my_rand()
{
	constexpr float f_min = 0.0f;
	constexpr float f_max = 1.0f;
	const float f = static_cast<float>(rand()) / RAND_MAX;
	return f_min + f * (f_max - f_min);
}
#endif

template <typename T>
std::pair<T, T> cpu_kernel_func_debug(T x0, T y0, T mass, T x1, T y1)
{
	const float dx = x0 - x1;
	const float dy = y0 - y1;

	const float dist_sqr = dx * dx + dy * dy + 1e-9f;
	const float inv_dist = 1.0f / sqrtf(dist_sqr);
	const float inv_dist3 = inv_dist * inv_dist * inv_dist;
	const float with_mass = inv_dist3 * mass;

	return {dx * with_mass, dy * with_mass};
}

float compute_rmse(const body_container& bodies,
                   const pair_f* us,
                   const size_t samples,
                   const bool show_results)
{
	assert(samples >= 10);

	const auto ground_truth = make_output_array<pair_f>(samples);

	// Compute ground truth
#pragma omp parallel for schedule(dynamic)
	for (size_t i = 0; i < samples; ++i)
	{
		for (size_t j = 0; j < bodies.size(); ++j)
		{
			const auto u = cpu_kernel_func_debug(
				bodies[i]->x, bodies[i]->y,
				bodies[j]->mass,
				bodies[j]->x, bodies[j]->y
			);

			ground_truth[i].first += u.first;
			ground_truth[i].second += u.second;
		}
	}

	// Print (part of) Result vs. Ground truth
	if (show_results)
	{
		constexpr unsigned n_to_print = 10;

		for (unsigned i = 0; i < n_to_print; ++i)
		{
			std::cout << '(' << us[i].first << ", " << us[i].second << ')' << std::endl;
		}

		std::cout << "==================" << std::endl;

		for (unsigned i = 0; i < n_to_print; ++i)
		{
			std::cout << '('
				<< ground_truth[i].first << ", "
				<< ground_truth[i].second << ')'
				<< std::endl;
		}
	}

	// Compute the RMSE
	float rmse{};
#pragma omp parallel for reduction(+:sum)
	for (size_t i = 0; i < samples; ++i)
	{
		const auto dx = ground_truth[i].first - us[i].first;
		const auto dy = ground_truth[i].second - us[i].second;

		rmse += powf(dx, 2);
		rmse += powf(dy, 2);
	}

	return sqrtf(rmse / static_cast<float>(samples * 2));
}
