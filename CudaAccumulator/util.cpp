#include "util.h"

#include <iostream>
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
	const float with_mass = inv_dist3 * mass; // z is the mass in this case

	return {dx * with_mass, dy * with_mass};
}

float compute_rmse(const std::vector<std::shared_ptr<body<float>>>& bodies,
                   const std::pair<float, float>* us,
                   const size_t samples = 100)
{
	float rmse{};

	const auto ground_truth = static_cast<std::pair<float, float>*>(calloc(samples, sizeof(std::pair<float, float>)));

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

	constexpr unsigned n_to_print = 10;
	std::cout << "==================" << std::endl;
	for (unsigned i = 0; i < n_to_print; ++i)
	{
		std::cout << '('
			<< ground_truth[i].first << ", "
			<< ground_truth[i].second << ')' << std::endl;
	}

#pragma omp parallel for reduction(+:sum)
	for (size_t i = 0; i < samples; ++i)
	{
		rmse += powf(ground_truth[i].first - us[i].first, 2);
		rmse += powf(ground_truth[i].second - us[i].second, 2);
	}

	return sqrtf(rmse / static_cast<float>(samples));
}
