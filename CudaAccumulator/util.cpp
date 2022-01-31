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
