#pragma once

#include <vector>
#include <memory>
#include "body.h"

using pair_f = std::pair<float, float>;
using body_container = std::vector<std::shared_ptr<body<float>>>;

template <typename T>
T* make_output_array(const size_t n)
{
	constexpr auto bytes_2_f = sizeof(T);
	return static_cast<T*>(calloc(n, bytes_2_f));
};

float my_rand();

float compute_rmse(const body_container& bodies,
                   const pair_f* us,
                   size_t samples);
