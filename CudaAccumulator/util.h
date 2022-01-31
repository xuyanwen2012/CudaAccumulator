#pragma once

#include <vector>

#include "body.h"

using pair_f = std::pair<float, float>;
using body_container = std::vector<std::shared_ptr<body<float>>>;

float my_rand();

float compute_rmse(const body_container& bodies,
                   const pair_f* us,
                   size_t samples);
