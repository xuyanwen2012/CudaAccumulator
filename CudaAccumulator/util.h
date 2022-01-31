#pragma once

#include <vector>

#include "body.h"

float my_rand();

float compute_rmse(const std::vector<std::shared_ptr<body<float>>>& bodies,
                   const std::pair<float, float>* us,
                   size_t samples);
