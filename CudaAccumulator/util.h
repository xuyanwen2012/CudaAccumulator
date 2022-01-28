#pragma once

#include <memory>
#include <vector>

#include "body.h"

float my_rand();

void print_ground_truth(const std::vector<std::shared_ptr<body<float>>>& bodies);
