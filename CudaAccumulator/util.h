#pragma once

#include <memory>
#include <vector>

#include "body.h"

#define TIME_THIS_SEGMENT(lines)  \
const auto start = std::chrono::steady_clock::now();\
lines;\
const auto end = std::chrono::steady_clock::now();\
const std::chrono::duration<double> elapsed_seconds = end - start;\
std::cout << "- elapsed time: " << elapsed_seconds.count() << "s\n";

float my_rand();

void print_ground_truth(const std::vector<std::shared_ptr<body<float>>>& bodies);
