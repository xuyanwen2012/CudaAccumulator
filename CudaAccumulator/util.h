#pragma once

#include <memory>
#include <vector>

#include "body.h"

#include <chrono>

//static void debug_print()
//{
//	constexpr bool show_log = true;
//
//	if (show_log)
//	{
//		printf("%s in %s at line %d\n");
//	}
//}
//
//#define HANDLE_ERROR(  ) (debug_print(  ))


float my_rand(float f_min = 0.0, float f_max = 1.0);

void print_ground_truth(const std::vector<std::shared_ptr<body<float>>>& bodies);
