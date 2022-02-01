//ReSharper disable CppCStyleCast, CppParameterMayBeConst, CppClangTidyModernizeDeprecatedHeaders

// ReSharper disable CppLocalVariableMayBeConst
#include <stdlib.h>
#include <stdio.h>

#include "accumulator.h"
#include "math.h"

int running_count;
long long avg_reduction_size;
int num_reductions;


typedef struct accumulator_handle
{
	float* result_addr;
	float x;
	float y;
} accumulator_handle;


accumulator_handle* get_accumulator()
{
	num_reductions = 0;
	avg_reduction_size = 0;
	return malloc(sizeof(accumulator_handle));
}

int accumulator_set_constants_and_result_address(float x, float y, float* addr, accumulator_handle* acc)
{
	avg_reduction_size += running_count;
	running_count = 0;
	++num_reductions;

	acc->result_addr = addr; //ux
	acc->x = x;
	acc->y = y;

	return 1;
}

int accumulator_accumulate(float x, float y, float mass, accumulator_handle* acc)
{
#if defined(SIMULATOR)

	float* tmp = acc->result_addr;
	tmp[0] += 1.0;
	tmp[1] += 1.0;

#else

	++running_count;

	// kernel function computation
	float dx = acc->x - x;
	float dy = acc->y - y;
	float dist_sqr = dx * dx + dy * dy + 1e-9f; // 1e-9 is softening
	float inv_dist = 1.0f / sqrtf(dist_sqr);
	float inv_dist3 = inv_dist * inv_dist * inv_dist;
	float with_mass = inv_dist3 * mass;

	// return results
	float* tmp = acc->result_addr;
	tmp[0] += dx * with_mass;
	tmp[1] += dy * with_mass;

#endif

	return 1;
}

int accumulator_finish(accumulator_handle* acc)
{
	return 1;
}

int release_accumulator(accumulator_handle* acc)
{
	avg_reduction_size += running_count;
	avg_reduction_size /= num_reductions;

	long long tmp = avg_reduction_size;

	// printf("Stats:\n");
	printf("avg_reduction_elements: %lld\n", avg_reduction_size);
	// printf("avg_reduction_size (elements * size of float * 3): %lld\n", avg_reduction_size * sizeof(float) * 3);

	free(acc);
	return 1;
}
