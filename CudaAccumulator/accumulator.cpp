#include "accumulator.h"
#include "cu_nbody.cuh"

using accumulator_handle = struct accumulator_handle
{
	double* result_addr;
	double x;
	double y;
};

accumulator_handle* get_accumulator()
{
	return nullptr;
}

int accumulator_set_constants_and_result_address(double x, double y, double* addr, accumulator_handle* acc)
{
	return 0;
}

int accumulator_accumulate(double x, double y, double mass, accumulator_handle* acc)
{
	return 0;
}

int release_accumulator(accumulator_handle* ret)
{
	return 0;
}
