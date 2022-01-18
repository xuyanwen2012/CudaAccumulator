#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>

static void handle_error(const cudaError_t err,
                         const char* file,
                         const int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
		       file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (handle_error( err, __FILE__, __LINE__ ))

struct universe
{
	double3* pos;
	double2* u;
};

void compute_with_cuda(int num_bodies);
