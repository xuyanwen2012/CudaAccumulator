#include "cu_nbody.cuh"

#include <array>



void randomize_bodies(double* data, int n)
{
	for (int i = 0; i < n; i++)
	{
		data[i] = rand() / static_cast<double>(RAND_MAX);
	}
}


void compute_with_cuda(const int num_bodies)
{





}

