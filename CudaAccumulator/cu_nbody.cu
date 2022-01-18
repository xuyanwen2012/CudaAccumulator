#include "cu_nbody.cuh"

#include <array>

#include "device_launch_parameters.h"

__global__ void body_force(const double3* pos, double2* us, const int n)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n)
	{
		for (int j = 0; j < n; j++)
		{
			const double dx = pos[j].x - pos[i].x;
			const double dy = pos[j].y - pos[i].y;

			const double dist_sqr = dx * dx + dy * dy + 1e-9;
			const double inv_dist = rsqrt(dist_sqr);
			const double inv_dist3 = inv_dist * inv_dist * inv_dist;
			const double with_mass = inv_dist3 * pos[j].z; // z is the mass in this

			us[i].x += dx * with_mass;
			us[i].y += dy * with_mass;
		}
	}
}

void randomize_bodies(double* data, int n)
{
	for (int i = 0; i < n; i++)
	{
		data[i] = rand() / static_cast<double>(RAND_MAX);
	}
}

void compute_with_cuda(const int num_bodies)
{
	HANDLE_ERROR(cudaSetDevice(0));

	double3* dev_pos; // x, y, and mass
	double2* dev_us; // ux, uy

	const unsigned in_bytes = num_bodies * sizeof(double3);
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_pos), in_bytes));

	const unsigned out_bytes = num_bodies * sizeof(double2);
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_us), out_bytes));

	auto pos = static_cast<double3*>(malloc(in_bytes));
	randomize_bodies(reinterpret_cast<double*>(pos), 3 * num_bodies);

	auto us = static_cast<double2*>(malloc(in_bytes));

	HANDLE_ERROR(cudaMemcpy(dev_pos, pos, in_bytes, cudaMemcpyHostToDevice));

	constexpr int block_size = 256;
	const int n_blocks = (num_bodies + block_size - 1) / block_size;

	body_force <<<n_blocks, block_size>>>(dev_pos, dev_us, num_bodies);

	HANDLE_ERROR(cudaMemcpy(us, dev_us, out_bytes, cudaMemcpyDeviceToHost));

	cudaFree(dev_pos);
	cudaFree(dev_us);

	for (int i = 0; i < 10; ++i)
	{
		printf("(%f, %f)\n", us[i].x, us[i].y);
	}

	HANDLE_ERROR(cudaDeviceReset());
}
