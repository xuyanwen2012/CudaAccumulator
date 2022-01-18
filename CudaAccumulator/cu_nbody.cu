#include "cu_nbody.cuh"

#include <array>

#include "device_launch_parameters.h"

__global__ void body_force(double3* pos, double2* us, const int n)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n)
	{
		us[tid].x = pos[tid].x + pos[tid].y;
		printf("%f\n", us[tid].x);
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

	HANDLE_ERROR(cudaDeviceReset());
}
