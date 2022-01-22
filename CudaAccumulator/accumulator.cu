#include "accumulator.h"

#include <cuda_runtime_api.h>
#include <vector_functions.h>

#include "cu_nbody.cuh"

using accumulator_handle = struct accumulator_handle
{
	double* result_addr;
	double x;
	double y;

	// CUDA Related
	double3* pos_buf;
	double2* us_buf;
	int buf_count;

	double3* dev_pos; // x, y, and mass
	double2* dev_us; // ux, uy
};

#include "device_launch_parameters.h"

__device__ double2 kernel_func(const double3 p, const double3 q)
{
	const double dx = p.x - q.x;
	const double dy = p.y - q.y;

	const double dist_sqr = dx * dx + dy * dy + 1e-9;
	const double inv_dist = rsqrt(dist_sqr);
	const double inv_dist3 = inv_dist * inv_dist * inv_dist;
	const double with_mass = inv_dist3 * q.z; // z is the mass in this

	return make_double2(dx * with_mass, dy * with_mass);
}


__global__ void body_force(const double3* pos, double2* us, const int n)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n)
	{
		for (int j = 0; j < n; j++)
		{
			const double dx = pos[j].x - pos[i].x;
			const double dy = pos[j].y - pos[i].y;

			const auto force = kernel_func(pos[j], pos[i]);

			us[i].x += force.x;
			us[i].y += force.y;
		}
	}
}

accumulator_handle* get_accumulator()
{
	// TODO: Parametrize this
	constexpr int max_num_bodies_per_compute = 1024;

	const auto acc = new accumulator_handle{};

	HANDLE_ERROR(cudaSetDevice(0));

	constexpr unsigned in_bytes = max_num_bodies_per_compute * sizeof(double3);
	constexpr unsigned out_bytes = max_num_bodies_per_compute * sizeof(double2);

	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_pos), in_bytes));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_us), out_bytes));

	acc->pos_buf = static_cast<double3*>(malloc(in_bytes));
	acc->us_buf = static_cast<double2*>(malloc(out_bytes));

	return acc;
}


int accumulator_set_constants_and_result_address(const double x, const double y, double* addr, accumulator_handle* acc)
{
	acc->x = x;
	acc->y = y;
	acc->result_addr = addr;
	acc->buf_count = 0;

	return 0;
}


int release_accumulator(const accumulator_handle* ret)
{
	cudaFree(ret->dev_pos);
	cudaFree(ret->dev_us);

	for (int i = 0; i < 10; ++i)
	{
		printf("(%f, %f)\n", ret->us_buf[i].x, ret->us_buf[i].y);
	}

	HANDLE_ERROR(cudaDeviceReset());
	return 0;
}

int accumulator_accumulate(double x, double y, double mass, accumulator_handle* acc)
{
	constexpr int max_num_bodies_per_compute = 1024;

	// Push this to the buffer 
	acc->pos_buf[acc->buf_count] = make_double3(x, y, mass);
	++acc->buf_count;

	// Once the buffer is filled, ship it to GPU to compute
	if (acc->buf_count >= max_num_bodies_per_compute)
	{
		constexpr unsigned in_bytes = max_num_bodies_per_compute * sizeof(double3);
		constexpr unsigned out_bytes = max_num_bodies_per_compute * sizeof(double2);

		HANDLE_ERROR(cudaMemcpy(acc->dev_pos, acc->pos_buf, in_bytes, cudaMemcpyHostToDevice));

		constexpr int block_size = 256;
		const int n_blocks = (max_num_bodies_per_compute + block_size - 1) / block_size;

		body_force << <n_blocks, block_size >> >(acc->dev_pos, acc->dev_us, max_num_bodies_per_compute);

		HANDLE_ERROR(cudaMemcpy(acc->us_buf, acc->dev_us, out_bytes, cudaMemcpyDeviceToHost));
	}

	return 0;
}
