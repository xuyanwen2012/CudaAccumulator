#include "accumulator.h"

#include <array>
#include <cuda_runtime_api.h>
#include <vector>
#include <vector_functions.h>

#include "device_launch_parameters.h"

#include "cu_nbody.cuh"

using accumulator_handle = struct accumulator_handle
{
	float* result_addr;

	// The {x, y} of the particle that you are working on. (constants)
	float x;
	float y;

	// CUDA Related
	std::vector<float3> bodies_buf;

	float3* dev_bodies; // stores the {x, y, mass} for all particles i..n
	float2* dev_forces; // stores the temporary forces G_ij of particles
	float2* dev_result; // stores the result of reduced force at [0]
};


float2 kernel_func(const float3 p, const float3 q)
{
	const float dx = p.x - q.x;
	const float dy = p.y - q.y;

	const float dist_sqr = dx * dx + dy * dy + 1e-9f;
	const float inv_dist = rsqrtf(dist_sqr);
	const float inv_dist3 = inv_dist * inv_dist * inv_dist;
	const float with_mass = inv_dist3 * q.z; // z is the mass in this case

	return make_float2(dx * with_mass, dy * with_mass);
}


__global__ void body_compute_forces(const float3 body, const float3* bodies, float2* forces, const size_t n)
{
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		const auto force = kernel_func(body, bodies[tid]);
		forces[tid] = force;
	}
}


__global__ void force_reduction(const float2* forces, float2* result, const size_t n)
{
	constexpr size_t sm_size = 256;
	__shared__ float2 partial_sum[sm_size];

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		partial_sum[threadIdx.x] = forces[tid];
	}
	else
	{
		partial_sum[threadIdx.x] = {0.0f, 0.0f};
	}

	__syncthreads();

	for (size_t s = 1; s < blockDim.x; s *= 2)
	{
		if (threadIdx.x % (2 * s) == 0)
		{
			const auto body_q = partial_sum[threadIdx.x + s];

			partial_sum[threadIdx.x].x += body_q.x;
			partial_sum[threadIdx.x].y += body_q.y;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		result[blockIdx.x] = partial_sum[0];
	}
}


accumulator_handle* get_accumulator()
{
	// TODO: Parametrize this
	constexpr size_t max_num_bodies_per_compute = 1024;

	const auto acc = new accumulator_handle{};

	HANDLE_ERROR(cudaSetDevice(0));

	constexpr unsigned bytes_f3 = max_num_bodies_per_compute * sizeof(float3);
	constexpr unsigned bytes_f2 = max_num_bodies_per_compute * sizeof(float3);

	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_bodies), bytes_f3));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_forces), bytes_f2));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_result), bytes_f2));

	acc->bodies_buf = {};

	return acc;
}


int accumulator_set_constants_and_result_address(const float x, const float y, float* addr, accumulator_handle* acc)
{
	acc->x = x;
	acc->y = y;
	acc->result_addr = addr;
	acc->bodies_buf.clear();

	return 0;
}


int release_accumulator(const accumulator_handle* ret)
{
	cudaFree(ret->dev_bodies);
	cudaFree(ret->dev_forces);
	cudaFree(ret->dev_result);

	HANDLE_ERROR(cudaDeviceReset());
	return 0;
}

std::array<float2, 1> compute_with_cuda(const accumulator_handle* acc, const size_t max_num_bodies_per_compute)
{
	const unsigned bytes_f3 = max_num_bodies_per_compute * sizeof(float3);

	HANDLE_ERROR(cudaMemcpy(acc->dev_bodies, acc->bodies_buf.data(), bytes_f3, cudaMemcpyHostToDevice));

	constexpr size_t block_size = 256;
	const size_t grid_size = (max_num_bodies_per_compute + block_size - 1) / block_size;

	const auto source_body = make_float3(acc->x, acc->y, 1.0f);

	body_compute_forces << <grid_size, block_size >> >(source_body, acc->dev_bodies, acc->dev_forces,
	                                                   max_num_bodies_per_compute);
	force_reduction << <grid_size, block_size >> >(acc->dev_forces, acc->dev_result, max_num_bodies_per_compute);
	force_reduction << <1, block_size >> >(acc->dev_result, acc->dev_result, max_num_bodies_per_compute);


	std::array<float2, 1> result{};
	HANDLE_ERROR(cudaMemcpy(result.data(), acc->dev_result, sizeof(float2), cudaMemcpyDeviceToHost));

	return result;
}

int accumulator_accumulate(const float x, const float y, const float mass, accumulator_handle* acc)
{
	// TODO: parameterize this, right now just make it 1024 all the time.
	constexpr size_t max_num_bodies_per_compute = 1024;

	// Push this to the buffer 
	acc->bodies_buf.push_back(make_float3(x, y, mass));

	if (acc->bodies_buf.size() >= max_num_bodies_per_compute)
	{
		// Once the buffer is filled, ship it to GPU to compute
		const auto result = compute_with_cuda(acc, max_num_bodies_per_compute);

		// Storing the result back 
		float* tmp = acc->result_addr;
		tmp[0] = result[0].x;
		tmp[1] = result[0].y;

		acc->bodies_buf.clear();
	}

	return 0;
}
