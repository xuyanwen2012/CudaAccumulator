#include "accumulator.h"

#include <array>
#include <cuda_runtime_api.h>
#include <vector>
#include <vector_functions.h>

#include "device_launch_parameters.h"

#include "cu_nbody.cuh"


constexpr unsigned max_num_bodies_per_compute = 1024;


using accumulator_handle = struct accumulator_handle
{
	float* result_addr;

	// The {x, y} of the particle that you are working on. (constants)
	float x;
	float y;

	// CUDA Related
	std::vector<float3> bodies_buf;

	float3* dev_bodies; // stores the {x, y, mass} for all particles i..n
	float2* dev_result; // stores the result of reduced force at [0]
};


/**
 * \brief find the previous power of 2 of this number
 * https://stackoverflow.com/questions/2679815/previous-power-of-2
 * \param x the number to round down
 * \return the previous power of 2
 */
uint32_t flp2(uint32_t x)
{
	x = x | x >> 1;
	x = x | x >> 2;
	x = x | x >> 4;
	x = x | x >> 8;
	x = x | x >> 16;
	return x - (x >> 1);
}


__device__ float2 kernel_func_gpu(const float3 p, const float3 q)
{
	const float dx = p.x - q.x;
	const float dy = p.y - q.y;

	const float dist_sqr = dx * dx + dy * dy + 1e-9f;
	const float inv_dist = rsqrtf(dist_sqr);
	const float inv_dist3 = inv_dist * inv_dist * inv_dist;
	const float with_mass = inv_dist3 * q.z; // z is the mass in this case

	return make_float2(dx * with_mass, dy * with_mass);
}


float2 kernel_func_cpu(const float dx, const float dy, const float mass)
{
	const float dist_sqr = dx * dx + dy * dy + 1e-9f;
	const float inv_dist = 1.0f / sqrtf(dist_sqr);
	const float inv_dist3 = inv_dist * inv_dist * inv_dist;
	const float with_mass = inv_dist3 * mass; // z is the mass in this case

	return make_float2(dx * with_mass, dy * with_mass);
}


//TODO: prepare 32-1024
__global__ void force_reduction(const float3 source_body, const float3* bodies, float2* result, const size_t n)
{
	__shared__ float2 partial_sum[max_num_bodies_per_compute];

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = kernel_func_gpu(source_body, bodies[tid]);

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

__global__ void force_reduction_2(const float3 source_body, const float3* bodies, float2* result, const size_t n)
{
	__shared__ float2 partial_sum[max_num_bodies_per_compute];

	const unsigned int tid = threadIdx.x;
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[tid] = i < n ? kernel_func_gpu(source_body, bodies[i]) : float2{};

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			const auto body_q = partial_sum[tid + s];

			partial_sum[tid].x += body_q.x;
			partial_sum[tid].y += body_q.y;
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		result[blockIdx.x] = partial_sum[0];
	}
}


std::array<float2, 1> compute_with_cuda(const accumulator_handle* acc, const unsigned n)
{
	// printf("	debug: shipped %d to GPU \n", n);

	const unsigned bytes_f3 = n * sizeof(float3);

	HANDLE_ERROR(cudaMemcpy(acc->dev_bodies, acc->bodies_buf.data(), bytes_f3, cudaMemcpyHostToDevice));

	const unsigned block_size = n;
	constexpr unsigned grid_size = 1;

	const auto source_body = make_float3(acc->x, acc->y, 1.0f);

	force_reduction_2 << <grid_size, block_size >> >(source_body, acc->dev_bodies, acc->dev_result, n);

	std::array<float2, 1> result{};
	HANDLE_ERROR(cudaMemcpy(result.data(), acc->dev_result, sizeof(float2), cudaMemcpyDeviceToHost));

	return result;
}


accumulator_handle* get_accumulator()
{
	const auto acc = new accumulator_handle{};

	HANDLE_ERROR(cudaSetDevice(0));

	constexpr unsigned bytes_f3 = max_num_bodies_per_compute * sizeof(float3);
	constexpr unsigned bytes_f2 = max_num_bodies_per_compute * sizeof(float3);

	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_bodies), bytes_f3));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&acc->dev_result), bytes_f2));

	acc->bodies_buf = {};

	return acc;
}


int check_and_clear_current_buffer(accumulator_handle* acc)
{
	// Finished the remaining bodies in the buffer before switching context
	if (!acc->bodies_buf.empty())
	{
		auto remaining_n = static_cast<uint32_t>(acc->bodies_buf.size());

		float2 rem_force = {};

		while (remaining_n >= 32)
		{
			const uint32_t previous_pow_of_2 = flp2(remaining_n);

			const auto result = compute_with_cuda(acc, previous_pow_of_2);
			rem_force.x += result[0].x;
			rem_force.y += result[0].y;

			remaining_n -= previous_pow_of_2;

			// drop the first 'previous_pow_of_2' particles
			std::vector<decltype(acc->bodies_buf)::value_type>(acc->bodies_buf.begin() + previous_pow_of_2,
			                                                   acc->bodies_buf.end()).swap(acc->bodies_buf);
		}

		// Do the rest on CPU ( < 32)
		for (unsigned j = 0; j < remaining_n; ++j)
		{
			const auto dx = acc->x - acc->bodies_buf[j].x;
			const auto dy = acc->y - acc->bodies_buf[j].y;
			const auto mass = acc->bodies_buf[j].z;

			const auto result = kernel_func_cpu(dx, dy, mass);
			rem_force.x += result.x;
			rem_force.y += result.y;
		}

		float* tmp = acc->result_addr;
		tmp[0] += rem_force.x;
		tmp[1] += rem_force.y;

		//if (remaining_n > 0)
		//{
		//	printf("	debug: %d was done on CPU, force is now %f,%f \n", remaining_n, rem_force.x, rem_force.y);
		//}
	}

	return 0;
}


int accumulator_set_constants_and_result_address(const float x, const float y, float* addr, accumulator_handle* acc)
{
	acc->x = x;
	acc->y = y;
	acc->result_addr = addr;
	acc->bodies_buf.clear();

	return 0;
}


int release_accumulator(const accumulator_handle* acc)
{
	cudaFree(acc->dev_bodies);
	cudaFree(acc->dev_result);

	HANDLE_ERROR(cudaDeviceReset());
	return 0;
}


int accumulator_accumulate(const float x, const float y, const float mass, accumulator_handle* acc)
{
	// Push this to the buffer 
	acc->bodies_buf.push_back(make_float3(x, y, mass));

	if (acc->bodies_buf.size() >= max_num_bodies_per_compute)
	{
		const auto result = compute_with_cuda(acc, max_num_bodies_per_compute);

		// Storing the result back 
		float* tmp = acc->result_addr;
		tmp[0] += result[0].x;
		tmp[1] += result[0].y;

		acc->bodies_buf.clear();
	}

	return 0;
}

int accumulator_finish(accumulator_handle* acc)
{
	check_and_clear_current_buffer(acc);

	return 0;
}
