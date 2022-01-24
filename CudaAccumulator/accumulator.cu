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
	size_t max_num_bodies_per_compute;
	std::vector<float3> bodies_buf;

	float3* dev_bodies; // stores the {x, y, mass} for all particles i..n
	float2* dev_forces; // stores the temporary forces G_ij of particles
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


__global__ void body_compute_forces(const float3 body, const float3* bodies, float2* forces, const size_t n)
{
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		const auto force = kernel_func_gpu(body, bodies[tid]);
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


std::array<float2, 1> compute_with_cuda(const accumulator_handle* acc, const unsigned n)
{
	const unsigned bytes_f3 = n * sizeof(float3);

	HANDLE_ERROR(cudaMemcpy(acc->dev_bodies, acc->bodies_buf.data(), bytes_f3, cudaMemcpyHostToDevice));

	// So I think the block size is what the max thread of a block
	constexpr unsigned block_size = 256;
	const unsigned grid_size = (n + block_size - 1) / block_size;

	const auto source_body = make_float3(acc->x, acc->y, 1.0f);

	body_compute_forces << <grid_size, block_size >> >(source_body,
	                                                   acc->dev_bodies,
	                                                   acc->dev_forces,
	                                                   n);

	//printf("----Debug-----\n");
	//constexpr unsigned n_to_ins = 10;
	//std::array<float2, n_to_ins> inspect_segment{};
	//HANDLE_ERROR(cudaMemcpy(inspect_segment.data(), acc->dev_forces, n_to_ins * sizeof(float2), cudaMemcpyDeviceToHost));

	//for (unsigned i = 0; i < n_to_ins; ++i)
	//{
	//	printf("%f,%f\n", inspect_segment[i].x, inspect_segment[i].y);
	//}

	//printf("----Debug-----\n");


	force_reduction << <grid_size, block_size >> >(acc->dev_forces, acc->dev_result, n);
	force_reduction << <1, block_size >> >(acc->dev_result, acc->dev_result, n);


	std::array<float2, 1> result{};
	HANDLE_ERROR(cudaMemcpy(result.data(), acc->dev_result, sizeof(float2), cudaMemcpyDeviceToHost));

	return result;
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
	acc->max_num_bodies_per_compute = max_num_bodies_per_compute;

	return acc;
}


int check_and_clear_current_buffer(accumulator_handle* acc)
{
	if (!acc->bodies_buf.empty())
	{
		// Finished the remaining bodies in the buffer before switching context
		uint32_t remaining_n = acc->bodies_buf.size();

		float2 rem_force = {};

		while (remaining_n > 256)
		{
			const uint32_t previous_pow_of_2 = flp2(remaining_n);
			// TODO: currently using ugly solution, need to ask Tyler
			cudaMemset(acc->dev_result, 0, acc->max_num_bodies_per_compute * sizeof(float2));

			const auto result = compute_with_cuda(acc, previous_pow_of_2);
			rem_force.x += result[0].x;
			rem_force.y += result[0].y;

			remaining_n -= previous_pow_of_2;

			// drop the first 'previous_pow_of_2' particles
			std::vector<decltype(acc->bodies_buf)::value_type>(acc->bodies_buf.begin() + previous_pow_of_2,
			                                                   acc->bodies_buf.end()).swap(acc->bodies_buf);

			// DEBUG LOG
			//printf("-- previous_pow_of_2 %d, remaining_n: %d\n", previous_pow_of_2, remaining_n);
		}


		// Do the rest on CPU ( < 256)
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
	}

	return 0;
}


int accumulator_set_constants_and_result_address(const float x, const float y, float* addr, accumulator_handle* acc)
{
	check_and_clear_current_buffer(acc);

	acc->x = x;
	acc->y = y;
	acc->result_addr = addr;
	acc->bodies_buf.clear();

	return 0;
}


int release_accumulator(accumulator_handle* ret)
{
	check_and_clear_current_buffer(ret);

	cudaFree(ret->dev_bodies);
	cudaFree(ret->dev_forces);
	cudaFree(ret->dev_result);

	HANDLE_ERROR(cudaDeviceReset());
	return 0;
}


int accumulator_accumulate(const float x, const float y, const float mass, accumulator_handle* acc)
{
	// Push this to the buffer 
	acc->bodies_buf.push_back(make_float3(x, y, mass));

	if (acc->bodies_buf.size() >= acc->max_num_bodies_per_compute)
	{
		// Once the buffer is filled, ship it to GPU to compute
		const auto result = compute_with_cuda(acc, acc->max_num_bodies_per_compute);

		// Storing the result back 
		float* tmp = acc->result_addr;
		tmp[0] += result[0].x;
		tmp[1] += result[0].y;

		acc->bodies_buf.clear();
	}

	return 0;
}
