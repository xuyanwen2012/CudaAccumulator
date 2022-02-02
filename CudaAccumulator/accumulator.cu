#include "accumulator.h"

#include <array>
#include <cuda_runtime_api.h>
#include <map>
#include <vector>
#include <vector_functions.h>

#include "device_launch_parameters.h"

constexpr unsigned max_num_bodies_per_compute = 1024;

int running_count;
long long avg_reduction_size;
int num_reductions;

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


using accumulator_handle = struct accumulator_handle
{
	float* result_addr;

	// The {x, y} of the particle that you are working on. (constants)
	float x;
	float y;

	// CUDA Related
	unsigned int body_count;
	float3* uni_bodies;
	float2* uni_results;
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


unsigned get_previous_pow_of_2(unsigned n)
{
	static std::map<unsigned, unsigned> lookup_table{};

	const auto val = lookup_table.find(n);
	if (val == lookup_table.end())
	{
		const unsigned v = flp2(n);
		lookup_table.insert(std::make_pair(n, v));
		return v;
	}

	return val->second;
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


float2 compute_with_cuda(const accumulator_handle* acc, const unsigned array_start, const unsigned n)
{
	const unsigned block_size = n;
	constexpr unsigned grid_size = 1;
	const auto source_body = make_float3(acc->x, acc->y, 1.0f);

	force_reduction_2 << <grid_size, block_size >> >(source_body, acc->uni_bodies + array_start, acc->uni_results, n);

	cudaDeviceSynchronize();

	return acc->uni_results[0];
}


accumulator_handle* get_accumulator()
{
	const auto acc = new accumulator_handle{};

	HANDLE_ERROR(cudaSetDevice(0));

	constexpr unsigned bytes_f3 = max_num_bodies_per_compute * sizeof(float3);
	constexpr unsigned bytes_f2 = max_num_bodies_per_compute * sizeof(float3);

	HANDLE_ERROR(cudaMallocManaged(reinterpret_cast<void**>(&acc->uni_bodies), bytes_f3));
	HANDLE_ERROR(cudaMallocManaged(reinterpret_cast<void**>(&acc->uni_results), bytes_f2));

	acc->body_count = 0;

	return acc;
}


int check_and_clear_current_buffer(const accumulator_handle* acc)
{
	// Finished the remaining bodies in the buffer before switching context
	if (acc->body_count != 0)
	{
		auto remaining_n = acc->body_count;

		float2 rem_force = {};

		unsigned current_index = 0;

		while (remaining_n >= 32)
		{
			const auto num_to_ship = get_previous_pow_of_2(remaining_n); //(remaining_n/32)*32;

			const auto result = compute_with_cuda(acc, current_index, num_to_ship);
			rem_force.x += result.x;
			rem_force.y += result.y;

			//static int count = 0;
			//printf("shipment %d: %d\n", count, previous_pow_of_2);
			//++count;

			remaining_n -= num_to_ship;

			// drop the first 'previous_pow_of_2' particles
			current_index += num_to_ship;
		}

		// Do the rest on CPU ( < 32)
		for (unsigned j = current_index; j < current_index + remaining_n; ++j)
		{
			const auto dx = acc->x - acc->uni_bodies[j].x;
			const auto dy = acc->y - acc->uni_bodies[j].y;
			const auto mass = acc->uni_bodies[j].z;

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
	avg_reduction_size += running_count;
	running_count = 0;
	++num_reductions;

	acc->x = x;
	acc->y = y;
	acc->result_addr = addr;
	acc->body_count = 0;

	return 0;
}


int release_accumulator(accumulator_handle* acc)
{
	cudaFree(acc->uni_bodies);
	cudaFree(acc->uni_results);

	free(acc);

	avg_reduction_size += running_count;
	avg_reduction_size /= num_reductions;
	printf("avg_reduction_elements: %lld\n", avg_reduction_size);

	HANDLE_ERROR(cudaDeviceReset());
	return 0;
}


int accumulator_accumulate(const float x, const float y, const float mass, accumulator_handle* acc)
{
	++running_count;

	// Push this to the buffer 
	acc->uni_bodies[acc->body_count] = make_float3(x, y, mass);
	++acc->body_count;

	if (acc->body_count >= max_num_bodies_per_compute)
	{
		const auto result = compute_with_cuda(acc, 0, max_num_bodies_per_compute);

		// Storing the result back 
		float* tmp = acc->result_addr;
		tmp[0] += result.x;
		tmp[1] += result.y;

		acc->body_count = 0;
	}

	return 0;
}

int accumulator_finish(accumulator_handle* acc)
{
	check_and_clear_current_buffer(acc);

	return 0;
}
