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
	float2* uni_results; // only one of this
};


float2 kernel_func_cpu(const float dx, const float dy, const float mass)
{
	const float dist_sqr = dx * dx + dy * dy + 1e-9f;
	const float inv_dist = 1.0f / sqrtf(dist_sqr);
	const float inv_dist3 = inv_dist * inv_dist * inv_dist;
	//const float with_mass = inv_dist3 * mass; // z is the mass in this case
	return make_float2(dx * inv_dist3, dy * inv_dist3);
}


constexpr int kMaxNumPerBlock = 32;

inline __device__ float2 KernelFuncGpu(const float3 p, const float3 q)
{
	const auto dx = p.x - q.x;
	const auto dy = p.y - q.y;
	const auto dist_sqr = dx * dx + dy * dy + 1e-9f;
	const auto inv_dist = rsqrtf(dist_sqr);
	const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
	//const auto with_mass = inv_dist * p.z; // z is mass
	return make_float2(dx * inv_dist3, dy * inv_dist3);
}

__global__ void ReduceForcesGpu(const float3 source_point, const float3* data,
                                float2* result, const size_t n)
{
	__shared__ float2 partial_sum[kMaxNumPerBlock];

	const unsigned int tid = threadIdx.x;
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[tid] = i < n ? KernelFuncGpu(data[i], source_point) : float2();

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
		result[blockIdx.x].x += partial_sum[0].x;
		result[blockIdx.x].y += partial_sum[0].y;
	}
}


accumulator_handle* get_accumulator()
{
	const auto acc = new accumulator_handle{};

	HANDLE_ERROR(cudaSetDevice(0));

	constexpr unsigned bytes_f3 = max_num_bodies_per_compute * sizeof(float3);
	constexpr unsigned bytes_f2 = sizeof(float2);

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
		float2 rem_force = {};

		const auto source_body = make_float3(acc->x, acc->y, 1.0f);
		const auto num_to_ship = acc->body_count;

		const auto prev_mult_of_32 = num_to_ship / 32 * 32;

		acc->uni_results[0].x = 0.0;
		acc->uni_results[0].y = 0.0;

		const size_t n_iters = prev_mult_of_32 / kMaxNumPerBlock;

		for (size_t i = 0; i < n_iters; ++i)
		{
			constexpr unsigned block_size = kMaxNumPerBlock;
			constexpr unsigned grid_size = 1;
			ReduceForcesGpu<<<grid_size, block_size>>>(
				source_body, acc->uni_bodies + i * kMaxNumPerBlock, acc->uni_results,
				kMaxNumPerBlock);
		}

		cudaDeviceSynchronize();

		rem_force.x = acc->uni_results[0].x;
		rem_force.y = acc->uni_results[0].y;

		const auto rem_num = num_to_ship - prev_mult_of_32;
		for (size_t i = 0; i < rem_num; ++i)
		{
			const auto dx = acc->x - acc->uni_bodies[prev_mult_of_32 + i].x;
			const auto dy = acc->y - acc->uni_bodies[prev_mult_of_32 + i].y;
			const auto mass = acc->uni_bodies[prev_mult_of_32 + i].z;

			const auto rem_result = kernel_func_cpu(dx, dy, mass);
			rem_force.x += rem_result.x;
			rem_force.y += rem_result.y;
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

	return 0;
}

int accumulator_finish(accumulator_handle* acc)
{
	check_and_clear_current_buffer(acc);

	return 0;
}
