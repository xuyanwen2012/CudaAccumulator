#include "cu_nbody.cuh"
#include "device_launch_parameters.h"

__global__ void body_force(const int* a, const int* b, int* c, const int n)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n)
	{
		c[tid] = a[tid] + b[tid];
	}
}

void compute_with_cuda()
{
	HANDLE_ERROR(cudaSetDevice(0));

	constexpr int n_bodies = 1024;


	int a[n_bodies];
	int b[n_bodies];
	int c[n_bodies];

	for (int i = 0; i < n_bodies; ++i)
	{
		a[i] = i;
		b[i] = n_bodies * 10 + i;
	}

	int* dev_a;
	int* dev_b;
	int* dev_c;

	constexpr int bytes = n_bodies * sizeof(int);
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_a), bytes));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_b), bytes));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_c), bytes));

	HANDLE_ERROR(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));


	constexpr int block_size = 256;
	constexpr int n_blocks = (n_bodies + block_size - 1) / block_size;

	body_force <<<n_blocks, block_size>>>(dev_a, dev_b, dev_c, n_bodies);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost));

	for (int i = 0; i < 10; ++i)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	HANDLE_ERROR(cudaDeviceReset());
}
