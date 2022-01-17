#include "cu_nbody.cuh"

__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}

void compute_with_cuda()
{
	HANDLE_ERROR(cudaSetDevice(0));

	int c;
	int* dev_c;
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_c), sizeof(int)));

	add<<<1,1>>>(2, 7, dev_c);

	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2+7=%d\n", c);
	cudaFree(dev_c);

	HANDLE_ERROR(cudaDeviceReset());
}
