#include "device_launch_parameters.h"

#include "CpuGpuMem.h"
#include "KernelGpuAdd.cuh"

__global__ void gpu_add(int* gpu_numbers, const int nc)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < nc)
		gpu_numbers[id] *= 2;
}

void cpuGpuExecute(CpuGpuMem* cg)
{
	int number_count = cg->allocSize;

	int blockDim = 64;
	int gridDim = (number_count + blockDim - 1) / blockDim;

	//execute
	for (size_t i = 0; i < 4; i++)
		gpu_add << <gridDim, blockDim, 0, cg->stream >> > ((int*)cg->gpuPtr, number_count);
}
