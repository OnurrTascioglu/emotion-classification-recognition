#pragma once
#include "CpuGpu.h"
#include "cuda_runtime.h"
#define PIN_LIMIT 4 * 1024 * 1024

long long int get_allocation_size(const int number_count)
{
	return number_count * sizeof(int);
}
void cpuGpuAlloc(CpuGpuMem* p_cg, char keyword)
{

	switch (keyword)
	{
	case 'i':
		long long int allocation_size = p_cg->imageWidthSize * p_cg->imageHeightSize * sizeof(unsigned char);

		if (allocation_size < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuImagePtr = (int*)malloc(allocation_size);
			cudaError_t result = cudaMalloc(&p_cg->gpuImagePtr, allocation_size);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
	case 'f':
		long long int allocation_size = p_cg->featureWidthSize * p_cg->featureHeightSize * p_cg->maskCount * sizeof(float);

		if (allocation_size < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuFeaturePtr = (float*)malloc(allocation_size);
			cudaError_t result = cudaMalloc(&p_cg->gpuFeaturePtr, allocation_size);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
	case 'm':
		long long int allocation_size = (p_cg->maskWHSize * p_cg->maskWHSize * p_cg->maskCount * p_cg->maskDim + p_cg->maskCount) * sizeof(float);

		if (allocation_size < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuMaskPtr = (float*)malloc(allocation_size);
			cudaError_t result = cudaMalloc(&p_cg->gpuMaskPtr, allocation_size);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
	default:
		assert(true);
		break;
	}

}
void cpuGpuFree(CpuGpuMem* p_cg, char keyword)
{

	switch (keyword)
	{
	case 'i':
		cudaError_t result = cudaFree(p_cg->gpuImagePtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuImagePtr); assert(true);

	case 'f':
		cudaError_t result = cudaFree(p_cg->gpuFeaturePtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuFeaturePtr);
	case 'm':
		cudaError_t result = cudaFree(p_cg->gpuMaskPtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuMaskPtr);
	default:
		assert(true);
		break;
	}

}



void cpu_gpu_alloc(CpuGpuMem* p_cg)
{
	long long int allocation_size = get_allocation_size(p_cg->allocSize);
	if (allocation_size < 3.5 * 1024 * 1024 * 1024) {
		p_cg->cpuPtr = malloc(allocation_size);
		cudaError_t result = cudaMalloc(&p_cg->gpuPtr, allocation_size);
		assert(result == cudaSuccess);
	}
	else {
		assert(true);
	}

	//cudaError_t result = cudaHostAlloc(&p_cg->gpu_p, allocation_size,0);//cuda sanal bellek

}
void cpu_gpu_free(CpuGpuMem* p_cg)
{
	cudaError_t result = cudaFree(p_cg->gpuPtr);
	assert(result == cudaSuccess);

	free(p_cg->cpuPtr);

}

void cpu_gpu_set_numbers(CpuGpuMem* p_cg)
{
	int* cpu_int32 = (int*)p_cg->cpuPtr;

	for (int i = 0; i < p_cg->allocSize; i++)
		cpu_int32[i] = i;
}

void cpu_gpu_pin(struct CpuGpuMem* p_cg)
{
	const int pinLimit = PIN_LIMIT;

	int allocation_size = get_allocation_size(p_cg->allocSize);
	cudaError_t result;

	bool pin = allocation_size > pinLimit;

	if (pin)
	{
		result = cudaHostRegister(p_cg->cpuPtr, allocation_size, 0);
		assert(result == cudaSuccess);
	}
}

void cpu_gpu_unpin(struct CpuGpuMem* p_cg)
{
	const int pinLimit = PIN_LIMIT;

	cudaError_t result;

	int allocation_size = get_allocation_size(p_cg->allocSize);

	bool pin = allocation_size > pinLimit;

	if (pin)
	{
		result = cudaHostUnregister(p_cg->cpuPtr);
		assert(result == cudaSuccess);
	}
}

void cpu_gpu_mem_cpy(enum cudaMemcpyKind copyKind, struct CpuGpuMem* p_cg)
{
	long long int allocation_size = get_allocation_size(p_cg->allocSize);

	cudaError_t result;

	switch (copyKind)
	{
	case cudaMemcpyHostToDevice:
		result = cudaMemcpyAsync(p_cg->gpuPtr, p_cg->cpuPtr, allocation_size, copyKind, p_cg->stream);
		assert(result == cudaSuccess);
		break;
	case cudaMemcpyDeviceToHost:
		result = cudaMemcpyAsync(p_cg->cpuPtr, p_cg->gpuPtr, allocation_size, copyKind, p_cg->stream);
		assert(result == cudaSuccess);
		break;
	default:
		abort();
		break;
	}
}

void cpu_gpu_h_to_d(CpuGpuMem* p_cg)
{
	cpu_gpu_mem_cpy(cudaMemcpyHostToDevice, p_cg);
}

void cpu_gpu_d_to_h(CpuGpuMem* p_cg)
{
	cpu_gpu_mem_cpy(cudaMemcpyDeviceToHost, p_cg);
}
//
//void cpu_gpu_print_results(CpuGpuMem* p_cg)
//{
//	int* cpu_int32 = (int*)p_cg->cpuPtr;
//
//	for (int i = p_cg->allocSize - 100; i < p_cg->allocSize; i++)
//		printf("%d\t%d\n", i, cpu_int32[i]);
//}