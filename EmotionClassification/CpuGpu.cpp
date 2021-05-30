#pragma once
#include "CpuGpu.h"
#include "cuda_runtime.h"
#define PIN_LIMIT 4 * 1024 * 1024


void cpuGpuAlloc(CpuGpuMem* p_cg, char keyword, int sizeOfType)
{
	cudaError_t result;
	switch (keyword)
	{
	case 'i':
		p_cg->imageAllocSize = p_cg->imageWidthSize * p_cg->imageHeightSize * sizeOfType;

		if (p_cg->imageAllocSize < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuImagePtr = (int*)malloc(p_cg->imageAllocSize);
			result = cudaMalloc(&p_cg->gpuImagePtr, p_cg->imageAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
	case 'f':
		p_cg->featureAllocSize = p_cg->featureWidthSize * p_cg->featureHeightSize * p_cg->maskCount * sizeOfType;

		if (p_cg->featureAllocSize < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuFeaturePtr = (float*)malloc(p_cg->featureAllocSize);
			result = cudaMalloc(&p_cg->gpuFeaturePtr, p_cg->featureAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
	case 'm':
		p_cg->maskAllocSize = (p_cg->maskWHSize * p_cg->maskWHSize * p_cg->maskCount * p_cg->maskDim + p_cg->maskCount) * sizeOfType;

		if (p_cg->maskAllocSize < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuMaskPtr = (float*)malloc(p_cg->maskAllocSize);
			result = cudaMalloc(&p_cg->gpuMaskPtr, p_cg->maskAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
	case 'b':
		p_cg->batchWeightSize = p_cg->maskCount * 4  * sizeOfType;

		if (p_cg->batchWeightSize < 3.5 * 1024 * 1024 * 1024) {
			p_cg->cpuBatchPtr = (float*)malloc(p_cg->batchWeightSize);
			result = cudaMalloc(&p_cg->gpuBatchPtr, p_cg->batchWeightSize);
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
	cudaError_t result;
	switch (keyword)
	{
	case 'i':
		result = cudaFree(p_cg->gpuImagePtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuImagePtr); assert(true);
		break;
	case 'f':
		result = cudaFree(p_cg->gpuFeaturePtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuFeaturePtr);
		break;
	case 'm':
		result = cudaFree(p_cg->gpuMaskPtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuMaskPtr);
		break;
	case 'b':
		result = cudaFree(p_cg->gpuBatchPtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuBatchPtr);
		break;
	default:
		assert(true);
		break;
	}

}

void cpuGpuPin(void* ptr, int allocSize)
{
	const int pinLimit = PIN_LIMIT;

	int allocation_size = allocSize;
	cudaError_t result;

	bool pin = allocation_size > pinLimit;

	if (pin)
	{
		result = cudaHostRegister(ptr, allocation_size, 0);
		assert(result == cudaSuccess);
	}
}

void cpuGpuUnpin(void* ptr, int allocSize)
{
	const int pinLimit = PIN_LIMIT;

	cudaError_t result;

	int allocation_size = allocSize ;

	bool pin = allocation_size > pinLimit;

	if (pin)
	{
		result = cudaHostUnregister(ptr);
		assert(result == cudaSuccess);
	}
}

void cpuGpuMemCopy(enum cudaMemcpyKind copyKind, struct CpuGpuMem* p_cg,void* destPtr, void* srcPtr, int allocSize)
{
	long long int allocation_size = allocSize ;

	cudaError_t result;

	switch (copyKind)
	{
	case cudaMemcpyHostToDevice:
		result = cudaMemcpyAsync(destPtr, srcPtr, allocation_size, copyKind, p_cg->stream);
		assert(result == cudaSuccess);

		break;
	case cudaMemcpyDeviceToHost:
		result = cudaMemcpyAsync(destPtr, srcPtr, allocation_size, copyKind, p_cg->stream);
		assert(result == cudaSuccess);
		break;
	default:
		abort();
		break;
	}
}
