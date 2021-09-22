#pragma once
#include "CpuGpu.h"
#include "cuda_runtime.h"
#define PIN_LIMIT 4 * 1024 * 1024
#define MEMORY_ALLOC_LIMIT 3.5 * 1024 * 1024 * 1024


void cpuGpuAlloc(CpuGpuMem* p_cg, enum cpuGpuMemVar keyword, int sizeOfType) //CpuGpuMem türünden yapý parametresi alýr, keyword olarak enum alýr, sizeOfType veri türünün boyutunu belirtir.
{
	cudaError_t result;
	switch (keyword)
	{
	case imageEnum:
		p_cg->imageAllocSize = p_cg->imageWidthSize * p_cg->imageHeightSize * sizeOfType; //görüntünün byte türünden tahsis boyutu

		if (p_cg->imageAllocSize < MEMORY_ALLOC_LIMIT) {  //tahsis edilen bölge 3.5 gb'dan fazla olmamalý (GPU belleði 4gb olduðu durumda)
			p_cg->cpuImagePtr = (int*)malloc(p_cg->imageAllocSize); //RAM bellek bölgesi tahsisi
			result = cudaMalloc((int**)&p_cg->gpuImagePtr, p_cg->imageAllocSize); //GPU bellek bölgesi tahsisi
			assert(result == cudaSuccess); // cuda kontrol
		}
		else {
			assert(true);
		}
		break;
	case featureEnum:
		p_cg->featureAllocSize = p_cg->featureWidthSize * p_cg->featureHeightSize * p_cg->maskCount * sizeOfType; //feature spacein byte türünden tahsis boyutu

		if (p_cg->featureAllocSize < MEMORY_ALLOC_LIMIT) {
			p_cg->cpuFeaturePtr = (float*)malloc(p_cg->featureAllocSize);
			result = cudaMalloc((float**)&p_cg->gpuFeaturePtr, p_cg->featureAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
		break;
	case maskEnum:
		p_cg->maskAllocSize = (p_cg->maskWHSize * p_cg->maskWHSize * p_cg->maskCount * p_cg->maskDim + p_cg->maskCount) * sizeOfType; //maskelerin byte türünden tahsis boyutu

		if (p_cg->maskAllocSize < MEMORY_ALLOC_LIMIT) {
			p_cg->cpuMaskPtr = (float*)malloc(p_cg->maskAllocSize);
			result = cudaMalloc((float**)&p_cg->gpuMaskPtr, p_cg->maskAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
		break;
	case batchEnum:
		p_cg->batchWeightSize = p_cg->batchWeightSize * 4 * sizeOfType; //batch aðýrlýklarýnýn byte türünden tahsis boyutu. 4 ile çarpýlmasýnýn sebebi her feature için 
																		//4 parametre(gamma,beta,a.o,varyans) gerekir.

		if (p_cg->batchWeightSize < MEMORY_ALLOC_LIMIT) {
			p_cg->cpuBatchPtr = (float*)malloc(p_cg->batchWeightSize);
			result = cudaMalloc((float**)&p_cg->gpuBatchPtr, p_cg->batchWeightSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
		break;
	case denseEnum:
		p_cg->denseOutputAllocSize = p_cg->denseOutputSize * sizeOfType; //dense çýkýþ katmaný byte türünden tahsis boyutu

		if (p_cg->denseOutputSize < MEMORY_ALLOC_LIMIT) {
			p_cg->cpuDensePtr = (float*)malloc(p_cg->denseOutputAllocSize);
			result = cudaMalloc((float**)&p_cg->gpuDensePtr, p_cg->denseOutputAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
		break;
	case denseWeightEnum:
		p_cg->denseWeightAllocSize = (p_cg->denseOutputSize * p_cg->denseInputSize + p_cg->denseOutputSize )* sizeOfType; //dense aðýrlýklarý byte türünden tahsis boyutu
		 
		if (p_cg->denseWeightAllocSize < MEMORY_ALLOC_LIMIT) {
			p_cg->cpuDenseWeightPtr = (float*)malloc(p_cg->denseWeightAllocSize);
			result = cudaMalloc((float**)&p_cg->gpuDenseWeightPtr, p_cg->denseWeightAllocSize);
			assert(result == cudaSuccess);
		}
		else {
			assert(true);
		}
		break;
	default:
		assert(true);
		break;
	}

}

void cpuGpuFree(CpuGpuMem* p_cg, enum cpuGpuMemVar keyword)
{
	cudaError_t result;
	switch (keyword)
	{
	case imageEnum:
		result = cudaFree(p_cg->gpuImagePtr); //cudafree
		assert(result == cudaSuccess);

		free(p_cg->cpuImagePtr); //cpufree
		break;
	case featureEnum:
		result = cudaFree(p_cg->gpuFeaturePtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuFeaturePtr);
		break;
	case maskEnum:
		result = cudaFree(p_cg->gpuMaskPtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuMaskPtr);
		break;
	case batchEnum:
		result = cudaFree(p_cg->gpuBatchPtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuBatchPtr);
		break;
	case denseEnum:
		result = cudaFree(p_cg->gpuDensePtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuDensePtr);
		break;
	case denseWeightEnum:
		result = cudaFree(p_cg->gpuDenseWeightPtr);
		assert(result == cudaSuccess);

		free(p_cg->cpuDenseWeightPtr);
		break;
	default:
		assert(true);
		break;
	}

}

void cpuGpuPin(void* ptr, int allocSize) //pinlenen gpu bellek bölgesi, gpu tarafýndan daha hýzlý eriþilmesini saðlar. Fakat bellekte fragmentasyona sebep olur.
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

void cpuGpuUnpin(void* ptr, int allocSize) //pinlenen bölgeyi kaldýrýr
{
	const int pinLimit = PIN_LIMIT;

	cudaError_t result;

	int allocation_size = allocSize;

	bool pin = allocation_size > pinLimit;

	if (pin)
	{
		result = cudaHostUnregister(ptr);
		assert(result == cudaSuccess);
	}
}

void cpuGpuMemCopy(enum cudaMemcpyKind copyKind, struct CpuGpuMem* p_cg, void* destPtr, void* srcPtr, int allocSize)
{
	long long int allocation_size = allocSize;

	cudaError_t result;

	switch (copyKind)
	{
	case cudaMemcpyHostToDevice: //RAM bellekten GPU belleðe transfer
		result = cudaMemcpy(destPtr, srcPtr, allocation_size, copyKind);
		assert(result == cudaSuccess);

		break;
	case cudaMemcpyDeviceToHost: //GPU bellekten RAM belleðe transfer
		result = cudaMemcpy(destPtr, srcPtr, allocation_size, copyKind);
		assert(result == cudaSuccess);
		break;
	default:
		abort();
		break;
	}
}
