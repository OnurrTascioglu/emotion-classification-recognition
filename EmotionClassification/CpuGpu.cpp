#pragma once
#include "CpuGpu.h"
#include "cuda_runtime.h"
#define PIN_LIMIT 4 * 1024 * 1024
#define MEMORY_ALLOC_LIMIT 3.5 * 1024 * 1024 * 1024


void cpuGpuAlloc(CpuGpuMem* p_cg, enum cpuGpuMemVar keyword, int sizeOfType) //CpuGpuMem t�r�nden yap� parametresi al�r, keyword olarak enum al�r, sizeOfType veri t�r�n�n boyutunu belirtir.
{
	cudaError_t result;
	switch (keyword)
	{
	case imageEnum:
		p_cg->imageAllocSize = p_cg->imageWidthSize * p_cg->imageHeightSize * sizeOfType; //g�r�nt�n�n byte t�r�nden tahsis boyutu

		if (p_cg->imageAllocSize < MEMORY_ALLOC_LIMIT) {  //tahsis edilen b�lge 3.5 gb'dan fazla olmamal� (GPU belle�i 4gb oldu�u durumda)
			p_cg->cpuImagePtr = (int*)malloc(p_cg->imageAllocSize); //RAM bellek b�lgesi tahsisi
			result = cudaMalloc((int**)&p_cg->gpuImagePtr, p_cg->imageAllocSize); //GPU bellek b�lgesi tahsisi
			assert(result == cudaSuccess); // cuda kontrol
		}
		else {
			assert(true);
		}
		break;
	case featureEnum:
		p_cg->featureAllocSize = p_cg->featureWidthSize * p_cg->featureHeightSize * p_cg->maskCount * sizeOfType; //feature spacein byte t�r�nden tahsis boyutu

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
		p_cg->maskAllocSize = (p_cg->maskWHSize * p_cg->maskWHSize * p_cg->maskCount * p_cg->maskDim + p_cg->maskCount) * sizeOfType; //maskelerin byte t�r�nden tahsis boyutu

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
		p_cg->batchWeightSize = p_cg->batchWeightSize * 4 * sizeOfType; //batch a��rl�klar�n�n byte t�r�nden tahsis boyutu. 4 ile �arp�lmas�n�n sebebi her feature i�in 
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
		p_cg->denseOutputAllocSize = p_cg->denseOutputSize * sizeOfType; //dense ��k�� katman� byte t�r�nden tahsis boyutu

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
		p_cg->denseWeightAllocSize = (p_cg->denseOutputSize * p_cg->denseInputSize + p_cg->denseOutputSize )* sizeOfType; //dense a��rl�klar� byte t�r�nden tahsis boyutu
		 
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

void cpuGpuPin(void* ptr, int allocSize) //pinlenen gpu bellek b�lgesi, gpu taraf�ndan daha h�zl� eri�ilmesini sa�lar. Fakat bellekte fragmentasyona sebep olur.
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

void cpuGpuUnpin(void* ptr, int allocSize) //pinlenen b�lgeyi kald�r�r
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
	case cudaMemcpyHostToDevice: //RAM bellekten GPU belle�e transfer
		result = cudaMemcpy(destPtr, srcPtr, allocation_size, copyKind);
		assert(result == cudaSuccess);

		break;
	case cudaMemcpyDeviceToHost: //GPU bellekten RAM belle�e transfer
		result = cudaMemcpy(destPtr, srcPtr, allocation_size, copyKind);
		assert(result == cudaSuccess);
		break;
	default:
		abort();
		break;
	}
}
