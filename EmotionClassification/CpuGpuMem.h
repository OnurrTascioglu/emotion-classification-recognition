#pragma once
#include "driver_types.h"
struct CpuGpuMem {

	void* cpuImagePtr; //for image
	void* gpuImagePtr;
	int imageWidthSize;
	int imageHeightSize;
	long long int imageAllocSize;

	float* cpuFeaturePtr; //for feature
	float* gpuFeaturePtr;

	int featureWidthSize;
	int featureHeightSize;
	long long int featureAllocSize;

	float* cpuMaskPtr; //for mask
	float* gpuMaskPtr;
	int maskWHSize;
	long long int maskAllocSize;
	int maskCount;
	int maskDim;

	float* gpuBatchPtr;
	float* cpuBatchPtr;
	long long int batchWeightSize;

	float* gpuTempLayer;
	float* gpuTempLayer2;

	float* cpuDensePtr;
	float* gpuDensePtr;
	float* cpuDenseWeightPtr;
	float* gpuDenseWeightPtr;
	int denseInputSize;
	int denseOutputSize;
	long long int denseInputAllocSize;
	long long int denseOutputAllocSize;
	long long int denseWeightAllocSize;


	int stride;
	int pool;


	cudaStream_t stream;
};

enum cpuGpuMemVar
{
	imageEnum = 0,
	featureEnum = 1,
	maskEnum = 2,
	dtoFeatureEnum = 3,
	batchEnum = 4,
	denseEnum = 5,
	denseWeightEnum = 6,
};