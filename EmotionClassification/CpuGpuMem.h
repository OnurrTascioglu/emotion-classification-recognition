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

	float* cpuDtoFeaturePtr; //for feature
	float* gpuDtoFeaturePtr;
	int dtoFeatureWidthSize;
	int dtoFeatureHeightSize;
	long long int dtoFeatureAllocSize;

	float* cpuMaskPtr; //for mask
	float* gpuMaskPtr;
	int maskWHSize;
	long long int maskAllocSize;
	int maskCount;
	int maskDim;

	float* gpuBatchPtr;
	float* cpuBatchPtr;
	int batchWeightSize;

	int stride;
	int pool;


	cudaStream_t stream;
};
