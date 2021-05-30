#pragma once
#include "driver_types.h"
struct CpuGpuMem {

	void* cpuPtr;
	void* gpuPtr;
	int allocSize;

	int* cpuImagePtr; //for image
	int* gpuImagePtr;
	int imageWidthSize;
	int imageHeightSize;

	float* cpuFeaturePtr; //for feature
	float* gpuFeaturePtr;
	int featureWidthSize;
	int featureHeightSize;

	float* cpuMaskPtr; //for mask
	float* gpuMaskPtr;
	int maskWHSize;
	int maskCount;
	int maskDim;



	cudaStream_t stream;
};
