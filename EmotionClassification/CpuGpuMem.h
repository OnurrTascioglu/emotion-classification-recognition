#pragma once
#include "driver_types.h"
struct CpuGpuMem {

	void* cpuImagePtr; //Görüntü için gerekli datalar
	void* gpuImagePtr; //void veri tipi ile gelen görüntünün data type istenen þekilde seçilebilir
	int imageWidthSize;
	int imageHeightSize;
	long long int imageAllocSize;

	float* cpuFeaturePtr; //feature space için gerekli datalar
	float* gpuFeaturePtr;
	int featureWidthSize;
	int featureHeightSize;
	long long int featureAllocSize;

	float* cpuMaskPtr; //maskeler için gerekli datalar
	float* gpuMaskPtr;
	int maskWHSize;
	long long int maskAllocSize;
	int maskCount;
	int maskDim;

	float* gpuBatchPtr; //batch aðýrlýklarý için gerekli datalar
	float* cpuBatchPtr;
	long long int batchWeightSize;

	float* gpuTempLayer; //Katmanlar arasý data transferi için gerekli geçici gpu bölgeleri
	float* gpuTempLayer2;

	float* cpuDensePtr; //dense katmaný için gerekli datalar
	float* gpuDensePtr;
	float* cpuDenseWeightPtr;
	float* gpuDenseWeightPtr;
	int denseInputSize;
	int denseOutputSize;
	long long int denseInputAllocSize;
	long long int denseOutputAllocSize;
	long long int denseWeightAllocSize;


	int stride;  //maxpool iþlemi için
	int pool;

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