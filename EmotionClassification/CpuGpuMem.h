#pragma once
#include "driver_types.h"
struct CpuGpuMem {

	void* cpuImagePtr; //G�r�nt� i�in gerekli datalar
	void* gpuImagePtr; //void veri tipi ile gelen g�r�nt�n�n data type istenen �ekilde se�ilebilir
	int imageWidthSize;
	int imageHeightSize;
	long long int imageAllocSize;

	float* cpuFeaturePtr; //feature space i�in gerekli datalar
	float* gpuFeaturePtr;
	int featureWidthSize;
	int featureHeightSize;
	long long int featureAllocSize;

	float* cpuMaskPtr; //maskeler i�in gerekli datalar
	float* gpuMaskPtr;
	int maskWHSize;
	long long int maskAllocSize;
	int maskCount;
	int maskDim;

	float* gpuBatchPtr; //batch a��rl�klar� i�in gerekli datalar
	float* cpuBatchPtr;
	long long int batchWeightSize;

	float* gpuTempLayer; //Katmanlar aras� data transferi i�in gerekli ge�ici gpu b�lgeleri
	float* gpuTempLayer2;

	float* cpuDensePtr; //dense katman� i�in gerekli datalar
	float* gpuDensePtr;
	float* cpuDenseWeightPtr;
	float* gpuDenseWeightPtr;
	int denseInputSize;
	int denseOutputSize;
	long long int denseInputAllocSize;
	long long int denseOutputAllocSize;
	long long int denseWeightAllocSize;


	int stride;  //maxpool i�lemi i�in
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