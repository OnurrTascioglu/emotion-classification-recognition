#include <windows.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "device_launch_parameters.h"

#include "CpuGpu.h"
#include "CpuGpuMem.h"
#include "KernelGpuAdd.cuh"
#include "cuda_runtime.h"

#define BIAS 1

//----------------conv1
__global__ void conv1GPU(float* resultImages, float* masks, int* image, int width, int height, int maskSize, int rMatrixWidth, int rMatrixHeight, int maskCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < maskCount * rMatrixWidth * rMatrixHeight) {
		int temp = 0;
		int j = id % rMatrixWidth;
		temp = id / rMatrixWidth;
		int i = temp % rMatrixHeight;
		int m = temp / rMatrixHeight;

		for (int k = 0; k < maskSize * maskSize; k++) {
			int mCol = k % maskSize;
			int mRow = k / maskSize;
			resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] +=
				(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];
		}
		resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] += (float)BIAS * masks[maskCount * (maskSize * maskSize) + m];
	}
}

__global__ void batchNormGPU(float* feature, float* batchWeights, int width, int height, int featureCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < featureCount * width * height) {
		int i = id % (width * height);
		int m = id / (width * height);

		float sDeviation = 0.0; // standart sapma için

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]);

		feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation;
		feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];

		if (fabs(feature[(m * width * height) + i]) + feature[(m * width * height) + i] < 0.0001) {
			feature[(m * width * height) + i] = 0.0;
		}
	}
}

__global__ void maxPoolingGPU(float* feature, float* tempFeature, int width, int height, int  featureCount, int pool, int stride)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;


	if (id < featureCount * (width / stride) * (height / stride)) {
		int temp2 = 0;
		int col = id % (width / stride);
		temp2 = id / (width / stride);
		int row = temp2 % (height / stride);
		int m = temp2 / (height / stride);

		float max = 0.0;
		float temp = 0.0;



		for (int k = 0; k < pool; k++) {
			for (int n = 0; n < pool; n++) {
				temp = feature[(m * width * height) + row * width * stride + col * stride + k * width + n];
				if ((temp - max) > 0.0001) {
					max = temp;
				}
			}
		}
		tempFeature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max;

	}

}

//-----------------------conv2


__global__ void convHiddenGPU(float* resultImages, float* feature, float* weights, int fWidth, int fHeight, int maskSize, int maskCount, int maskDim)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int rMatrixWidth = fWidth - maskSize + 1;
	int rMatrixHeight = fHeight - maskSize + 1;

	if (id < maskCount * rMatrixWidth * rMatrixHeight) {
		int temp = 0;
		int j = id % rMatrixWidth;
		temp = id / rMatrixWidth;
		int i = temp % rMatrixHeight;
		int c = temp / rMatrixHeight;

		for (int d = 0; d < maskDim; d++) {
			for (int k = 0; k < maskSize * maskSize; k++) {
				int mCol = k % maskSize;
				int mRow = k / maskSize;
				resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] +=
					(float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * weights[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k];

			}
		}
		resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] += BIAS * weights[maskCount * maskDim * maskSize * maskSize + c];
	}
}



void convHidden1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;
	int dfhs = cg->dtoFeatureHeightSize;
	int dfws = cg->dtoFeatureWidthSize;


	int blockDim = 1024;
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1);
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	convHiddenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDtoFeaturePtr, cg->gpuFeaturePtrTemp, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDtoFeaturePtr, cg->gpuBatchPtr, cg->dtoFeatureWidthSize, cg->dtoFeatureHeightSize, cg->maskCount);


	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	result = cudaFree(cg->gpuFeaturePtrTemp);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuFeaturePtrTemp, threadCount * sizeof(float));
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDtoFeaturePtr, cg->gpuFeaturePtrTemp, cg->dtoFeatureWidthSize, cg->dtoFeatureHeightSize, cg->maskCount, cg->pool, cg->stride);
	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDtoFeaturePtr, cg->gpuFeaturePtrTemp, threadCount * sizeof(float));

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

}


void conv1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int iws = cg->imageWidthSize;
	int ihs = cg->imageHeightSize;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize = iws - ms + 1;
	int fhs = cg->featureHeightSize = ihs - ms + 1;

	int blockDim = 1024;
	int threadCount = cg->maskCount * fws * fhs;


	int gridDim = (threadCount + blockDim - 1) / blockDim;


	conv1GPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuMaskPtr, (int*)cg->gpuImagePtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//temp array for pooling result
	result = cudaMalloc((float**)&cg->gpuFeaturePtrTemp, threadCount * sizeof(float));
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuFeaturePtrTemp, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

}



