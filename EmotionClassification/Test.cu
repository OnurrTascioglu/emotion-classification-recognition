#include <windows.h>
#include <cmath>

#include "device_launch_parameters.h"

#include "CpuGpuMem.h"
#include "KernelGpuAdd.cuh"
#include "cuda_runtime.h"

#define BIAS 1

__global__ void conv1GPU(float* resultImages, float* masks, int* image, int width, int height, int maskSize, int rMatrixWidth, int rMatrixHeight, int maskCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if(id < maskCount * rMatrixWidth * rMatrixHeight ) {
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

__global__ void batchConv1GPU(float* feature, float* batchWeights, int width, int height, int featureCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < featureCount * width * height) {
		int i = id % (width * height);
		int m = id / (width * height);

		float sDeviation = 0.0; // standart sapma i�in

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]);

		feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation;
		feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];

		if (fabs(feature[(m * width * height) + i]) + feature[(m * width * height) + i] < 0.01) {
			feature[(m * width * height) + i] = 0.0;
		}
	}
}

__global__ void maxPoolingGPU(float* feature, int& width, int& height, int  featureCount, int pool, int stride)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;


	if () {
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
				if (isgreater(temp, max)) {
					max = temp;
				}
			}
		}
		feature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max;

	}

}

void conv1ExecGPU(CpuGpuMem* cg, const int maskCount)
{
	int iws = cg->imageWidthSize;
	int ihs = cg->imageHeightSize;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize = iws - ms + 1;
	int fhs = cg->featureHeightSize = ihs - ms + 1;

	int blockDim = 1024;
	int threadCount = maskCount * fws * fhs;


	int gridDim = (threadCount + blockDim - 1) / blockDim;

	
	conv1GPU << <gridDim, blockDim, 0 ,cg->stream>> > (cg->gpuFeaturePtr, cg->gpuMaskPtr, (int*)cg->gpuImagePtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize,cg->featureWidthSize, cg->featureHeightSize, maskCount);
	batchConv1GPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, maskCount);

	threadCount = maskCount * (fws/2) * (fhs/2);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (float* feature, int& width, int& height, int  featureCount, int pool, int stride)

	width = width / stride;
	height = height / stride;

}


