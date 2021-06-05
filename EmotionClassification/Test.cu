#include <windows.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "device_launch_parameters.h"
#include <time.h>
#include "CpuGpu.h"
#include "CpuGpuMem.h"
#include "KernelGpuAdd.cuh"
#include "cuda_runtime.h"

#define BIAS 1

//----------------conv1
__global__ void conv1GPU(int* image, float* resultImages, float* masks, int width, int height, int maskSize, int rMatrixWidth, int rMatrixHeight, int maskCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < maskCount * rMatrixWidth * rMatrixHeight) {
		int temp = 0;
		int j = id % rMatrixWidth;
		temp = id / rMatrixWidth;
		int i = temp % rMatrixHeight;
		int m = temp / rMatrixHeight;
		float tempSum = 0.0;

		for (int k = 0; k < maskSize * maskSize; k++) {
			int mCol = k % maskSize;
			int mRow = k / maskSize;
			tempSum +=
				(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];
		}
		resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] =  tempSum + (float)BIAS * masks[maskCount * (maskSize * maskSize) + m];
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

		if (fabs(feature[(m * width * height) + i]) + feature[(m * width * height) + i] < 0.001) {
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
				if ((temp - max) > 0.00001) {
					max = temp;
				}
			}
		}
		tempFeature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max;

	}

}

//-----------------------conv2


__global__ void convHiddenGPU(float* feature, float* resultImages, float* weights, int fWidth, int fHeight, int maskSize, int maskCount, int maskDim)
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
		float tempSum = 0.0;

		for (int d = 0; d < maskDim; d++) {
			for (int k = 0; k < maskSize * maskSize; k++) {
				int mCol = k % maskSize;
				int mRow = k / maskSize;
				tempSum +=	(float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * weights[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k];

			}
		}
		resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + BIAS * weights[maskCount * maskDim * maskSize * maskSize + c];
	}
}

__global__ void flattenGPU(float* features, float* flattenArray, int width, int height, int featureCount) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < featureCount * width * height) {

		int temp = 0;
		int f = id % featureCount;
		temp = id / featureCount;
		int j = temp % width;
		int i = temp / width;

		flattenArray[id] = features[f * width * height + i * width + j];
	}
}

__global__ void denseGPU(float* inputLayer, float* outputLayer, float* weights, int inputLayerSize, int outputLayerSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < outputLayerSize) {
		// optimize edilmeli
		float tempSum = 0.0;

		for (int j = 0; j < inputLayerSize; j++) {
			tempSum += inputLayer[j] * weights[j * outputLayerSize + id];
		}
		outputLayer[id] = tempSum + BIAS * weights[inputLayerSize * outputLayerSize + id];

	}
}

__global__ void batchAndReLuDenseGPU(float* input, float* batchWeights, int inputSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < inputSize) {
		float sDeviation = 0.0; 

		sDeviation = sqrt(batchWeights[(inputSize * 3) + id]);
		input[id] = (input[id] - batchWeights[(inputSize * 2) + id]) / sDeviation;
		input[id] = input[id] * batchWeights[id] + batchWeights[inputSize + id];

		if (fabs(input[id]) + input[id] < 0.00001) {
			input[id] = 0.0;
		}
	}
}


void model2Dense3ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;

	int blockDim = 64;
	int threadCount = cg->denseOutputSize;
	int gridDim = (threadCount + blockDim - 1) / blockDim;

	int allocSize = threadCount * sizeof(float);

	result = cudaFree(cg->gpuDensePtr);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuDensePtr, allocSize);
	assert(result == cudaSuccess);

	cudaMemset(cg->gpuTempLayer2, 0, allocSize);
	denseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer2,cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);

	free(cg->cpuDensePtr);
	cg->cpuDensePtr = (float*)malloc(cg->denseOutputAllocSize);
	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuDensePtr, cg->denseOutputSize * sizeof(float));

}

void model2Dense2ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;

	int blockDim = 64;
	int threadCount = cg->denseOutputSize;
	int gridDim = (threadCount + blockDim - 1) / blockDim;

	int allocSize = threadCount * sizeof(float);

	result = cudaFree(cg->gpuTempLayer2);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	denseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDensePtr, cg->gpuTempLayer2, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);
	
	batchAndReLuDenseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer2, cg->gpuBatchPtr, cg->denseOutputSize);

	//free(cg->cpuDensePtr);
	//cg->cpuDensePtr = (float*)malloc(cg->denseOutputAllocSize);

	//cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuTempLayer2, cg->denseOutputSize * sizeof(float));

}

void model2Dense1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int mc = cg->maskCount;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;


	int blockDim = 64;
	int threadCount = cg->maskCount * fws * fhs;
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	int allocSize = threadCount * sizeof(float);

	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	flattenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuTempLayer2, fws, fhs, mc);

	threadCount = cg->denseOutputSize;
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//clock_t tStart = clock();
	//double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	denseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer2, cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);

	batchAndReLuDenseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDensePtr, cg->gpuBatchPtr, cg->denseOutputSize);

}

void model2Conv4ExecGpu(CpuGpuMem* cg)
{
	cudaError_t result;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;


	int blockDim = 64;
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1);
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	int tempAllocSize = threadCount * sizeof(float);
	result = cudaFree(cg->gpuTempLayer);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuTempLayer, tempAllocSize);
	assert(result == cudaSuccess);


	convHiddenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, tempAllocSize);

}

void model2Conv3ExecGpu(CpuGpuMem* cg)
{
	cudaError_t result;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;


	int blockDim = 64;
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1);
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	int tempAllocSize = threadCount * sizeof(float);
	result = cudaFree(cg->gpuTempLayer);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuTempLayer, tempAllocSize);
	assert(result == cudaSuccess);


	convHiddenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	cg->featureAllocSize = threadCount * sizeof(float);
	free(cg->cpuFeaturePtr);
	cg->cpuFeaturePtr = (float*)malloc(cg->featureAllocSize);
	result = cudaFree(cg->gpuFeaturePtr);
	assert(result == cudaSuccess);

	result = cudaMalloc((float**)&cg->gpuFeaturePtr, cg->featureAllocSize);
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuFeaturePtr, cg->featureAllocSize);

}

void model2Conv2ExecGpu(CpuGpuMem* cg) {
	cudaError_t result;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;


	int blockDim = 64;
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1);
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	cg->featureAllocSize = threadCount * sizeof(float);
	free(cg->cpuFeaturePtr);
	cg->cpuFeaturePtr = (float*)malloc(cg->featureAllocSize);
	result = cudaFree(cg->gpuFeaturePtr);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuFeaturePtr, cg->featureAllocSize);
	assert(result == cudaSuccess);


	convHiddenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuFeaturePtr, threadCount * sizeof(float));

}

void model2Conv1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int iws = cg->imageWidthSize;
	int ihs = cg->imageHeightSize;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize = iws - ms + 1;
	int fhs = cg->featureHeightSize = ihs - ms + 1;

	int blockDim = 64;
	int threadCount = cg->maskCount * fws * fhs;


	int gridDim = (threadCount + blockDim - 1) / blockDim;


	conv1GPU << <gridDim, blockDim, 0, cg->stream >> > ((int*)cg->gpuImagePtr, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//temp array for pooling result
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);


	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float));

}



void dense2ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;

	int blockDim = 64;
	int threadCount = cg->denseOutputSize;
	int gridDim = (threadCount + blockDim - 1) / blockDim;

	int allocSize = threadCount * sizeof(float);

	result = cudaFree(cg->gpuTempLayer2);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	cudaMemset(cg->gpuTempLayer2, 0, allocSize);
	denseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDensePtr, cg->gpuTempLayer2, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);

	free(cg->cpuDensePtr);
	cg->cpuDensePtr = (float*)malloc(cg->denseOutputAllocSize);
	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuTempLayer2, cg->denseOutputSize * sizeof(float));

}

void dense1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int mc = cg->maskCount;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;


	int blockDim = 64;
	int threadCount = cg->maskCount * fws * fhs;
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	int allocSize = threadCount * sizeof(float);

	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	flattenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuTempLayer2, fws, fhs, mc);

	threadCount = cg->denseOutputSize;
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//clock_t tStart = clock();
	//double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	denseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer2, cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);

	batchAndReLuDenseGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuDensePtr, cg->gpuBatchPtr, cg->denseOutputSize);

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuDensePtr, cg->denseOutputSize * sizeof(float));
}

void convHidden1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize;
	int fhs = cg->featureHeightSize;


	int blockDim = 64;
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1);
	int gridDim = (threadCount + blockDim - 1) / blockDim;


	cg->featureAllocSize = threadCount * sizeof(float);
	free(cg->cpuFeaturePtr);
	cg->cpuFeaturePtr = (float*)malloc(cg->featureAllocSize);
	result = cudaFree(cg->gpuFeaturePtr);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuFeaturePtr, cg->featureAllocSize);
	assert(result == cudaSuccess);
	cudaMemset(cg->gpuFeaturePtr, 0, cg->featureAllocSize);


	convHiddenGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	result = cudaFree(cg->gpuTempLayer);
	assert(result == cudaSuccess);

	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float));

}

void conv1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int iws = cg->imageWidthSize;
	int ihs = cg->imageHeightSize;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize = iws - ms + 1;
	int fhs = cg->featureHeightSize = ihs - ms + 1;

	int blockDim = 64;
	int threadCount = cg->maskCount * fws * fhs;


	int gridDim = (threadCount + blockDim - 1) / blockDim;


	conv1GPU << <gridDim, blockDim, 0, cg->stream >> > ((int*)cg->gpuImagePtr, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	batchNormGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//temp array for pooling result
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);


	maxPoolingGPU << <gridDim, blockDim, 0, cg->stream >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float));

}



