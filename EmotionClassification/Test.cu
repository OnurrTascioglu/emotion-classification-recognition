#include <windows.h>
#include <cmath>

#include "device_launch_parameters.h"

#include "CpuGpuMem.h"
#include "KernelGpuAdd.cuh"

#define BIAS 1

__global__ void gpu_add(int* gpu_numbers, const int nc)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < nc)
		gpu_numbers[id] *= 2;
}

__global__ void conv1GPU(float* resultImages, float* masks, BYTE* image, int width, int height, int maskSize, int rMatrixWidth, int rMatrixHeight, int maskCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int temp = 0;
	int k = id % (maskSize * maskSize);
	temp = id / (maskSize * maskSize);
	int j = temp % rMatrixWidth;
	temp = temp / rMatrixWidth;
	int i = temp % rMatrixHeight;
	int m = temp / rMatrixHeight;
	int mCol = k % maskSize;
	int mRow = k / maskSize;

	resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] +=
		(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];

	if(id%(maskSize*maskSize) == 0)
		resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] += (float)BIAS * masks[maskCount * (maskSize * maskSize) + m];

}

__global__ void conv1GPUSetting(float* resultImages, float* masks, BYTE* image, float* weights,int width, int height, int maskSize, int rMatrixWidth, int rMatrixHeight, int maskCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;



	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) {
		resultImages[i] = 0.0;
	}

	for (int i = 0; i < maskSize * maskSize; i++) {
		for (int j = 0; j < maskCount; j++) {
			masks[j * maskSize * maskSize + i] = weights[i * maskCount + j];
		}
	}

	for (int i = 0; i < maskCount; i++) {
		masks[maskCount * maskSize * maskSize + i] = weights[maskCount * maskSize * maskSize + i];
	}

}

void conv1ExecGPU(CpuGpuMem* cg, BYTE* inputImages,float* weights,const int maskCount, const int imageCount)
{
	int iws = cg->imageWidthSize;
	int ihs = cg->imageHeightSize;
	int ms = cg->maskWHSize;
	int fws = cg->featureWidthSize = iws - ms + 1;
	int fhs = cg->featureHeightSize = ihs - ms + 1;

	int blockDim = 1024;
	int threadCount = maskCount * fws * fhs * ms * ms;




	for (int i = 0; i < maskCount; i++) {
		masks[maskCount * ms * ms + i] = weights[maskCount * ms * ms + i];
	}




	int gridDim = (threadCount + blockDim - 1) / blockDim;

	conv1GPU << <gridDim, blockDim, 0, cg->stream >> > ((int*)cg->gpuPtr, number_count);


	delete[] masks;
	delete[] image;
}





void cpuGpuExecute(CpuGpuMem* cg)
{
	int number_count = cg->allocSize;

	int blockDim = 1024;
	int gridDim = (number_count + blockDim - 1) / blockDim;

	execute
		for (size_t i = 0; i < 4; i++)
			gpu_add << <gridDim, blockDim, 0, cg->stream >> > ((int*)cg->gpuPtr, number_count);
}

//------------------------------------------------------------------
float* conv1(BYTE* inputImage, float* weights, int& width, int& height, int maskSize, int maskCount, int imageCount) {


	int rMatrixWidth = width - maskSize + 1; //extern
	int rMatrixHeight = height - maskSize + 1; //extern


	float* masks = new float[maskSize * maskSize * maskCount + maskCount]; //cpugpuAlloc
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight]; //cpugpuAlloc
	BYTE* image = new BYTE[width * height]; //cpugpuAlloc



	for (int i = 0; i < width * height; i++) {
		image[i] = inputImage[(imageCount * width * height) + i]; //
	}

	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) {
		resultImages[i] = 0.0;
	}

	for (int i = 0; i < maskSize * maskSize; i++) {
		for (int j = 0; j < maskCount; j++) {
			masks[j * maskSize * maskSize + i] = weights[i * maskCount + j];
		}
	}
	for (int i = 0; i < maskCount; i++) {
		masks[maskCount * maskSize * maskSize + i] = weights[maskCount * maskSize * maskSize + i]
	}

	for (int m = 0; m < maskCount; m++) {
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int k = 0; k < maskSize * maskSize; k++) {
					int mCol = k % maskSize;
					int mRow = k / maskSize;
					resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] +=
						(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];
				}
				resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] += (float)BIAS * masks[maskCount * (maskSize * maskSize) + m];
			}
		}
	}

	delete[] masks;
	delete[] image;

	width = width - maskSize + 1;
	height = height - maskSize + 1;

	return resultImages;
}