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
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tan�mlan�r

	//Grafik kart�nda olu�turulan threadler warplara (32�lik thread b�l�mleri) ba��ml�
	//olu�turuldu�u i�in 32�nin kat� olmayan durumlarda fazladan thread olu�turulur.

	if (id < maskCount * rMatrixWidth * rMatrixHeight) { //��k�� n�ronlar� hesapland�ktan sonra Fazlal�k threadlerin ko�mamas� i�in
		int temp = 0;
		int j = id % rMatrixWidth;		//conv1() fonksiyonunda evri�im i�lemini yapan ana for d�ng�s�ndeki  j,i,m de�erlerinin bulunmas� i�in
		temp = id / rMatrixWidth;		//buradaki mod i�lemleri yap�l�r. i, j, m de�erleri threadin kendi id'sindeki for iterasyonunu bulmas�n� sa�lar.
		int i = temp % rMatrixHeight;
		int m = temp / rMatrixHeight;
		float tempSum = 0.0;

		for (int k = 0; k < maskSize * maskSize; k++) { // evri�im i�lemi burada yap�l�r conv1() fonksiyonu ile ayn� i�lemi yapar
			int mCol = k % maskSize;    //maske i�inde gezebilmek i�in mCol ve mRow de�erleri hesaplan�r.
			int mRow = k / maskSize;
			tempSum +=
				(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];  //maske gezdirme i�lemi
		}
		resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] =  tempSum + (float)BIAS * masks[maskCount * (maskSize * maskSize) + m]; //Maske i�leminden sonra bias de�eri ile �arp�l�p toplan�r.
	}
}


__global__ void convHiddenGPU(float* feature, float* resultImages, float* weights, int fWidth, int fHeight, int maskSize, int maskCount, int maskDim)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tan�mlan�r

	int rMatrixWidth = fWidth - maskSize + 1;  //padding olmadan ��k�� katman�n�n y�kseklik ve geni�li�i hesaplan�r
	int rMatrixHeight = fHeight - maskSize + 1;

	//Grafik kart�nda olu�turulan threadler warplara (32�lik thread b�l�mleri) ba��ml�
	//olu�turuldu�u i�in 32�nin kat� olmayan durumlarda fazladan thread olu�turulur.

	if (id < maskCount * rMatrixWidth * rMatrixHeight) {  //��k�� n�ronlar� hesapland�ktan sonra Fazlal�k threadlerin ko�mamas� i�in
		int temp = 0;
		int j = id % rMatrixWidth;		//convHidden() fonksiyonunda evri�im i�lemini yapan ana for d�ng�s�ndeki  j,i,c de�erlerinin bulunmas� i�in	
		temp = id / rMatrixWidth;		//buradaki mod i�lemleri yap�l�r. i, j, c de�erleri threadin kendi id'sindeki for iterasyonunu bulmas�n� sa�lar.
		int i = temp % rMatrixHeight;
		int c = temp / rMatrixHeight;
		float tempSum = 0.0;

		// evri�im i�lemi burada yap�l�r conv1() fonksiyonu ile ayn� i�lemi yapar
		for (int d = 0; d < maskDim; d++) {  // maskenin derinlinin hesaplanmas� i�in
			for (int k = 0; k < maskSize * maskSize; k++) {  
				int mCol = k % maskSize;    // maske i�inde gezebilmek i�in mCol ve mRow de�erleri hesaplan�r.
				int mRow = k / maskSize;
				tempSum += (float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * weights[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k]; //maske gezdirme i�lemi
			}
		}
		resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + BIAS * weights[maskCount * maskDim * maskSize * maskSize + c]; //Maske i�leminden sonra bias de�eri ile �arp�l�p toplan�r.
	}
}


__global__ void batchNormGPU(float* feature, float* batchWeights, int width, int height, int featureCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tan�mlan�r

	if (id < featureCount * width * height) {
		int i = id % (width * height); //threadin kendi id'sindeki for iterasyonunu bulmas�n� sa�lar.
		int m = id / (width * height);

		float sDeviation = 0.0; // standart sapma i�in

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]); //varyans kullan�larak standart sapma hesaplan�r. (featureCount * 3) dizide varyans elemanlar�na eri�ir

		feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation; //Her bir de�er aritmetik ortalamadan ��kar�l�p standart sapmaya b�l�n�r. (featureCount * 2) aritmetik ortalama de�erlerine eri�ir
		feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];  //Sonu� gamma ile �arp�l�r beta ile toplan�r.

		if (fabs(feature[(m * width * height) + i]) + feature[(m * width * height) + i] < 0.001) { 
			feature[(m * width * height) + i] = 0.0; //ReLU i�lemi
		}
	}
}

__global__ void maxPoolingGPU(float* feature, float* tempFeature, int width, int height, int  featureCount, int pool, int stride)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;  //bloklardaki threadlar belirlenip idleri tan�mlan�r


	if (id < featureCount * (width / stride) * (height / stride)) {
		int temp2 = 0;
		int col = id % (width / stride);   //threadin kendi id'sindeki for iterasyonunu bulmas�n� sa�lar.
		temp2 = id / (width / stride);
		int row = temp2 % (height / stride);
		int m = temp2 / (height / stride);

		float max = 0.0;
		float temp = 0.0;

		for (int k = 0; k < pool; k++) {  //pool*pool kadarl�k alandaki de�erlerden en y�ksek olan se�ilmelidir.
			for (int n = 0; n < pool; n++) {
				temp = feature[(m * width * height) + row * width * stride + col * stride + k * width + n]; //Pool �er�evesinin denk geldi�i feature de�erleri temp e atan�r.
				if ((temp - max) > 0.00001) {
					max = temp; //max de�er hesaplan�r.
				}
			}
		}
		tempFeature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max;  //bellek alan�ndan tasarruf amac�yla yeni dizi a�mak yerine, max de�erler feature dizisine atan�r.

	}

}

__global__ void flattenGPU(float* features, float* flattenArray, int width, int height, int featureCount) {
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tan�mlan�r

	if (id < featureCount * width * height) {

		int temp = 0;
		int f = id % featureCount;  //threadin kendi id'sindeki for iterasyonunu bulmas�n� sa�lar.
		temp = id / featureCount;
		int j = temp % width;
		int i = temp / width;

		flattenArray[id] = features[f * width * height + i * width + j];  //flatten i�lemi burada yap�l�r. Giri� n�ronlar� a��rl�klara uygun gelecek �ekilde s�ralan�r.
	}
}

__global__ void denseGPU(float* inputLayer, float* outputLayer, float* weights, int inputLayerSize, int outputLayerSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tan�mlan�r

	if (id < outputLayerSize) {
		// optimize edilmeli
		float tempSum = 0.0;

		for (int j = 0; j < inputLayerSize; j++) {
			tempSum += inputLayer[j] * weights[j * outputLayerSize + id]; // giri� n�ronlar� ve a��rl�klar �arp�l�p toplan�r.��k�� katman�na yaz�l�r
		} 
		outputLayer[id] = tempSum + BIAS * weights[inputLayerSize * outputLayerSize + id]; //bias de�eri eklenir

	}
}

__global__ void batchAndReLuDenseGPU(float* input, float* batchWeights, int inputSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;  //bloklardaki threadlar belirlenip idleri tan�mlan�r

	if (id < inputSize) {
		float sDeviation = 0.0; 

		sDeviation = sqrt(batchWeights[(inputSize * 3) + id]);   //varyans kullan�larak standart sapma hesaplan�r. (featureCount * 3) dizide varyans elemanlar�na eri�ir
		input[id] = (input[id] - batchWeights[(inputSize * 2) + id]) / sDeviation;  //Her bir de�er aritmetik ortalamadan ��kar�l�p standart sapmaya b�l�n�r. (featureCount * 2) aritmetik ortalama de�erlerine eri�ir
		input[id] = input[id] * batchWeights[id] + batchWeights[inputSize + id];  //Sonu� gamma ile �arp�l�r beta ile toplan�r.

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

	denseGPU << <gridDim, blockDim>> > (cg->gpuTempLayer2,cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);

	result = cudaFree(cg->gpuTempLayer2);
	assert(result == cudaSuccess);

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

	denseGPU << <gridDim, blockDim>> > (cg->gpuDensePtr, cg->gpuTempLayer2, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);
	
	batchAndReLuDenseGPU << <gridDim, blockDim>> > (cg->gpuTempLayer2, cg->gpuBatchPtr, cg->denseOutputSize);

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

	flattenGPU << <gridDim, blockDim>> > (cg->gpuTempLayer, cg->gpuTempLayer2, fws, fhs, mc);

	result = cudaFree(cg->gpuTempLayer);
	assert(result == cudaSuccess);

	threadCount = cg->denseOutputSize;
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//clock_t tStart = clock();
	//double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	denseGPU << <gridDim, blockDim>> > (cg->gpuTempLayer2, cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);

	batchAndReLuDenseGPU << <gridDim, blockDim>> > (cg->gpuDensePtr, cg->gpuBatchPtr, cg->denseOutputSize);
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


	convHiddenGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

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


	convHiddenGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;
	fws = cg->featureWidthSize;
	fhs = cg->featureHeightSize;

	batchNormGPU << <gridDim, blockDim>> > (cg->gpuTempLayer, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	cg->featureAllocSize = threadCount * sizeof(float);
	free(cg->cpuFeaturePtr);
	cg->cpuFeaturePtr = (float*)malloc(cg->featureAllocSize);
	
	result = cudaFree(cg->gpuFeaturePtr);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuFeaturePtr, cg->featureAllocSize);
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

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


	convHiddenGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;

	batchNormGPU << <gridDim, blockDim>> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

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


	conv1GPU << <gridDim, blockDim >> > ((int*)cg->gpuImagePtr, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	batchNormGPU << <gridDim, blockDim>> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool);
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//temp array for pooling result
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);


	maxPoolingGPU << <gridDim, blockDim>> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float));

}




void dense2ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;

	int blockDim = 64;//Bir bloktaki thread say�s�
	int threadCount = cg->denseOutputSize;	//��k�� katman�ndaki toplam n�ron say�s� thread say�s�n� olu�turur. 
											//Bu durumda her n�ronun matematiksel hesab�n� o thread id sine sahip thread yapar.
	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok say�s�d�r. Bu normal threadCount / blockDim ile hesaplan�r. Fakat threadCount, blockDim'in tam katlar�ndan biri olmad���
														   //durumda do�ru blok say�s� olu�turulamaz. Bunun �n�ne ge�mek i�in (threadCount + blockDim - 1) / blockDim �eklinde hesaplanmal�

	int allocSize = threadCount * sizeof(float);

	result = cudaFree(cg->gpuTempLayer2);// ��k�� n�ronlar�n�n sonu�lar�n�n yaz�laca�� bir ge�ici GPU bellek b�lgesi tahsis edilir.
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	denseGPU << <gridDim, blockDim >> > (cg->gpuDensePtr, cg->gpuTempLayer2, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);//2. Tam ba�lant�l� katman�n GPU'da ko�mas�.

	free(cg->cpuDensePtr);//Sonu� RAM bellek b�lgesine aktar�lmadan �nce RAM pointer�n�n i�aret etti�i bellek b�lgesi serbest b�rak�l�r.
	cg->cpuDensePtr = (float*)malloc(cg->denseOutputAllocSize);
	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuTempLayer2, cg->denseOutputSize * sizeof(float)); //Hesaplamalar�n do�rulu�unu kontrol etmek ama�l� sonu�lar�n GPU belle�inden RAM belle�e transferi

}

void dense1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int mc = cg->maskCount; //Maske say�s�
	int fws = cg->featureWidthSize; //Feature geni�li�i
	int fhs = cg->featureHeightSize; //Feature y�ksekli�i


	int blockDim = 64; //Bir bloktaki thread say�s�

	int threadCount = cg->maskCount * fws * fhs; //Flatten i�lemi i�in thread say�s� hesaplan�r.

	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok say�s�d�r. Bu normal threadCount / blockDim ile hesaplan�r. Fakat threadCount, blockDim'in tam katlar�ndan biri olmad���
														   //durumda do�ru blok say�s� olu�turulamaz. Bunun �n�ne ge�mek i�in (threadCount + blockDim - 1) / blockDim �eklinde hesaplanmal�


	int allocSize = threadCount * sizeof(float); // ��k�� n�ronlar�n�n sonu�lar�n�n yaz�laca�� bir ge�ici GPU bellek b�lgesi tahsis edilir.
	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	flattenGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuTempLayer2, fws, fhs, mc); //Feature uzay�ndaki feature'lar flatten i�lemine tabii tutulur.

	threadCount = cg->denseOutputSize; //��k�� katman�ndaki toplam n�ron say�s� thread say�s�n� olu�turur. 
									   //Bu durumda her n�ronun matematiksel hesab�n� o thread id sine sahip thread yapar.
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//clock_t tStart = clock();
	//double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	denseGPU << <gridDim, blockDim >> > (cg->gpuTempLayer2, cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize); //1. Tam ba�lant�l� katman�n GPU'da ko�mas�.

	batchAndReLuDenseGPU << <gridDim, blockDim >> > (cg->gpuDensePtr, cg->gpuBatchPtr, cg->denseOutputSize);// 3. Batch Norm katman�n�n ve Relu i�leminin GPU'da ko�mas�

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuDensePtr, cg->denseOutputSize * sizeof(float)); //Hesaplamalar�n do�rulu�unu kontrol etmek ama�l� sonu�lar�n GPU belle�inden RAM belle�e transferi
}

void convHidden1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int ms = cg->maskWHSize;  //maske  boyutu (default=3)
	int fws = cg->featureWidthSize; //feature geni�li�i
	int fhs = cg->featureHeightSize; //feature y�ksekli�i


	int blockDim = 64; //Bir bloktaki thread say�s�
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1); //��k�� evri�im katman�ndaki toplam n�ron say�s� thread say�s�n� olu�uturur. 
																	   //Bu durumda her n�ronun matematiksel hesab�n� o thread id sine sahip thread yapar.

	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok say�s�d�r. Bu normal threadCount / blockDim ile hesaplan�r. Fakat threadCount, blockDim'in tam katlar�ndan biri olmad���
														   //durumda do�ru blok say�s� olu�turulamaz. Bunun �n�ne ge�mek i�in (threadCount + blockDim - 1) / blockDim �eklinde hesaplanmal�

	cg->featureAllocSize = threadCount * sizeof(float); //Gizli evri�im katman�n�n ��k���ndaki feature space i�in boyut g�ncellemesi yap�l�r. Ard�ndan RAM ve GPU bellek b�lgeleri tahsis edilir.
	free(cg->cpuFeaturePtr);
	cg->cpuFeaturePtr = (float*)malloc(cg->featureAllocSize);
	result = cudaFree(cg->gpuFeaturePtr);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuFeaturePtr, cg->featureAllocSize);
	assert(result == cudaSuccess);
	cudaMemset(cg->gpuFeaturePtr, 0, cg->featureAllocSize);//Gizli evri�im katman�n sonu�lar�n�n yaz�laca��, GPU bellek b�lgesinin i�eri�i s�f�ra e�itlenir.


	//Gizli evri�im katman�n�n GPU'da ko�mas�
	convHiddenGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;
	fws = cg->featureWidthSize;
	fhs = cg->featureHeightSize;

	//2. Batch Norm katman�n�n GPU'da ko�mas�
	batchNormGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool); //Maxpool i�lemi i�in thread say�lar� belinlenmeli
	gridDim = (threadCount + blockDim - 1) / blockDim;

	result = cudaFree(cg->gpuTempLayer);
	assert(result == cudaSuccess);

	//Maxpool i�leminin sonucu i�in ge�ici bir bellek tahsisi yap�l�r.
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;// maxpool sonunda feature boyutu g�ncellenir
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float)); // Gizli Evri�im katman�n�n sonucu GPU bellekten RAM belle�e transfer edilir.

}

void conv1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int iws = cg->imageWidthSize; //g�r�nt� geni�li�i
	int ihs = cg->imageHeightSize; //g�r�nt� y�ksekli�i
	int ms = cg->maskWHSize;  //maske  boyutu (default=3)
	int fws = cg->featureWidthSize = iws - ms + 1; //feature geni�ili�i
	int fhs = cg->featureHeightSize = ihs - ms + 1;//feature y�ksekli�i

	int blockDim = 64; //Bir bloktaki thread say�s�
	int threadCount = cg->maskCount * fws * fhs; //��k�� evri�im katman�ndaki toplam n�ron say�s� thread say�s�n� olu�uturur. 
												 //Bu durumda her n�ronun matematiksel hesab�n� o thread id sine sahip thread yapar.

	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok say�s�d�r. Bu normal threadCount / blockDim ile hesaplan�r. Fakat threadCount, blockDim'in tam katlar�ndan biri olmad���
														   //durumda do�ru blok say�s� olu�turulamaz. Bunun �n�ne ge�mek i�in (threadCount + blockDim - 1) / blockDim �eklinde hesaplanmal�

	conv1GPU << <gridDim, blockDim >> > ((int*)cg->gpuImagePtr, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount); //1. evri�im katman� GPU'da ko�mas�

	batchNormGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount); //1. batch norm GPU'da ko�mas�

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool); //Maxpool i�lemi i�in thread say�lar� belinlenmeli
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//Maxpool i�leminin sonucu i�in ge�ici bir bellek tahsisi yap�l�r
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);


	maxPoolingGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride); //maxpool GPU'da ko�mas�

	cg->featureWidthSize /= cg->stride; // maxpool sonunda feature boyutu g�ncellenir
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float)); // 1. katman�n sonucu GPU bellekten RAM belle�e transfer edilir.

}

