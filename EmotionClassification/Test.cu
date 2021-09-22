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
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tanýmlanýr

	//Grafik kartýnda oluþturulan threadler warplara (32’lik thread bölümleri) baðýmlý
	//oluþturulduðu için 32’nin katý olmayan durumlarda fazladan thread oluþturulur.

	if (id < maskCount * rMatrixWidth * rMatrixHeight) { //Çýkýþ nöronlarý hesaplandýktan sonra Fazlalýk threadlerin koþmamasý için
		int temp = 0;
		int j = id % rMatrixWidth;		//conv1() fonksiyonunda evriþim iþlemini yapan ana for döngüsündeki  j,i,m deðerlerinin bulunmasý için
		temp = id / rMatrixWidth;		//buradaki mod iþlemleri yapýlýr. i, j, m deðerleri threadin kendi id'sindeki for iterasyonunu bulmasýný saðlar.
		int i = temp % rMatrixHeight;
		int m = temp / rMatrixHeight;
		float tempSum = 0.0;

		for (int k = 0; k < maskSize * maskSize; k++) { // evriþim iþlemi burada yapýlýr conv1() fonksiyonu ile ayný iþlemi yapar
			int mCol = k % maskSize;    //maske içinde gezebilmek için mCol ve mRow deðerleri hesaplanýr.
			int mRow = k / maskSize;
			tempSum +=
				(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];  //maske gezdirme iþlemi
		}
		resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] =  tempSum + (float)BIAS * masks[maskCount * (maskSize * maskSize) + m]; //Maske iþleminden sonra bias deðeri ile çarpýlýp toplanýr.
	}
}


__global__ void convHiddenGPU(float* feature, float* resultImages, float* weights, int fWidth, int fHeight, int maskSize, int maskCount, int maskDim)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tanýmlanýr

	int rMatrixWidth = fWidth - maskSize + 1;  //padding olmadan çýkýþ katmanýnýn yükseklik ve geniþliði hesaplanýr
	int rMatrixHeight = fHeight - maskSize + 1;

	//Grafik kartýnda oluþturulan threadler warplara (32’lik thread bölümleri) baðýmlý
	//oluþturulduðu için 32’nin katý olmayan durumlarda fazladan thread oluþturulur.

	if (id < maskCount * rMatrixWidth * rMatrixHeight) {  //Çýkýþ nöronlarý hesaplandýktan sonra Fazlalýk threadlerin koþmamasý için
		int temp = 0;
		int j = id % rMatrixWidth;		//convHidden() fonksiyonunda evriþim iþlemini yapan ana for döngüsündeki  j,i,c deðerlerinin bulunmasý için	
		temp = id / rMatrixWidth;		//buradaki mod iþlemleri yapýlýr. i, j, c deðerleri threadin kendi id'sindeki for iterasyonunu bulmasýný saðlar.
		int i = temp % rMatrixHeight;
		int c = temp / rMatrixHeight;
		float tempSum = 0.0;

		// evriþim iþlemi burada yapýlýr conv1() fonksiyonu ile ayný iþlemi yapar
		for (int d = 0; d < maskDim; d++) {  // maskenin derinlinin hesaplanmasý için
			for (int k = 0; k < maskSize * maskSize; k++) {  
				int mCol = k % maskSize;    // maske içinde gezebilmek için mCol ve mRow deðerleri hesaplanýr.
				int mRow = k / maskSize;
				tempSum += (float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * weights[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k]; //maske gezdirme iþlemi
			}
		}
		resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] = tempSum + BIAS * weights[maskCount * maskDim * maskSize * maskSize + c]; //Maske iþleminden sonra bias deðeri ile çarpýlýp toplanýr.
	}
}


__global__ void batchNormGPU(float* feature, float* batchWeights, int width, int height, int featureCount)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tanýmlanýr

	if (id < featureCount * width * height) {
		int i = id % (width * height); //threadin kendi id'sindeki for iterasyonunu bulmasýný saðlar.
		int m = id / (width * height);

		float sDeviation = 0.0; // standart sapma için

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]); //varyans kullanýlarak standart sapma hesaplanýr. (featureCount * 3) dizide varyans elemanlarýna eriþir

		feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation; //Her bir deðer aritmetik ortalamadan çýkarýlýp standart sapmaya bölünür. (featureCount * 2) aritmetik ortalama deðerlerine eriþir
		feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];  //Sonuç gamma ile çarpýlýr beta ile toplanýr.

		if (fabs(feature[(m * width * height) + i]) + feature[(m * width * height) + i] < 0.001) { 
			feature[(m * width * height) + i] = 0.0; //ReLU iþlemi
		}
	}
}

__global__ void maxPoolingGPU(float* feature, float* tempFeature, int width, int height, int  featureCount, int pool, int stride)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;  //bloklardaki threadlar belirlenip idleri tanýmlanýr


	if (id < featureCount * (width / stride) * (height / stride)) {
		int temp2 = 0;
		int col = id % (width / stride);   //threadin kendi id'sindeki for iterasyonunu bulmasýný saðlar.
		temp2 = id / (width / stride);
		int row = temp2 % (height / stride);
		int m = temp2 / (height / stride);

		float max = 0.0;
		float temp = 0.0;

		for (int k = 0; k < pool; k++) {  //pool*pool kadarlýk alandaki deðerlerden en yüksek olan seçilmelidir.
			for (int n = 0; n < pool; n++) {
				temp = feature[(m * width * height) + row * width * stride + col * stride + k * width + n]; //Pool çerçevesinin denk geldiði feature deðerleri temp e atanýr.
				if ((temp - max) > 0.00001) {
					max = temp; //max deðer hesaplanýr.
				}
			}
		}
		tempFeature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max;  //bellek alanýndan tasarruf amacýyla yeni dizi açmak yerine, max deðerler feature dizisine atanýr.

	}

}

__global__ void flattenGPU(float* features, float* flattenArray, int width, int height, int featureCount) {
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tanýmlanýr

	if (id < featureCount * width * height) {

		int temp = 0;
		int f = id % featureCount;  //threadin kendi id'sindeki for iterasyonunu bulmasýný saðlar.
		temp = id / featureCount;
		int j = temp % width;
		int i = temp / width;

		flattenArray[id] = features[f * width * height + i * width + j];  //flatten iþlemi burada yapýlýr. Giriþ nöronlarý aðýrlýklara uygun gelecek þekilde sýralanýr.
	}
}

__global__ void denseGPU(float* inputLayer, float* outputLayer, float* weights, int inputLayerSize, int outputLayerSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x; //bloklardaki threadlar belirlenip idleri tanýmlanýr

	if (id < outputLayerSize) {
		// optimize edilmeli
		float tempSum = 0.0;

		for (int j = 0; j < inputLayerSize; j++) {
			tempSum += inputLayer[j] * weights[j * outputLayerSize + id]; // giriþ nöronlarý ve aðýrlýklar çarpýlýp toplanýr.Çýkýþ katmanýna yazýlýr
		} 
		outputLayer[id] = tempSum + BIAS * weights[inputLayerSize * outputLayerSize + id]; //bias deðeri eklenir

	}
}

__global__ void batchAndReLuDenseGPU(float* input, float* batchWeights, int inputSize) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;  //bloklardaki threadlar belirlenip idleri tanýmlanýr

	if (id < inputSize) {
		float sDeviation = 0.0; 

		sDeviation = sqrt(batchWeights[(inputSize * 3) + id]);   //varyans kullanýlarak standart sapma hesaplanýr. (featureCount * 3) dizide varyans elemanlarýna eriþir
		input[id] = (input[id] - batchWeights[(inputSize * 2) + id]) / sDeviation;  //Her bir deðer aritmetik ortalamadan çýkarýlýp standart sapmaya bölünür. (featureCount * 2) aritmetik ortalama deðerlerine eriþir
		input[id] = input[id] * batchWeights[id] + batchWeights[inputSize + id];  //Sonuç gamma ile çarpýlýr beta ile toplanýr.

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

	int blockDim = 64;//Bir bloktaki thread sayýsý
	int threadCount = cg->denseOutputSize;	//Çýkýþ katmanýndaki toplam nöron sayýsý thread sayýsýný oluþturur. 
											//Bu durumda her nöronun matematiksel hesabýný o thread id sine sahip thread yapar.
	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok sayýsýdýr. Bu normal threadCount / blockDim ile hesaplanýr. Fakat threadCount, blockDim'in tam katlarýndan biri olmadýðý
														   //durumda doðru blok sayýsý oluþturulamaz. Bunun önüne geçmek için (threadCount + blockDim - 1) / blockDim þeklinde hesaplanmalý

	int allocSize = threadCount * sizeof(float);

	result = cudaFree(cg->gpuTempLayer2);// Çýkýþ nöronlarýnýn sonuçlarýnýn yazýlacaðý bir geçici GPU bellek bölgesi tahsis edilir.
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	denseGPU << <gridDim, blockDim >> > (cg->gpuDensePtr, cg->gpuTempLayer2, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize);//2. Tam baðlantýlý katmanýn GPU'da koþmasý.

	free(cg->cpuDensePtr);//Sonuç RAM bellek bölgesine aktarýlmadan önce RAM pointerýnýn iþaret ettiði bellek bölgesi serbest býrakýlýr.
	cg->cpuDensePtr = (float*)malloc(cg->denseOutputAllocSize);
	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuTempLayer2, cg->denseOutputSize * sizeof(float)); //Hesaplamalarýn doðruluðunu kontrol etmek amaçlý sonuçlarýn GPU belleðinden RAM belleðe transferi

}

void dense1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int mc = cg->maskCount; //Maske sayýsý
	int fws = cg->featureWidthSize; //Feature geniþliði
	int fhs = cg->featureHeightSize; //Feature yüksekliði


	int blockDim = 64; //Bir bloktaki thread sayýsý

	int threadCount = cg->maskCount * fws * fhs; //Flatten iþlemi için thread sayýsý hesaplanýr.

	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok sayýsýdýr. Bu normal threadCount / blockDim ile hesaplanýr. Fakat threadCount, blockDim'in tam katlarýndan biri olmadýðý
														   //durumda doðru blok sayýsý oluþturulamaz. Bunun önüne geçmek için (threadCount + blockDim - 1) / blockDim þeklinde hesaplanmalý


	int allocSize = threadCount * sizeof(float); // Çýkýþ nöronlarýnýn sonuçlarýnýn yazýlacaðý bir geçici GPU bellek bölgesi tahsis edilir.
	result = cudaMalloc((float**)&cg->gpuTempLayer2, allocSize);
	assert(result == cudaSuccess);

	flattenGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuTempLayer2, fws, fhs, mc); //Feature uzayýndaki feature'lar flatten iþlemine tabii tutulur.

	threadCount = cg->denseOutputSize; //Çýkýþ katmanýndaki toplam nöron sayýsý thread sayýsýný oluþturur. 
									   //Bu durumda her nöronun matematiksel hesabýný o thread id sine sahip thread yapar.
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//clock_t tStart = clock();
	//double cpuClock = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	denseGPU << <gridDim, blockDim >> > (cg->gpuTempLayer2, cg->gpuDensePtr, cg->gpuDenseWeightPtr, cg->denseInputSize, cg->denseOutputSize); //1. Tam baðlantýlý katmanýn GPU'da koþmasý.

	batchAndReLuDenseGPU << <gridDim, blockDim >> > (cg->gpuDensePtr, cg->gpuBatchPtr, cg->denseOutputSize);// 3. Batch Norm katmanýnýn ve Relu iþleminin GPU'da koþmasý

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuDensePtr, cg->gpuDensePtr, cg->denseOutputSize * sizeof(float)); //Hesaplamalarýn doðruluðunu kontrol etmek amaçlý sonuçlarýn GPU belleðinden RAM belleðe transferi
}

void convHidden1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int ms = cg->maskWHSize;  //maske  boyutu (default=3)
	int fws = cg->featureWidthSize; //feature geniþliði
	int fhs = cg->featureHeightSize; //feature yüksekliði


	int blockDim = 64; //Bir bloktaki thread sayýsý
	int threadCount = cg->maskCount * (fws - ms + 1) * (fhs - ms + 1); //Çýkýþ evriþim katmanýndaki toplam nöron sayýsý thread sayýsýný oluþuturur. 
																	   //Bu durumda her nöronun matematiksel hesabýný o thread id sine sahip thread yapar.

	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok sayýsýdýr. Bu normal threadCount / blockDim ile hesaplanýr. Fakat threadCount, blockDim'in tam katlarýndan biri olmadýðý
														   //durumda doðru blok sayýsý oluþturulamaz. Bunun önüne geçmek için (threadCount + blockDim - 1) / blockDim þeklinde hesaplanmalý

	cg->featureAllocSize = threadCount * sizeof(float); //Gizli evriþim katmanýnýn çýkýþýndaki feature space için boyut güncellemesi yapýlýr. Ardýndan RAM ve GPU bellek bölgeleri tahsis edilir.
	free(cg->cpuFeaturePtr);
	cg->cpuFeaturePtr = (float*)malloc(cg->featureAllocSize);
	result = cudaFree(cg->gpuFeaturePtr);
	assert(result == cudaSuccess);
	result = cudaMalloc((float**)&cg->gpuFeaturePtr, cg->featureAllocSize);
	assert(result == cudaSuccess);
	cudaMemset(cg->gpuFeaturePtr, 0, cg->featureAllocSize);//Gizli evriþim katmanýn sonuçlarýnýn yazýlacaðý, GPU bellek bölgesinin içeriði sýfýra eþitlenir.


	//Gizli evriþim katmanýnýn GPU'da koþmasý
	convHiddenGPU << <gridDim, blockDim >> > (cg->gpuTempLayer, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskWHSize, cg->maskCount, cg->maskDim);

	cg->featureWidthSize = fws - ms + 1;
	cg->featureHeightSize = fhs - ms + 1;
	fws = cg->featureWidthSize;
	fhs = cg->featureHeightSize;

	//2. Batch Norm katmanýnýn GPU'da koþmasý
	batchNormGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount);

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool); //Maxpool iþlemi için thread sayýlarý belinlenmeli
	gridDim = (threadCount + blockDim - 1) / blockDim;

	result = cudaFree(cg->gpuTempLayer);
	assert(result == cudaSuccess);

	//Maxpool iþleminin sonucu için geçici bir bellek tahsisi yapýlýr.
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);

	maxPoolingGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride);

	cg->featureWidthSize /= cg->stride;// maxpool sonunda feature boyutu güncellenir
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float)); // Gizli Evriþim katmanýnýn sonucu GPU bellekten RAM belleðe transfer edilir.

}

void conv1ExecGPU(CpuGpuMem* cg)
{
	cudaError_t result;
	int iws = cg->imageWidthSize; //görüntü geniþliði
	int ihs = cg->imageHeightSize; //görüntü yüksekliði
	int ms = cg->maskWHSize;  //maske  boyutu (default=3)
	int fws = cg->featureWidthSize = iws - ms + 1; //feature geniþiliði
	int fhs = cg->featureHeightSize = ihs - ms + 1;//feature yüksekliði

	int blockDim = 64; //Bir bloktaki thread sayýsý
	int threadCount = cg->maskCount * fws * fhs; //Çýkýþ evriþim katmanýndaki toplam nöron sayýsý thread sayýsýný oluþuturur. 
												 //Bu durumda her nöronun matematiksel hesabýný o thread id sine sahip thread yapar.

	int gridDim = (threadCount + blockDim - 1) / blockDim; //gridDim toplam blok sayýsýdýr. Bu normal threadCount / blockDim ile hesaplanýr. Fakat threadCount, blockDim'in tam katlarýndan biri olmadýðý
														   //durumda doðru blok sayýsý oluþturulamaz. Bunun önüne geçmek için (threadCount + blockDim - 1) / blockDim þeklinde hesaplanmalý

	conv1GPU << <gridDim, blockDim >> > ((int*)cg->gpuImagePtr, cg->gpuFeaturePtr, cg->gpuMaskPtr, cg->imageWidthSize, cg->imageHeightSize,
		cg->maskWHSize, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount); //1. evriþim katmaný GPU'da koþmasý

	batchNormGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuBatchPtr, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount); //1. batch norm GPU'da koþmasý

	threadCount = cg->maskCount * (fws / cg->pool) * (fhs / cg->pool); //Maxpool iþlemi için thread sayýlarý belinlenmeli
	gridDim = (threadCount + blockDim - 1) / blockDim;

	//Maxpool iþleminin sonucu için geçici bir bellek tahsisi yapýlýr
	result = cudaMalloc((float**)&cg->gpuTempLayer, threadCount * sizeof(float));
	assert(result == cudaSuccess);


	maxPoolingGPU << <gridDim, blockDim >> > (cg->gpuFeaturePtr, cg->gpuTempLayer, cg->featureWidthSize, cg->featureHeightSize, cg->maskCount, cg->pool, cg->stride); //maxpool GPU'da koþmasý

	cg->featureWidthSize /= cg->stride; // maxpool sonunda feature boyutu güncellenir
	cg->featureHeightSize /= cg->stride;

	cpuGpuMemCopy(cudaMemcpyDeviceToHost, cg, cg->cpuFeaturePtr, cg->gpuTempLayer, threadCount * sizeof(float)); // 1. katmanýn sonucu GPU bellekten RAM belleðe transfer edilir.

}

