#include <windows.h>
#include <cmath>


float* conv1(BYTE* inputImage, float* weights, int width, int height, int maskSize, int maskCount , int imageCount, int& sizeW, int& sizeH) {


	
	
	int rMatrixWidth = width - maskSize + 1;
	int rMatrixHeight = height - maskSize + 1;


	float* masks = new float[maskSize * maskSize * maskCount];
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight];
	BYTE* image = new BYTE[width * height];
	//BYTE* bResultImages = new BYTE[maskCount * rMatrixWidth * rMatrixHeight];
	
	for (int i = 0; i < width * height; i++) {
		image[i] = inputImage[(imageCount * width * height) + i];
	}

	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) {
		resultImages[i] = 0;
	}

	for (int i = 0; i < maskSize * maskSize; i++) {
		for (int j = 0; j < maskCount; j++) {
			masks[j * maskSize * maskSize + i] = weights[i * maskCount + j];
		}
	}


	for (int m = 0; m < maskCount; m++) {
		for (int i = 0; i < rMatrixWidth; i++) {
			for (int j = 0; j < rMatrixHeight; j++) {
				for (int k = 0; k < maskSize * maskSize; k++) {
					int mCol = k % maskSize;
					int mRow = k / maskSize;
					resultImages[(m * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] +=
						(float)image[(width * i + j) + mRow * width + mCol] * masks[m * (maskSize * maskSize) + k];
				}
			}
		}
	}

	//int max = 0;
	//int min = 0;
	//float ratio = 0.0;

	//for (int m = 0; m < maskCount; m++) {
	//	for (int i = 0; i < rMatrixWidth * rMatrixHeight; i++) {
	//		if ((int)resultImages[(m * rMatrixWidth * rMatrixHeight) + i] > max) {
	//			max = resultImages[(m * rMatrixWidth * rMatrixHeight) + i];
	//		}
	//		if ((int)resultImages[(m * rMatrixWidth * rMatrixHeight) + i] < min) {
	//			min = resultImages[(m * rMatrixWidth * rMatrixHeight) + i];
	//		}
	//	}
	//	for (int i = 0; i < rMatrixWidth * rMatrixHeight; i++) {
	//		resultImages[(m * rMatrixWidth * rMatrixHeight) + i] = resultImages[(m * rMatrixWidth * rMatrixHeight) + i] - (min);
	//	}
	//	ratio = (float)(max - min) / 254;

	//	for (int i = 0; i < rMatrixWidth * rMatrixHeight; i++) {
	//		bResultImages[(m * rMatrixWidth * rMatrixHeight) + i] = resultImages[(m * rMatrixWidth * rMatrixHeight) + i] / ratio;
	//	}
	//}


	delete[] masks;
	delete[] image;

	sizeW = width - maskSize + 1;
	sizeH = height - maskSize + 1;

	return resultImages;
}

float* batchNormalization(float* feature, int width, int height , int maskCount) { //Batch normalize yöntemi

	float sum = 0.0;// aritmetik ortalama için
	float sDeviation = 0.0; // standart sapma için

	int a;
	for (int m = 0; m < maskCount; m++) {
		for (int i = 0; i < width * height; i++)
		{
			sum += feature[(m*width*height) + i];
		}
		sum = sum / (float)(width * height); // aritmetik ortalama alýnýr

		for (int i = 0; i < width * height; i++)
		{
			sDeviation += pow((feature[(m * width * height) + i] - sum), 2);
		}
		sDeviation = sqrt(sDeviation / (float)((width * height) - 1));

		for (int i = 0; i < width * height; i++)
		{
			feature[(m * width * height) + i] = (feature[(m * width * height) + i] - sum) / sDeviation;
			if (isgreaterequal(feature[(m * width * height) + i], 2)) {
				a++;
			}
		}
		sum = 0.0;
		sDeviation = 0.0;
	}

	return feature;
}

float* reLU(float* feature,int width,int height , int maskCount) {
	for (int i = 0; i < width * height * maskCount; i++) {
		if (islessequal(feature[i] , 0.0)) {
			feature[i] = 0.0;
		}
	}
	return feature;
}