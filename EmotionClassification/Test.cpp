#include <windows.h>
#include <cmath>


float* conv1(BYTE* inputImage, float* weights, int& width, int& height, int maskSize, int maskCount, int imageCount) {


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
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
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

	width = width - maskSize + 1;
	height = height - maskSize + 1;

	return resultImages;
}

float* convHidden(float* feature, float* weights, int& fWidth, int& fHeight, int maskSize, int maskCount, int maskDim) {

	float* masks = new float[maskSize * maskSize * maskCount * maskDim];
	int rMatrixWidth = fWidth - maskSize + 1;
	int rMatrixHeight = fHeight - maskSize + 1;
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight];
	for (int i = 0; i < maskCount * rMatrixWidth * rMatrixHeight; i++) {
		resultImages[i] = 0.0;
	}

	//weights resorting
	int count = 0;
	for (int i = 0; i < maskSize * maskSize; i++) {
		for (int j = 0; j < maskDim; j++) {
			for (int k = 0; k < maskCount; k++) {
				masks[k * maskSize * maskSize * maskDim + (j * maskSize * maskSize) + i] = weights[count];
				count++;
			}
		}
	}

	for (int c = 0; c < maskCount; c++) {
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int d = 0; d < maskDim; d++) {
					for (int k = 0; k < maskSize * maskSize; k++) {
						int mCol = k % maskSize;
						int mRow = k / maskSize;
						resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] += 
						(float)feature[d* fWidth* fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * masks[c * (maskDim * maskSize * maskSize) + d* maskSize*maskSize +  k];
					}
				}
			}
		}
	}
	fWidth = fWidth - maskSize + 1;
	fHeight = fHeight - maskSize + 1;

	delete[] masks;
	return resultImages;
}

float* batchNormalization(float* feature, int width, int height, int featureCount) { //Batch normalize yöntemi

	float sum = 0.0;// aritmetik ortalama için
	float sDeviation = 0.0; // standart sapma için

	for (int m = 0; m < featureCount; m++) {
		for (int i = 0; i < width * height; i++)
		{
			sum += feature[(m * width * height) + i];
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
			}
		}
		sum = 0.0;
		sDeviation = 0.0;
	}

	return feature;
}

float* reLU(float* feature, int width, int height, int featureCount) {
	for (int i = 0; i < width * height * featureCount; i++) {
		if (islessequal(feature[i], 0.0)) {
			feature[i] = 0.0;
		}
	}
	return feature;
}

float* maxPooling(float* feature, int& width, int& height, int  featureCount, int pool, int stride) {

	//float* poolingResult = new float[(width / stride) * (height / stride) * featureCount];
	float max = 0.0;
	float temp = 0.0;


	for (int m = 0; m < featureCount; m++) {
		for (int row = 0; row < height / stride; row++) {
			for (int col = 0; col < width / stride; col++) {
				for (int k = 0; k < pool; k++) {
					for (int n = 0; n < pool; n++) {
						temp = feature[(m * width * height) + row * width * stride + col * stride + k * width + n];
						if (isgreater(temp, max)) {
							max = temp;
						}
					}
				}
				feature[(m * (width / stride) * (height / stride)) + (row * (width / stride)) + col] = max;
				max = 0.0;
				temp = 0.0;
			}
		}
	}

	width = width / stride;
	height = height / stride;


	return 0;
}

float *dense(float *inputLayer, float *weights, int inputLayerSize, int outputLayerSize) {

	float* outputLayer = new float[outputLayerSize];
	for (int i = 0; i < outputLayerSize; i++) {
		outputLayer[i] = 0.0;
	}
	int a;

	for (int i = 0; i < outputLayerSize; i++) {
		for (int j = 0; j < inputLayerSize; j++) {
			outputLayer[i] += inputLayer[j] * weights[j*outputLayerSize+i];
		}
	}

	return outputLayer;
}