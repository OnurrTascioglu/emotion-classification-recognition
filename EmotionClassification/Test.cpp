#include <windows.h>
#include <cmath>

#define BIAS 1


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
		masks[maskCount * maskSize * maskSize + i] = weights[maskCount * maskSize * maskSize + i];
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

float* convHidden(float* feature, float* weights, int& fWidth, int& fHeight, int maskSize, int maskCount, int maskDim) {

	float* masks = new float[maskSize * maskSize * maskCount * maskDim + maskCount];
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
	for (int i = 0; i < maskCount; i++)
	{
		masks[maskCount * maskDim * maskSize * maskSize + i] = weights[maskCount * maskDim * maskSize * maskSize + i];
	}

	for (int c = 0; c < maskCount; c++) {
		for (int i = 0; i < rMatrixHeight; i++) {
			for (int j = 0; j < rMatrixWidth; j++) {
				for (int d = 0; d < maskDim; d++) {
					for (int k = 0; k < maskSize * maskSize; k++) {
						int mCol = k % maskSize;
						int mRow = k / maskSize;
						resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] +=
							(float)feature[d * fWidth * fHeight + (fWidth * i + j) + mRow * fWidth + mCol] * masks[c * (maskDim * maskSize * maskSize) + d * maskSize * maskSize + k];
					}

				}
				resultImages[(c * rMatrixWidth * rMatrixHeight) + i * rMatrixWidth + j] += BIAS * masks[maskCount * maskDim * maskSize * maskSize + c];
			}
		}
	}
	fWidth = fWidth - maskSize + 1;
	fHeight = fHeight - maskSize + 1;

	delete[] masks;
	return resultImages;
}

//float* batchNormalization(float* feature, float* batchWeights, int width, int height, int featureCount) { //Batch normalize yöntemi
//
//	float sum = 0.0;// aritmetik ortalama için
//	float sDeviation = 0.0; // standart sapma için
//
//	for (int m = 0; m < featureCount; m++) {
//		for (int i = 0; i < width * height; i++)
//		{
//			sum += feature[(m * width * height) + i];
//		}
//		sum = sum / (float)(width * height); // aritmetik ortalama alýnýr
//
//		for (int i = 0; i < width * height; i++)
//		{
//			sDeviation += powf((feature[(m * width * height) + i] - sum), 2);
//		}
//		sDeviation = sqrt(sDeviation / (float)((width * height) - 1));
//
//		for (int i = 0; i < width * height; i++)
//		{
//			feature[(m * width * height) + i] = (feature[(m * width * height) + i] - sum) / sDeviation;
//			feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];
//		}
//
//		sum = 0.0;
//		sDeviation = 0.0;
//	}
//
//	return feature;
//}

float* batchNormalizationConv(float* feature, float* batchWeights, int width, int height, int featureCount) { //Batch normalize yöntemi

	float sDeviation = 0.0; // standart sapma için

	for (int m = 0; m < featureCount; m++) {

		sDeviation = sqrt(batchWeights[(featureCount * 3) + m]);

		for (int i = 0; i < width * height; i++)
		{
			feature[(m * width * height) + i] = (feature[(m * width * height) + i] - batchWeights[featureCount * 2 + m]) / sDeviation;
			feature[(m * width * height) + i] = feature[(m * width * height) + i] * batchWeights[m] + batchWeights[featureCount + m];
		}

		sDeviation = 0.0;
	}

	return feature;
}

float* batchNormalizationDense(float* input, float* batchWeights, int inputSize) {


	float sDeviation = 0.0; // standart sapma için

	for (int i = 0; i < inputSize; i++) {
		sDeviation = sqrt(batchWeights[(inputSize * 3) + i]);
		input[i] = (input[i] - batchWeights[(inputSize * 2) + i]) / sDeviation;
		input[i] = input[i] * batchWeights[i] + batchWeights[inputSize + i];
	}
	return input;
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

float* flatten(float* features, int width, int height, int featureCount) {

	float* flattenFeatures = new float[width * height * featureCount];
	int count = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int f = 0; f < featureCount; f++) {
				flattenFeatures[count] = features[f * width * height + i * width + j];
				count++;
			}
		}
	}
	for (int i = 0; i < width * height * featureCount; i++)
		features[i] = flattenFeatures[i];

	return features;
}

float* dense(float* inputLayer, float* weights, int inputLayerSize, int outputLayerSize) {

	float* outputLayer = new float[outputLayerSize];
	for (int i = 0; i < outputLayerSize; i++) {
		outputLayer[i] = 0.0;
	}
	int a;

	for (int i = 0; i < outputLayerSize; i++) {
		for (int j = 0; j < inputLayerSize; j++) {
			outputLayer[i] += inputLayer[j] * weights[j * outputLayerSize + i];
		}
		outputLayer[i] += BIAS * weights[inputLayerSize * outputLayerSize + i];
		int a = 0;
	}

	return outputLayer;
}

float* softmax(float* input, int size) {

	float m = -INFINITY;
	for (size_t i = 0; i < size; i++) {
		if (input[i] > m) {
			m = input[i];
		}
	}

	float sum = 0.0;
	for (size_t i = 0; i < size; i++) {
		sum += expf(input[i] - m);
	}

	float offset = m + logf(sum);
	for (size_t i = 0; i < size; i++) {
		input[i] = expf(input[i] - offset);
	}

	return input;
}