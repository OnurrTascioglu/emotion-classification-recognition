#include <windows.h>
#include <cmath>


BYTE* conv1(BYTE* inputImage, float* weights, int width, int height, int maskSize, int maskCount , int imageCount) {


	
	
	int rMatrixWidth = width - maskSize + 1;
	int rMatrixHeight = height - maskSize + 1;


	float* masks = new float[maskSize * maskSize * maskCount];
	float* resultImages = new float[maskCount * rMatrixWidth * rMatrixHeight];
	BYTE* image = new BYTE[width * height];
	BYTE* bResultImages = new BYTE[maskCount * rMatrixWidth * rMatrixHeight];
	
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

	int max = 0;
	int min = 0;
	float ratio = 0.0;

	for (int m = 0; m < maskCount; m++) {
		for (int i = 0; i < rMatrixWidth * rMatrixHeight; i++) {
			if ((int)resultImages[(m * rMatrixWidth * rMatrixHeight) + i] > max) {
				max = resultImages[(m * rMatrixWidth * rMatrixHeight) + i];
			}
			if ((int)resultImages[(m * rMatrixWidth * rMatrixHeight) + i] < min) {
				min = resultImages[(m * rMatrixWidth * rMatrixHeight) + i];
			}
		}
		for (int i = 0; i < rMatrixWidth * rMatrixHeight; i++) {
			resultImages[(m * rMatrixWidth * rMatrixHeight) + i] = resultImages[(m * rMatrixWidth * rMatrixHeight) + i] - (min);
		}
		ratio = (float)(max - min) / 254;

		for (int i = 0; i < rMatrixWidth * rMatrixHeight; i++) {
			bResultImages[(m * rMatrixWidth * rMatrixHeight) + i] = resultImages[(m * rMatrixWidth * rMatrixHeight) + i] / ratio;
		}
	}


	delete[] masks;
	delete[] image;
	delete[] resultImages;
	return bResultImages;
}
