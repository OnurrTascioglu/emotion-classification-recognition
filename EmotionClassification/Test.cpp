#include <windows.h>
#include <cmath>

float* conv1(BYTE* image, float* weights, int width, int height, int maskSize, int maskDim) {

	float* masks = new float [maskSize * maskSize * maskDim];

	for (int i = 0; i < maskSize * maskSize; i++) {
		for (int j = 0; j < maskDim; j++) {
			masks[j * maskSize * maskSize + i] = weights[i*maskDim+j];
		}
	}

	for (int i = 0; i < maskSize * maskSize * maskDim; i++) {
		masks[i];
	}

	return NULL;
}
