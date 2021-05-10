#pragma once
#include <windows.h>
float* conv1(BYTE* image, float* weights, int &width, int &height, int maskSize, int maskCount, int imageCount );
float* batchNormalization(float* feature, int width, int height , int featureCount);
float* reLU(float* feature, int width, int height , int featureCount);
float* maxPooling(float* feature, int &width, int &height, int  featureCount, int pool, int stride);