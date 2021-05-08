#pragma once
#include <windows.h>
float* conv1(BYTE* image, float* weights, int width, int height, int maskSize, int maskCount, int imageCount ,int &sizeW, int &sizeH);
float* batchNormalization(float* feature, int width, int height , int maskCount);
float* reLU(float* feature, int width, int height , int maskCount);