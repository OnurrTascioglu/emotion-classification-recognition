#pragma once
#include <windows.h>
float* conv1(BYTE* image, float* weights, int &width, int &height, int maskSize, int maskCount, int imageCount );
float* convHidden(float* feature, float* weights, int& fWidth, int& fHeight, int maskSize, int maskCount, int maskDim);
float* batchNormalizationConv(float* feature, float* batchWeights, int width, int height , int featureCount);
float* batchNormalizationDense(float* input, float* batchWeights, int inputSize);
float* reLU(float* feature, int width, int height , int featureCount);
float* maxPooling(float* feature, int &width, int &height, int  featureCount, int pool, int stride);
float* dense(float* inputLayer, float* weights, int inputLayerSize, int outputLayerSize);
float* flatten(float* features, int width, int height, int featureCount);
float* softmax(float* input, int size);