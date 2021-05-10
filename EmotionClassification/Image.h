#pragma once
#include <windows.h>
#include <string>


BYTE* LoadBMP(int* width, int* height, long* size, LPCTSTR bmpfile);
bool  SaveBMP(BYTE* Buffer, int width, int height, long paddedsize, LPCTSTR bmpfile);
BYTE* ConvertBMPToIntensity(BYTE* Buffer, int width, int height);
BYTE* ConvertIntensityToBMP(BYTE* Buffer, int width, int height, long* newsize);
void readWeightFromFile(float* convWeight, std::string filePath);