#pragma once
#include "cpuGpuMem.h"

void conv1ExecGPU(CpuGpuMem* cg, const int maskCount);
void batchAndReLuConv1ExecGPU(CpuGpuMem* cg, const int featureCount);