#pragma once
#include "cpuGpuMem.h"

void conv1ExecGPU(CpuGpuMem* cg);
void convHidden1ExecGPU(CpuGpuMem* cg);
void dense1ExecGPU(CpuGpuMem* cg);
void dense2ExecGPU(CpuGpuMem* cg);