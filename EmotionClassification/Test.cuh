#pragma once
#include "cpuGpuMem.h"

void conv1ExecGPU(CpuGpuMem* cg);
void convHidden1ExecGPU(CpuGpuMem* cg);

void model2Conv1ExecGPU(CpuGpuMem* cg);
void model2Conv2ExecGpu(CpuGpuMem* cg);
void model2Conv3ExecGpu(CpuGpuMem* cg);
void model2Conv4ExecGpu(CpuGpuMem* cg);

void dense1ExecGPU(CpuGpuMem* cg);
void dense2ExecGPU(CpuGpuMem* cg);

void model2Dense1ExecGPU(CpuGpuMem* cg);
void model2Dense2ExecGPU(CpuGpuMem* cg);
void model2Dense3ExecGPU(CpuGpuMem* cg);