#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "CpuGpuMem.h"

void cpuGpuAlloc(CpuGpuMem* p_cg, enum cpuGpuMemVar keyword, int sizeOfType);

void cpuGpuFree(CpuGpuMem* p_cg, enum cpuGpuMemVar keyword);


void cpuGpuMemCopy(enum cudaMemcpyKind copyKind, struct CpuGpuMem* p_cg, void* destPtr, void* srcPtr, int allocSize);

void cpuGpuPin(void* ptr, int allocSize);

void cpuGpuUnpin(void* ptr, int allocSize);


