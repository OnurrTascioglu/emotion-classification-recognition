#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "CpuGpuMem.h"

void cpuGpuAlloc(CpuGpuMem* p_cg, char keyword);
void cpuGpuFree(CpuGpuMem* p_cg, char keyword);


void cpu_gpu_alloc(CpuGpuMem* p_cg);
void cpu_gpu_free(CpuGpuMem* p_cg);

void cpu_gpu_set_numbers(CpuGpuMem* p_cg);

void cpu_gpu_h_to_d(CpuGpuMem* p_cg);
void cpu_gpu_d_to_h(CpuGpuMem* p_cg);
void cpu_gpu_pin(CpuGpuMem* p_cg);
void cpu_gpu_unpin(CpuGpuMem* p_cg);