/*
 ============================================================================
 Name        : helper.cu
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#include "helper.hpp"

#include <cstdlib>
#include <iostream>

float RandomFloat() {
  return ((float)rand())/((float)rand());
}

double CPUSecond() {
  return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
  if (err == cudaSuccess)
    return;
  std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
  exit (1);
}

