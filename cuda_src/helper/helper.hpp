/*
 ============================================================================
 Name        : helper.hpp
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#ifndef FAST_INTEGRAL_CUDA_SRC_HELPER_HPP_
#define FAST_INTEGRAL_CUDA_SRC_HELPER_HPP_

void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CHECK_CUDA_ERR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

double CPUSecond();
float RandomFloat();

#endif

