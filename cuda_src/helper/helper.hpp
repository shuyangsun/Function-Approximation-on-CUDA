/*
 ============================================================================
 Name        : helper.hpp
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#ifndef OPPC_CUDA_SRC_HELPER_HELPER_HPP_
#define OPPC_CUDA_SRC_HELPER_HELPER_HPP_

#define TIME_KERNEL(X)\
start_time = CPUSecond();\
X<<<grid_dim, block_dim>>>(data_in_d, data_out_d, num_ele);\
cudaDeviceSynchronize();\
end_time = CPUSecond();\
duration += end_time - start_time;

void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CHECK_CUDA_ERR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

double CPUSecond();
float RandomFloat();

#endif

