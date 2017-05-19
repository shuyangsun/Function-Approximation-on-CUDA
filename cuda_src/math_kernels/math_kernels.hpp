/*
 ============================================================================
 Name        : math_kernels.hpp
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#ifndef OPPC_CUDA_SRC_MATH_KERNELS_MATH_KERNELS_HPP_
#define OPPC_CUDA_SRC_MATH_KERNELS_MATH_KERNELS_HPP_

__global__ void FPO_1();
__global__ void FPO_2();
__global__ void FPO_3();
__global__ void FPO_4();
__global__ void FPO_5();
__global__ void FPO_6();
__global__ void FPO_7();
__global__ void FPO_8();
__global__ void FPO_9();
__global__ void FPO_10();
__global__ void FPO_11();
__global__ void FPO_12();
__global__ void FPO_13();
__global__ void FPO_14();
__global__ void FPO_15();
__global__ void FPO_16();

__global__ void SFU_1();
__global__ void SFU_2();
__global__ void SFU_3();
__global__ void SFU_4();
__global__ void SFU_5();
__global__ void SFU_6();
__global__ void SFU_7();
__global__ void SFU_8();
__global__ void SFU_9();
__global__ void SFU_10();
__global__ void SFU_11();
__global__ void SFU_12();
__global__ void SFU_13();
__global__ void SFU_14();
__global__ void SFU_15();
__global__ void SFU_16();

__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size);
__global__ void PolyNormalFunc(const float * const data_in, float * const data_out, size_t const size);

#endif

