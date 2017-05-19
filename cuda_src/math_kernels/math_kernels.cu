/*
 ============================================================================
 Name        : math_kernels.cu
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#include "math_kernels.hpp"

__global__ void FPO_1() {
  float const res{1.0f * 1.5f};
}
__global__ void FPO_2() {
  float const res{1.0f * 1.5f + 2.0f};
}
__global__ void FPO_3() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f};
}

__global__ void FPO_4() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f};
}

__global__ void FPO_5() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f};
}

__global__ void FPO_6() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f};
}

__global__ void FPO_7() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f};
}

__global__ void FPO_8() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f};
}

__global__ void FPO_9() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f};
}

__global__ void FPO_10() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f};
}

__global__ void FPO_11() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f * 8.9f};
}

__global__ void FPO_12() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f * 8.9f - 3.5f};
}

__global__ void FPO_13() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f * 8.9f - 3.5f + 11.6f};
}

__global__ void FPO_14() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f * 8.9f - 3.5f + 11.6f - 7.8f};
}

__global__ void FPO_15() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f * 8.9f - 3.5f + 11.6f - 7.8f * 3.0f};
}

__global__ void FPO_16() {
  float const res{1.0f * 1.5f + 2.0f * 8.0f - 9.6f * 2.0f + 3.0f - 2.5f * 0.5f - 2.4f + 2.3f * 8.9f - 3.5f + 11.6f - 7.8f * 3.0f - 2.0f};
}


__global__ void SFU_1() {
  float const res{__sinf(2.0f)};
}

__global__ void SFU_2() {
  float const res{__sinf(__cosf(2.0f))};
}

__global__ void SFU_3() {
  float const res{__sinf(__cosf(__sinf(2.0f)))};
}

__global__ void SFU_4() {
  float const res{__sinf(__cosf(__sinf(__cosf(2.0f))))};
}

__global__ void SFU_5() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(2.0f)))))};
}

__global__ void SFU_6() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(2.0f))))))};
}

__global__ void SFU_7() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(2.0f)))))))};
}

__global__ void SFU_8() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(2.0f))))))))};
}

__global__ void SFU_9() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(2.0f)))))))))};
}

__global__ void SFU_10() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(2.0f))))))))))};
}

__global__ void SFU_11() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(2.0f)))))))))))};
}

__global__ void SFU_12() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(2.0f))))))))))))};
}

__global__ void SFU_13() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(2.0f)))))))))))))};
}

__global__ void SFU_14() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(2.0f))))))))))))))};
}

__global__ void SFU_15() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(2.0f)))))))))))))))};
}

__global__ void SFU_16() {
  float const res{__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(__sinf(__cosf(2.0f))))))))))))))))};
}


