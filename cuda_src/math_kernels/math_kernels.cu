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

__device__ float TrigRes(float const val) {
  return ((0.75f * val * val - 4.71239f * val + 5.9022f) * __cosf(val) + (-0.0833333f * val * val + 0.523599f * val - 0.803949f) * __cosf(3.0f * val) + 4.5f * val - 1.5f * val * __sinf(val) + 0.0555556f * val * __sinf(3.0f * val) + 6.96239f * __sinf(val) + 0.0754671f * __sinf(3.0f * val))/(9.0f * 3.141592653f);
}

__device__ float PolyNormalRes(float const val) {
  return -0.011f * (val - 9.0517f) * (val + 3.8958f) * (val * (val - 0.5146f) + 5.1595f);
}

__device__ float PolyNormalCachedRes(float const val) {
  return -0.011f * (val - 9.0517f) * (val + 3.8958f) * (val * (val - 0.5146f) + 5.1595f);
}

__device__ float PolyNestedRes(float const val) {
  return -0.011f * (val - 9.0517f) * (val + 3.8958f) * (val * (val - 0.5146f) + 5.1595f);
}

__device__ float PolyRootsRes(float const val) {
  return -0.011f * (val - 9.0517f) * (val + 3.8958f) * (val * (val - 0.5146f) + 5.1595f);
}

__global__ void TrigFunc_2(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};
  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{TrigRes(x1)};
    const float res2{TrigRes(x2)};

    data_out[idx] = res2 - res1;
  }
}

__global__ void PolyNormalFunc_2(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalRes(x1)};
    const float res2{PolyNormalRes(x2)};

    data_out[idx] = res2 - res1;
  }
}

__global__ void PolyNormalCachedFunc_2(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalCachedRes(x1)};
    const float res2{PolyNormalCachedRes(x2)};

    data_out[idx] = res2 - res1;
  }
}

__global__ void PolyNestedFunc_2(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNestedRes(x1)};
    const float res2{PolyNestedRes(x2)};

    data_out[idx] = res2 - res1;
  }
}

__global__ void PolyRootsFunc_2(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyRootsRes(x1)};
    const float res2{PolyRootsRes(x2)};

    data_out[idx] = res2 - res1;
  }
}

