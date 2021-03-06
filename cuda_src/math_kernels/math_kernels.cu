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

__device__ float TrigRes(float const x) {
  return ((0.75f * x * x - 4.71239f * x + 5.9022f) * __cosf(x) + (-0.0833333f * x * x + 0.523599f * x - 0.803949f) * __cosf(3.0f * x) + 4.5f * x - 1.5f * x * __sinf(x) + 0.0555556f * x * __sinf(3.0f * x) + 6.96239f * __sinf(x) + 0.0754671f * __sinf(3.0f * x))/(9.0f * 3.141592653f);
}

__device__ float PolyNormalRes(float const x) {
  return 0.20019404249547249f - 0.01066466223648254f * x + 0.027284743817578543f * x * x + 0.006805423711959009f * x * x * x - 0.00110029250856299f * x * x * x * x;
}

__device__ float PolyNormalCachedRes(float const x) {
  float const x2{x * x};
  float const x3{x2 * x};
  float const x4{x3 * x};
  return 0.20019404249547249f - 0.01066466223648254f * x + 0.027284743817578543f * x2 + 0.006805423711959009f * x3 - 0.00110029250856299f * x4;
}

__device__ float PolyNestedRes(float const x) {
  return 0.20019404249547249f - 0.01066466223648254f * x * (1.0f - 2.558425500269543f * x * (1.0f + 0.24942230564666426f * x * (1.0f - 0.1616787661037875f * x)));
}

__device__ float PolyRootsRes(float const x) {
  return -0.011f * (x - 9.0517f) * (x + 3.8958f) * (x * (x - 0.5146f) + 5.1595f);
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

__global__ void TrigFunc_4(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};
  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{TrigRes(x1)};
    const float res2{TrigRes(x2)};
    const float res3{TrigRes(x1 + 1.0f)};
    const float res4{TrigRes(x2 + 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3;
  }
}

__global__ void PolyNormalFunc_4(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalRes(x1)};
    const float res2{PolyNormalRes(x2)};
    const float res3{PolyNormalRes(x1 + 1.0f)};
    const float res4{PolyNormalRes(x2 + 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3;
  }
}

__global__ void PolyNormalCachedFunc_4(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalCachedRes(x1)};
    const float res2{PolyNormalCachedRes(x2)};
    const float res3{PolyNormalCachedRes(x1 + 1.0f)};
    const float res4{PolyNormalCachedRes(x2 + 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3;
  }
}

__global__ void PolyNestedFunc_4(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNestedRes(x1)};
    const float res2{PolyNestedRes(x2)};
    const float res3{PolyNestedRes(x1 + 1.0f)};
    const float res4{PolyNestedRes(x2 + 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3;
  }
}

__global__ void PolyRootsFunc_4(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyRootsRes(x1)};
    const float res2{PolyRootsRes(x2)};
    const float res3{PolyRootsRes(x1 + 1.0f)};
    const float res4{PolyRootsRes(x2 + 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3;
  }
}

__global__ void TrigFunc_6(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};
  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{TrigRes(x1)};
    const float res2{TrigRes(x2)};
    const float res3{TrigRes(x1 + 1.0f)};
    const float res4{TrigRes(x2 + 1.0f)};
    const float res5{TrigRes(x1 - 1.0f)};
    const float res6{TrigRes(x2 - 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5;
  }
}

__global__ void PolyNormalFunc_6(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalRes(x1)};
    const float res2{PolyNormalRes(x2)};
    const float res3{PolyNormalRes(x1 + 1.0f)};
    const float res4{PolyNormalRes(x2 + 1.0f)};
    const float res5{PolyNormalRes(x1 - 1.0f)};
    const float res6{PolyNormalRes(x2 - 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5;
  }
}

__global__ void PolyNormalCachedFunc_6(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalCachedRes(x1)};
    const float res2{PolyNormalCachedRes(x2)};
    const float res3{PolyNormalCachedRes(x1 + 1.0f)};
    const float res4{PolyNormalCachedRes(x2 + 1.0f)};
    const float res5{PolyNormalCachedRes(x1 - 1.0f)};
    const float res6{PolyNormalCachedRes(x2 - 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5;
  }
}

__global__ void PolyNestedFunc_6(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNestedRes(x1)};
    const float res2{PolyNestedRes(x2)};
    const float res3{PolyNestedRes(x1 + 1.0f)};
    const float res4{PolyNestedRes(x2 + 1.0f)};
    const float res5{PolyNestedRes(x1 - 1.0f)};
    const float res6{PolyNestedRes(x2 - 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5;
  }
}

__global__ void PolyRootsFunc_6(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyRootsRes(x1)};
    const float res2{PolyRootsRes(x2)};
    const float res3{PolyRootsRes(x1 + 1.0f)};
    const float res4{PolyRootsRes(x2 + 1.0f)};
    const float res5{PolyRootsRes(x1 - 1.0f)};
    const float res6{PolyRootsRes(x2 - 1.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5;
  }
}

__global__ void TrigFunc_8(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};
  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{TrigRes(x1)};
    const float res2{TrigRes(x2)};
    const float res3{TrigRes(x1 + 1.0f)};
    const float res4{TrigRes(x2 + 1.0f)};
    const float res5{TrigRes(x1 - 1.0f)};
    const float res6{TrigRes(x2 - 1.0f)};
    const float res7{TrigRes(x1 + 2.0f)};
    const float res8{TrigRes(x2 + 2.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
  }
}

__global__ void PolyNormalFunc_8(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalRes(x1)};
    const float res2{PolyNormalRes(x2)};
    const float res3{PolyNormalRes(x1 + 1.0f)};
    const float res4{PolyNormalRes(x2 + 1.0f)};
    const float res5{PolyNormalRes(x1 - 1.0f)};
    const float res6{PolyNormalRes(x2 - 1.0f)};
    const float res7{PolyNormalRes(x1 + 2.0f)};
    const float res8{PolyNormalRes(x2 + 2.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
  }
}

__global__ void PolyNormalCachedFunc_8(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNormalCachedRes(x1)};
    const float res2{PolyNormalCachedRes(x2)};
    const float res3{PolyNormalCachedRes(x1 + 1.0f)};
    const float res4{PolyNormalCachedRes(x2 + 1.0f)};
    const float res5{PolyNormalCachedRes(x1 - 1.0f)};
    const float res6{PolyNormalCachedRes(x2 - 1.0f)};
    const float res7{PolyNormalCachedRes(x1 + 2.0f)};
    const float res8{PolyNormalCachedRes(x2 + 2.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
  }
}

__global__ void PolyNestedFunc_8(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyNestedRes(x1)};
    const float res2{PolyNestedRes(x2)};
    const float res3{PolyNestedRes(x1 + 1.0f)};
    const float res4{PolyNestedRes(x2 + 1.0f)};
    const float res5{PolyNestedRes(x1 - 1.0f)};
    const float res6{PolyNestedRes(x2 - 1.0f)};
    const float res7{PolyNestedRes(x1 + 2.0f)};
    const float res8{PolyNestedRes(x2 + 2.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
  }
}

__global__ void PolyRootsFunc_8(const float * const data_in, float * const data_out, size_t const size) {
  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyRootsRes(x1)};
    const float res2{PolyRootsRes(x2)};
    const float res3{PolyRootsRes(x1 + 1.0f)};
    const float res4{PolyRootsRes(x2 + 1.0f)};
    const float res5{PolyRootsRes(x1 - 1.0f)};
    const float res6{PolyRootsRes(x2 - 1.0f)};
    const float res7{PolyRootsRes(x1 + 2.0f)};
    const float res8{PolyRootsRes(x2 + 2.0f)};

    data_out[idx] = res2 - res1 + res4 - res3 + res6 - res5 + res8 - res7;
  }
}
