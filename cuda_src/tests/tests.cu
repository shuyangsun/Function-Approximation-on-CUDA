/*
 ============================================================================
 Name        : tests.hpp
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#include "tests.hpp"
#include "../helper/helper.hpp"
#include "../math_kernels/math_kernels.hpp"

#include <iostream>
#include <iomanip>

enum class KernelFunc {
  Trigonometry,
  PolynomialNormal,
  PolynomialNormalCached,
  PolynomialNested,
  PolynomialRoots
};

void TestKernelFunc(KernelFunc const func, const float * const data_h, size_t const data_size, unsigned int const block_dim_x);

void TestFPOs(dim3 const grid_dim, dim3 const block_dim) {
  std::cout << "Started testing FPO performance..." << std::endl;

  constexpr size_t num_loop{100};

  for (size_t i{0}; i < num_loop; ++i) { FPO_1<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_2<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_3<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_4<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_5<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_6<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_7<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_8<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_9<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_10<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_11<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_12<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_13<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_14<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_15<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { FPO_16<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  cudaDeviceReset();

  std::cout << "FPO performance testing finished." << std::endl;
}

void TestSFUs(dim3 const grid_dim, dim3 const block_dim) {
  std::cout << "Started testing SFU performance..." << std::endl;

  constexpr size_t num_loop{100};

  for (size_t i{0}; i < num_loop; ++i) { SFU_1<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_2<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_3<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_4<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_5<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_6<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_7<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_8<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_9<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_10<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_11<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_12<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_13<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_14<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_15<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  for (size_t i{0}; i < num_loop; ++i) { SFU_16<<<grid_dim, block_dim>>>(); cudaDeviceSynchronize(); }
  cudaDeviceReset();

  std::cout << "SFU performance testing finished." << std::endl;
}

void TestMathKernels(float const gig_count) {
  // Customization for testing.
  size_t const num_loop{50};

  // Initialize attributes
  size_t const data_size{static_cast<size_t>(gig_count * 1024 * 1024 * 1024)};
  size_t const num_ele{data_size / sizeof(float)};

  // Generate random data array
  float * const data_h{reinterpret_cast<float*>(malloc(data_size))};
  std::cout << std::setprecision(2) << "Generating random float array of " << gig_count << "GB..." << std::endl;
  srand(time(NULL));
  for (size_t i{0}; i < num_ele; ++i) {
    data_h[i] = RandomFloat();
  }

  std::cout << "Testing math kernels..." << std::endl;

  // Start inner loop (repetition loop)
  for (size_t j{0}; j < num_loop; ++j) {
    unsigned int const block_dim_x{1024};

    TestKernelFunc(KernelFunc::Trigonometry, data_h, data_size, block_dim_x);
    TestKernelFunc(KernelFunc::PolynomialNormal, data_h, data_size, block_dim_x);
    TestKernelFunc(KernelFunc::PolynomialNormalCached, data_h, data_size, block_dim_x);
    TestKernelFunc(KernelFunc::PolynomialNested, data_h, data_size, block_dim_x);
    TestKernelFunc(KernelFunc::PolynomialRoots, data_h, data_size, block_dim_x);
  }

  free(data_h);
  CHECK_CUDA_ERR(cudaDeviceReset());

  std::cout << "Math kernel testing finished." << std::endl;
}

void TestKernelFunc(KernelFunc const func, const float * const data_h, size_t const data_size, unsigned int const block_dim_x) {
  float *data_d;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&data_d), data_size));
  CHECK_CUDA_ERR(cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice));

  float *res;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&res), data_size/2));

  size_t const num_ele{data_size / sizeof(float)};
  dim3 const block_dim{block_dim_x};
  dim3 const grid_dim{static_cast<unsigned int>((num_ele + block_dim.x - 1) / block_dim.x)};

  switch (func) {
    case KernelFunc::Trigonometry:
      TrigFunc_2<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      TrigFunc_4<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      TrigFunc_6<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      TrigFunc_8<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      break;
    case KernelFunc::PolynomialNormal:
      PolyNormalFunc_2<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNormalFunc_4<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNormalFunc_6<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNormalFunc_8<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      break;
    case KernelFunc::PolynomialNormalCached:
      PolyNormalCachedFunc_2<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNormalCachedFunc_4<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNormalCachedFunc_6<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNormalCachedFunc_8<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      break;
    case KernelFunc::PolynomialNested:
      PolyNestedFunc_2<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNestedFunc_4<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNestedFunc_6<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyNestedFunc_8<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      break;
    case KernelFunc::PolynomialRoots:
      PolyRootsFunc_2<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyRootsFunc_4<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyRootsFunc_6<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      PolyRootsFunc_8<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      break;
    default:
      break;
  };

  CHECK_CUDA_ERR(cudaFree(data_d));
  CHECK_CUDA_ERR(cudaFree(res));
}

