/*
 ============================================================================
 Name        : main.cu
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 ============================================================================
 */

#include <iostream>
#include <iomanip>

#include "cuda_src/tests/tests.hpp"
#include "cuda_src/helper/helper.hpp"

enum class KernelFunc {
  Polynomial, Trigonometry
};

double TimeKernel(const KernelFunc func, const float * const data_h, size_t const data_size, unsigned int const block_dim_x);

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size);
__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size);

int main(int arc, char *argv[]) {

  dim3 const grid_dim{1024};
  dim3 const block_dim{1024};
  TestFPOs(grid_dim, block_dim);
  TestSFUs(grid_dim, block_dim);

  // Customization for testing.
  size_t const num_loop{10};
  float const max_gig_count{1.0f};
  float const step_size{0.5f};

  // Initialize attributes
  size_t const max_data_size{static_cast<size_t>(max_gig_count * 1024 * 1024 * 1024)};
  size_t const max_num_ele{max_data_size / sizeof(float)};

  // Generate random data array
  float * const data_h{reinterpret_cast<float*>(malloc(max_data_size))};
  std::cout << std::setprecision(2) << "Generating random float array of " << max_gig_count << "GB..." << std::endl;
  srand(time(NULL));
  for (size_t i{0}; i < max_num_ele; ++i) {
    data_h[i] = RandomFloat();
  }
  std::cout << "Finished generating random float array." << std::endl;

  // Start outter loop (data size loop)
  for (float i{step_size}; i <= max_gig_count; i += step_size) {
    std::cout << std::setprecision(2) << "------------ " << i << "GB ------------" << std::endl;
    double duration_trig{0.0};
    double duration_poly{0.0};

    // Start inner loop (repetition loop)
    for (size_t j{0}; j < num_loop; ++j) {

      float const gig_count{i};
      size_t const data_size{static_cast<size_t>(gig_count * (1 << 30))};
      unsigned int const block_dim_x{1024};

      // Trig Kernel
      double const tmp_dur_trig{TimeKernel(KernelFunc::Trigonometry, data_h, data_size, block_dim_x)};
      duration_trig += tmp_dur_trig * 1000.0;

      // Poly Kernel
      double const tmp_dur_poly{TimeKernel(KernelFunc::Polynomial, data_h, data_size, block_dim_x)};
      duration_poly += tmp_dur_poly * 1000.0;
    }

    // Calculate average
    duration_trig /= num_loop;
    duration_poly /= num_loop;

    // Print out information
    std::cout << std::setprecision(3) << "Finished trig kernel in average " << duration_trig << " ms." << std::endl;
    std::cout << std::setprecision(3) << "Finished poly kernel in average " << duration_poly << " ms." << std::endl;
    std::cout << std::setprecision(5) << "Trig time / Poly time: " << duration_trig/duration_poly << std::endl;
    std::cout << std::setprecision(3) << "Speed up: " << (1.0f - duration_poly/duration_trig) * 100 << "%" << std::endl;
  }

  free(data_h);
  CHECK_CUDA_ERR(cudaDeviceReset());

  std::cout << "-------------------------------" << std::endl;
  return 0;
}

double TimeKernel(const KernelFunc func, const float * const data_h, size_t const data_size, unsigned int const block_dim_x) {
  float *data_d;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&data_d), data_size));
  CHECK_CUDA_ERR(cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice));

  float *res;
  CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&res), data_size/2));

  size_t const num_ele{data_size / sizeof(float)};
  dim3 const block_dim{block_dim_x};
  dim3 const grid_dim{static_cast<unsigned int>((num_ele + block_dim.x - 1) / block_dim.x)};

  double start{0.0f};
  double end{0.0f};

  switch (func) {
    case KernelFunc::Trigonometry:
      start = CPUSecond();
      TrigFunc<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      end = CPUSecond();
      break;
    case KernelFunc::Polynomial:
      start = CPUSecond();
      PolyFunc<<<grid_dim, block_dim>>>(data_d, res, num_ele);
      cudaDeviceSynchronize();
      end = CPUSecond();
      break;
    default:
      break;
  };

  CHECK_CUDA_ERR(cudaFree(data_d));
  CHECK_CUDA_ERR(cudaFree(res));

  return end - start;
}

__device__ float PolyRes(float const val) {
  return -0.011f * (val - 9.0517f) * (val + 3.8958f) * (val * (val - 0.5146f) + 5.1595f);
}

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size) {

  const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
  const size_t idx_2{idx * 2};

  if (idx_2 < size) {
    const float x1{data_in[idx_2]};
    const float x2{data_in[idx_2 + 1]};

    const float res1{PolyRes(x1)};
    const float res2{PolyRes(x2)};

    data_out[idx] = res2 - res1;
  }
}

__device__ float TrigRes(float const val) {
  return ((0.75f * val * val - 4.71239f * val + 5.9022f) * __cosf(val) + (-0.0833333f * val * val + 0.523599f * val - 0.803949f) * __cosf(3.0f * val) + 4.5f * val - 1.5f * val * __sinf(val) + 0.0555556f * val * __sinf(3.0f * val) + 6.96239f * __sinf(val) + 0.0754671f * __sinf(3.0f * val))/(9.0f * 3.141592653f);
}

__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size) {
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

