/*
 ============================================================================
 Name        : fast_integral_practice.cu
 Author      : Shuyang Sun
 Version     :
 Copyright   : Shuyang Sun, all rights reserved.
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

double CPUSecond();

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size);
__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size);

int main(void)
{
	size_t const num_loop{10};
	size_t const max_gig_count{6};
	const size_t max_data_size{max_gig_count * 1024 * 1024 * 1024};
	const size_t max_num_ele{max_data_size / sizeof(float)};
	float *data_h{reinterpret_cast<float*>(malloc(max_data_size))};

	srand(time(NULL));
	std::cout << "Generating random float array..." << std::endl;
	for (size_t i{0}; i < max_num_ele; ++i) {
		data_h[i] = ((float)rand())/((float)rand());
	}
	std::cout << "Finished generating random float array." << std::endl;
	for (size_t i{0}; i < max_gig_count; ++i) {
		std::cout << "------------ " << i + 1 << "GB ------------" << std::endl;
		double duration_trig{0.0};
		double duration_poly{0.0};
		for (size_t j{0}; j < num_loop; ++j) {

			const size_t gig_count{i + 1};
			const size_t data_size{gig_count * 1024 * 1024 * 1024};
			const size_t num_ele{data_size / sizeof(float)};

			float *data_d;
			CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&data_d), data_size));
			CUDA_CHECK_RETURN(cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice));

			float *res;
			CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&res), data_size/2));

			const dim3 block_dim{1024};
			const dim3 grid_dim{static_cast<unsigned int>((num_ele + block_dim.x - 1) / block_dim.x)};

			// Trig Kernel
			double start{CPUSecond()};
			TrigFunc<<<grid_dim, block_dim>>>(data_d, res, num_ele);
			cudaDeviceSynchronize();
			double end{CPUSecond()};
			duration_trig += (end - start) * 1000.0;

			CUDA_CHECK_RETURN(cudaFree(res));
			CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&res), data_size/2));

			// Poly Kernel
			start = CPUSecond();
			PolyFunc<<<grid_dim, block_dim>>>(data_d, res, num_ele);
			cudaDeviceSynchronize();
			end = CPUSecond();
			duration_poly += (end - start) * 1000.0;

			CUDA_CHECK_RETURN(cudaFree(res));
			CUDA_CHECK_RETURN(cudaFree(data_d));

		}
		duration_trig /= num_loop;
		duration_poly /= num_loop;
		std::cout << "Finished trig kernel in average " << duration_trig << " ms." << std::endl;
		std::cout << "Finished poly kernel in average " << duration_poly << " ms." << std::endl;
		std::cout << "Trig time / Poly time: " << duration_trig/duration_poly << std::endl;
		std::cout << "Speed up: " << (1.0f - duration_poly/duration_trig) * 100 << "%" << std::endl;
	}

	free(data_h);
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}

__global__ void PolyFunc(const float * const data_in, float * const data_out, size_t const size) {

	const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
	const size_t idx_2{idx * 2};

	if (idx_2 < size) {
		const float a{data_in[idx_2]};
		const float b{data_in[idx_2 + 1]};
		const float res1 = 0.36f + 0.68f * a * (1 + 0.28f * a * (1 + 0.78f * a * (1 - 0.57f * a * (1 + 0.68f * a * (1 + 0.68f * a * (1 + 0.68f * a * (1 + 0.68f * a)))))));
		const float res2 = 0.36f + 0.68f * b * (1 + 0.28f * b * (1 + 0.78f * b * (1 - 0.57f * b * (1 + 0.68f * b * (1 + 0.68f * b * (1 + 0.68f * b * (1 + 0.68f * b)))))));

		data_out[idx] = res2 - res1;
	}
}

__global__ void TrigFunc(const float * const data_in, float * const data_out, size_t const size) {
	const size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
	const size_t idx_2{idx * 2};
	if (idx_2 < size) {
		const float a = data_in[idx_2];
		const float b = data_in[idx_2 + 1];
		const float res1 = 2.5f * __sinf(a) + 0.25f * __sinf(3 * a) - 8.5f * __cosf(a) + 9.3f * __cosf(3 * a) + 0.34f * __cosf(5 * a) + 9.3f * __cosf(7 * a);
		const float res2 = 2.5f * __sinf(b) + 0.25f * __sinf(3 * b) - 8.5f * __cosf(b) + 9.3f * __cosf(3 * b) + 0.34f * __cosf(5 * b) + 9.3f * __cosf(7 * b);

		data_out[idx] = res2 - res1;
	}
}

double CPUSecond() {
	return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

