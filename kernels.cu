#include "assert.h"
#include "kernels.h"
#include <stdio.h>



__global__
void gpu_add_double_vectors(int n, double *this_dev, double *other_dev)
{
  
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (tidx < n) {
    this_dev[tidx] = other_dev[tidx] + this_dev[tidx];
    // printf("tidx= %d, other_dev= %f this_dev= %f \n", tidx, other_dev[tidx], this_dev[tidx]);
  }
  
}

void kernel_gpu_add_double_vectors(int size, double *this_dev, double *other_dev) {
  const int BLOCKSIZE = 256;
  int numblocks = (size + 255) / 256;
  gpu_add_double_vectors<<<numblocks, BLOCKSIZE>>>(size, this_dev, other_dev);
}

__global__
void gpu_multiply_matrix_by_matrix_CbyR(int M, int N, double *A_T_eth_col, double *B_eth_row, double *C) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (tidx < M * N) {
     int j = tidx % N;
     int i = tidx / N;
     C[tidx] += A_T_eth_col[i] * B_eth_row[j];
  }
}

void kernel_gpu_multiply_matrix_by_matrix_CbyR(int M, int K, int N, double *A_T, double *B, double *C) {
    const int BLOCKSIZE = 256;
    int numblocks = (M * N + 255) / 256;
    for (int e = 0; e < K; e++) {
      auto A_T_eth_col = A_T + (e * M);
      auto B_eth_row = B + (e * N);
      gpu_multiply_matrix_by_matrix_CbyR<<<numblocks, BLOCKSIZE>>>(M, N, A_T_eth_col, B_eth_row, C);
    }
}

__global__
void gpu_dot_product_vectors(int n, int size, double *this_dev, double *other_dev) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (tidx < size) {
    this_dev[tidx] = other_dev[tidx] * this_dev[tidx];
    // printf("tidx= %d, other_dev= %f this_dev= %f \n", tidx, other_dev[tidx], this_dev[tidx]);
  }
  
  while (n > 1) {
    n /= 2;
    assert(n >= 1);
    if (tidx < n) {
      double to_be_added = 0;
      if (tidx + n < size) {
	to_be_added = this_dev[tidx + n];
      }
      this_dev[tidx] += to_be_added;
    }
  }
}

void kernel_gpu_dot_product_vectors(int size, double *this_dev, double *other_dev) {
  const int BLOCKSIZE = 256;
  int numblocks = (size + 255) / 256;
  auto n = size;
  int count = 0;
  for (; n > 0; n /= 2) {
    count++;
  }
  n = 1;
  for (int i = 0; i < count; i++) {
    n *= 2;
  }
  assert(n >= size);
  assert(n / 2 < size);
  gpu_dot_product_vectors<<<numblocks, BLOCKSIZE>>>(n, size, this_dev, other_dev);
}
__global__
void gpu_matrix_multiply_RbyC(int M, int N, int Kpowered, int K, double *A, double *B_T, double* C) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ double local_vector[512];
  // int K = blockDim.x;
  assert(K <= Kpowered);
  assert(K <= 512);
  // int M passed as arg

  int k = threadIdx.x;
  assert(0 <= k);
  assert(     k <= Kpowered);
  int n = blockIdx.x % N;
  assert(0 <= n);
  assert(     n <= N);
  int m = blockIdx.x / N;
  assert(0 <= m);
  assert(     m <= M);

  __syncthreads();
  if (tidx < Kpowered * N * M && k < K) {
    local_vector[k] = A[m * K + k] * B_T[n * K + k];
    // printf("m = %d, n = %d, k = %d, A[..] = %f, B_T[../] = %f, local_vector[k] = %f \n", m, n, k, A[m * K + k], B_T[n * K + k], local_vector[k]);
  }
  
  __syncthreads();
  while (Kpowered > 1) {
      Kpowered /= 2;
      assert(Kpowered >= 1);
      auto right_idx = k + Kpowered;
      if (k < Kpowered && right_idx < K) {
          __syncthreads();
          local_vector[k] += local_vector[right_idx];
      }
    }
 
  /* if (k == 0) {
    for (int i = 1; i < K; i++) {
      local_vector[0] += local_vector[i];
    }
  }
  */
  //  __syncthreads();
  if (k == 0) {
    C[m * N + n] += local_vector[0];
  }
  // __syncthreads();
}

int uproundToPowerOfTwo(int k) {
  assert(k > 0);
  auto n = k - 1;
  int count = 0;
  
  for (; n > 0; n /= 2) {
    count++;
  }
  n = 1;
  for (int i = 0; i < count; i++) {
    n *= 2;
  }
  assert(n >= k);
  assert(n / 2 < k);
  return n;
}

void kernel_gpu_multiply_matrix_by_matrix_RbyC(int M, int K, int N, double *A, double *B_T, double *C) {
    cudaDeviceSynchronize();
    gpu_matrix_multiply_RbyC<<<M * N, uproundToPowerOfTwo(K)>>>(M, N, uproundToPowerOfTwo(K), K, A, B_T, C);
  cudaDeviceSynchronize();
}

