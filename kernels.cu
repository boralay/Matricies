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
      cudaDeviceSynchronize();  
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
void gpu_matrix_multiply_RbyC(int M, int N, int K, int offset, int K_partitioned_powered, int K_partitioned, double *A, double *B_T, double* C) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ double local_vector[512];
  // int K_partitioned = blockDim.x;
  assert(K_partitioned <= K_partitioned_powered);
  assert(K_partitioned <= 512);
  // int M passed as arg

  int k = threadIdx.x;
  assert(0 <= k);
  assert(     k <= K_partitioned_powered);
  int n = blockIdx.x % N;
  assert(0 <= n);
  assert(     n <= N);
  int m = blockIdx.x / N;
  assert(0 <= m);
  assert(     m <= M);

  __syncthreads();
  if (tidx < K_partitioned_powered * N * M && k < K_partitioned) {
    local_vector[k] = A[m * K + offset + k] * B_T[n * K + offset + k];
    // printf("m= %d, n= %d, k= %d, off= %d, ii= %d, jj= %d, A[.]= %f, B_T[.]= %f, lvec[k]= %f \n", m, n, k, offset, m * K + offset + k, n * K + offset + k, A[m * K + offset + k], B_T[n * K + offset + k], local_vector[k]);
  }
  
  __syncthreads();
  while (K_partitioned_powered > 1) {
      K_partitioned_powered /= 2;
      assert(K_partitioned_powered >= 1);
      auto right_idx = k + K_partitioned_powered;
      if (k < K_partitioned_powered && right_idx < K_partitioned) {
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
  int count_parts = 1; // default = 1
  int offset = 0;
  int partitioned_K = (K > 512) ? 512 : K;
  
  while (offset < K) {
    
    gpu_matrix_multiply_RbyC<<<M * N, uproundToPowerOfTwo(partitioned_K)>>>(M, N, K, offset, uproundToPowerOfTwo(partitioned_K), partitioned_K, A, B_T, C);
    cudaDeviceSynchronize();
    //  printf("ONE CYCLE DONE \n");


    if (offset + 512 < K) {
      count_parts++;
    }
    if (count_parts * 512 > K) {
      partitioned_K = K - ((count_parts - 1) * 512);
    }
    offset += 512;
    
  }  
}


