
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
void gpu_multiply_matrix_by_matrix(int first_rows, int second_cols, double *transposed_matrix_on_device, double *other_on_device, double *result_on_device) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (tidx < first_rows * second_cols) {
     int j = tidx % second_cols;
     int i = tidx / second_cols;
     result_on_device[tidx] += transposed_matrix_on_device[i] * other_on_device[j];
  }
}

void kernel_gpu_multiply_matrix_by_matrix(int first_rows, int k, int second_cols, double *transposed_matrix_on_device, double *other_on_device, double *result_on_device) {
    const int BLOCKSIZE = 256;
    int numblocks = (first_rows * second_cols + 255) / 256;
    for (int i = 0; i < k; i++) {
      gpu_multiply_matrix_by_matrix<<<numblocks, BLOCKSIZE>>>(first_rows, second_cols, transposed_matrix_on_device + (i * first_rows), other_on_device + (i * second_cols), result_on_device);
    }
}
