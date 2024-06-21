
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
