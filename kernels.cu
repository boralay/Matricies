
#include "kernels.h"




__global__
void gpu_add_double_vectors(int n, double *this_dev, double *other_dev)
{
  
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (tidx < n) {
    this_dev[tidx] = other_dev[tidx] + this_dev[tidx];
  }
}

void kernel_gpu_add_double_vectors(int n, double *this_dev, double *other_dev) {
  gpu_add_double_vectors<<<5, 5>>>(n, this_dev, other_dev);
}
