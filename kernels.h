#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
  void kernel_gpu_add_double_vectors(int n, double *this_dev, double *other_dev);
}
#endif// KERNELS_H

