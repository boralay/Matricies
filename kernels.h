#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
  void kernel_gpu_add_double_vectors(int size, double *this_dev, double *other_dev);
  void kernel_gpu_multiply_matrix_by_matrix(int first_rows, int k, int second_cols, double *transposed_matrix_on_device, double *other_on_device, double *result_on_device);
  
}

#endif// KERNELS_H

