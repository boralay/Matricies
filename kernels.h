#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
  typedef void (*func_idpdp)(int size, double *this_dev, double *other_dev);
  void kernel_gpu_add_double_vectors(int size, double *this_dev, double *other_dev);
  void kernel_gpu_multiply_matrix_by_matrix_CbyR(int first_rows, int k, int second_cols, double* other_on_device, double *transposed_matrix_on_device, double *result_on_device);
  void kernel_gpu_multiply_matrix_by_matrix_RbyC(int M, int K, int N, double *A, double *B_T, double *C);
  void kernel_gpu_dot_product_vectors(int size, double *this_dev, double *other_dev);
}

#endif// KERNELS_H

