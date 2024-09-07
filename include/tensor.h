#ifndef TENSOR_H
#define TENSOR_H


#include <stdbool.h>
#define MIN_RAND -1
#define MAX_RAND 1

typedef struct {
    double* data;
    int* shape;
    int ndim;
    long int total_size;
    double* grad;
    bool req_grad;
} Tensor;



void print_tensor(Tensor* tensor);

Tensor* create_tensor(const int* shape, int ndim, bool req_grad);
Tensor* create_tensor_rand(const int* shape, int ndim, bool req_grad);
void free_tensor(Tensor* tensor);


#endif // !TENSOR_H
