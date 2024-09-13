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
// shhh between the one person that might see this and me, 
// I still really don't get the point of header files but don't tell anyone
void free_tensor(Tensor* tensor);


#endif // !TENSOR_H
