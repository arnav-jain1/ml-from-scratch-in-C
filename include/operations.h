#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "tensor.h"


Tensor* matmul(Tensor* t1, Tensor* t2);
void matmul_d(Tensor* t1, Tensor* t2, Tensor* prev_layer_grad);
Tensor* add_bias(Tensor* t1, Tensor* bias);
Tensor* relu(Tensor* t1);
void relu_d(Tensor* input, Tensor* grad_output);
Tensor* softmax(Tensor* t1);

#endif //  !TENSOR_H
