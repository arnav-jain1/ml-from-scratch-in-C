#ifndef UTILS_H
#define UTILS_H

#include "tensor.h"
#include "model.h"
double cross_entropy_loss(Tensor* y_pred, Tensor* y_act);
void cross_entropy_softmax_backwards(Tensor* input, Tensor* output, Tensor* actual);
void SGD_step(Model* model, double learning_rate);
void zero_grad(Model* model);

#endif // !UTILS_H

