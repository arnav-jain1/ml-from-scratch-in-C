#include "../include/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// The matmul operation which is VERY slow and I will def speed up either from scratch or just yoink BLAS, also only up to 2 dim for right now
Tensor* matmul(Tensor* t1, Tensor* t2) {

    if (t1->ndim > 2 || t2->ndim > 2) {
        fprintf(stderr, "Mat mul is currently only supports 2 dimentional matricies\n");
        exit(EXIT_FAILURE);
    }

    if (t1->shape[1] != t2->shape[0]) {
        fprintf(stderr, "Invalid dimentions for mat mul: %dx%d * %dx%d", t1->shape[0], t1->shape[1], t2->shape[0], t2->shape[1]);
        exit(EXIT_FAILURE);
    }
    
    // TODO: Can make slightly more memory effecient here
    int m = t1->shape[0];
    int n = t1->shape[1];
    int p = t2->shape[1];

    int shape[] =  {m, p};

    Tensor* res = create_tensor(shape, 2, t1->req_grad || t2->req_grad);

    //TODO: FASTER
    for (int i = 0; i < m; i++) {
         for (int j = 0; j < p; j++) {
             double sum = 0.0;
             for (int k = 0; k < n; k++) {
                 sum += t1->data[i * n + k] * t2->data[k * p + j];
             }
             res->data[i * p + j] = sum;
         }
    }
    return res;
}

// I kept messing up the deriv so I commented it heavily
void matmul_d(Tensor* t1, Tensor* t2, Tensor* prev_layer_grad) {
    int m = t1->shape[0]; // i iterates over rows of t1, prev
    int n = t1->shape[1]; // j iterates over rows of t1, cols of t2
    int p = t2->shape[1]; // k iterates over cols of t2, prev
    
    // t1.grad = Prev times t2^T
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < p; k++) {
                // row of grad times col (transposed row) of t2
                sum += prev_layer_grad->grad[i * p + k] * t2->data[j * p + k];


            }
            t1->grad[i * n + j] += sum;
        }
    }


    // t2.grad = t1^T times prev
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < p; k++) {
            double sum = 0;
            for (int i = 0; i < m; i++) {
                // col (transposed row) of t1 times row of grad 
                sum += t1->data[i * n + j] * prev_layer_grad->grad[i * p + k];
            }
            t2->grad[j * p + k] += sum;
        }
    }
}


// Adds bias, pretty simple
Tensor* add_bias(Tensor* t1, Tensor* bias) {
    if (t1->ndim > 2) {
        fprintf(stderr, "Adding bias currently only supports 2 dim tensors");
        exit(EXIT_FAILURE);
    }
    if (t1->ndim < 1 || bias->ndim != 1 || t1->shape[1] != bias->shape[0]) {
        fprintf(stderr, "Invalid shapes for adding biases: adding dim - 1: %d with bias %d \n", t1->shape[1], bias->shape[0]);
        exit(EXIT_FAILURE);
    }

    Tensor* output = create_tensor(t1->shape, t1->ndim, t1->req_grad);
    for (int batch = 0; batch < t1->shape[0]; batch++) {
        for (int i = 0; i < t1->shape[1]; i++) {
            output->data[batch * t1->shape[1] + i] = bias->data[i] + t1->data[batch * t1->shape[1] + i];
        }
    }
    return output;

}

//TODO Add SIGMOID
// ReLu Max(0, i)
Tensor* relu(Tensor* t1) {
    Tensor* res = create_tensor(t1->shape, t1->ndim, t1->req_grad);
    if (t1->total_size != res->total_size) {
        fprintf(stderr, "Incompatible sizes for relu input: %ld and output: %ld", t1->total_size, res->total_size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < t1->total_size; i++) {
         res->data[i] = t1->data[i] * (t1->data[i] > 0);
    }
    return res;
}

// deriv of relu, pretty much if the input data was more than 0, it kept it the same so mult by 1
// If the input data was < 0, it would have made it 0 so the deriv should also 0 it out
void relu_d(Tensor* input, Tensor* grad_output) {
    for (int i = 0; i < input->total_size; i++) {
        input->grad[i] = grad_output->grad[i] * (input->data[i] > 0);
    }
}

// Basic softmax, followed pytorch
Tensor* softmax(Tensor* t1) {
    Tensor* res = create_tensor(t1->shape, t1->ndim, t1->req_grad);

    int batch_size = t1->shape[0];
    int size = t1->shape[1];
    for (int b = 0; b < batch_size; b++) {

        double max = t1->data[size * b];
        for (int i = 0; i < size; i++) {
            if (t1->data[size * b + i] > max) {
                max = t1->data[size * b + i];
            }
        }
        
        double sum = 0;
        for (int i = 0; i < size; i++) {
            // int index = size * b + i;
            res->data[size * b + i] = exp(t1->data[size * b + i] - max);
            sum += res->data[size * b + i];
        }


        for (int i = 0; i < size; i++) {
            res->data[size * b + i] /= sum;
        }
    }

    return res;
}
