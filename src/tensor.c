#include "../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>


// prints
void print_tensor(Tensor* tensor) {
    printf("Tensor shape: (");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf(")\n");

    printf("Data: ");
    if (tensor->ndim == 1) {
        printf("[ ");
        for (int i = 0; i < tensor->total_size; i++) {
            printf("%.4f ", tensor->data[i]);
        }
        printf("]");
    } else if (tensor->ndim == 2) {
        printf("[ \n");
        for (int i = 0; i < tensor->shape[0]; i++) {
            printf("[ ");
            for (int j = 0; j <tensor->shape[1]; j++) {
                printf("%.10f ", tensor->data[i * tensor->shape[1] + j]);
            }
            printf("]\n");
        }
        printf("]");
    } else {
        printf("[ ");
        for (int i =0; i < tensor->total_size; i++) {
            printf("%.4f ", tensor->data[i]);
        }
        printf("]");
    }

    printf("\n");
}

// makes a tensor with initialized to 0
Tensor* create_tensor(const int* shape, int ndim, bool req_grad) {

    if (ndim < 0) {
        fprintf(stderr, "Failed to create tensor, dim is less than 0: %d\n", ndim);

    }

    Tensor* tensor = (Tensor *) malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed to allocate space for tensor\n");
        exit(EXIT_FAILURE);
    }
    tensor->shape = (int*) malloc(sizeof(int) * ndim);
    if (tensor->shape == NULL) {
        fprintf(stderr, "Failed to allocate space for shape of tensor\n");
        exit(EXIT_FAILURE);
    }
    tensor->ndim = ndim;
    
    unsigned int total_size = 1;
    for (int i = 0; i < ndim; i ++) {
        tensor->shape[i] = shape[i];
        total_size *= shape[i];
    }
    tensor->total_size = total_size;

    tensor->data = (double *) calloc(total_size, sizeof(double));
    if (tensor->data == NULL) {
        fprintf(stderr, "Failed to allocate space for data of tensor\n");
        exit(EXIT_FAILURE);
    }

    tensor->req_grad = req_grad;
    if (req_grad) {
        tensor->grad = (double *) calloc(total_size, sizeof(double));
        if (tensor->grad == NULL) {
            fprintf(stderr, "Failed to allocate space for grad of tensor\n");
            exit(EXIT_FAILURE);
        }
    }


    return tensor;
}
// Tensor values randomized
Tensor* create_tensor_rand(const int* shape, int ndim, bool req_grad) {

    if (ndim < 0) {
        fprintf(stderr, "Failed to create tensor, dim is less than 0: %d\n", ndim);

    }

    Tensor* tensor = (Tensor *) malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed to allocate space for tensor\n");
        exit(EXIT_FAILURE);
    }
    tensor->shape = (int*) malloc(sizeof(int) * ndim);
    if (tensor->shape == NULL) {
        fprintf(stderr, "Failed to allocate space for shape of tensor\n");
        exit(EXIT_FAILURE);
    }
    tensor->ndim = ndim;
    
    unsigned int total_size = 1;
    for (int i = 0; i < ndim; i ++) {
        tensor->shape[i] = shape[i];
        total_size *= shape[i];
    }
    tensor->total_size = total_size;

    tensor->data = (double *) malloc(total_size * sizeof(double));
    for (int i = 0; i < total_size; i++) {
            tensor->data[i] = MIN_RAND + (rand() / (double)RAND_MAX) * (MAX_RAND - MIN_RAND);
    }
    if (tensor->data == NULL) {
        fprintf(stderr, "Failed to allocate space for data of tensor\n");
        exit(EXIT_FAILURE);
    }

    tensor->req_grad = req_grad;
    if (req_grad) {
        tensor->grad = (double *) calloc(total_size, sizeof(double));
        if (tensor->grad == NULL) {
            fprintf(stderr, "Failed to allocate space for grad of tensor\n");
            exit(EXIT_FAILURE);
        }
    }


    return tensor;
}


void free_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor->shape);
    if (tensor->req_grad) {
        free(tensor->grad);
    }
    free(tensor);
}
