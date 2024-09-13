#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

// enum starting at 1 so that uninitialized is 0
typedef enum {
    LINEAR_LAYER = 1,
    RELU_LAYER,
    SOFTMAX_LAYER,
} LAYER_TYPE;

typedef struct {
    LAYER_TYPE layer_type;
    Tensor* weights;
    Tensor* bias;
    Tensor* input;  // For grad
    Tensor* output; // For grad
} Layer;
typedef struct {
    Layer* layers;
    int layer_size;
    int layer_index;
} Model;


void add_layer(Model* model, LAYER_TYPE type, int input, int output);
Model* create_model(int num_layers);
Tensor* forward(Model* model, Tensor* input);
void backwards(Model* model, Tensor* pred, Tensor* act);
void free_layer(Layer *layer);
void free_model(Model* model);


#endif // !MODEL_H
