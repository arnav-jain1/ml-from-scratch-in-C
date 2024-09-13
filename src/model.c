#include "../include/model.h"
#include "../include/operations.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>



// Adds layer to the model, makes sure the model is big enough first
void add_layer(Model* model, LAYER_TYPE type, int input, int output) {
    if (model->layer_index >= model->layer_size) { 
        fprintf(stderr, "Trying to put a layer at index more than layer size: index %d, size %d\n", model->layer_index, model->layer_size); 
        exit(EXIT_FAILURE);
    }


    model->layers[model->layer_index].layer_type = type; 

    if (type == LINEAR_LAYER) {
        int shape[] = {input, output};
        model->layers[model->layer_index].weights = create_tensor_rand(shape, 2, true);
        int bias_shape[] = {output};
        model->layers[model->layer_index].bias = create_tensor(bias_shape, 1, true);
    } else {
        model->layers[model->layer_index].weights = NULL;
        model->layers[model->layer_index].bias = NULL;
    }

    model->layers[model->layer_index].output = NULL;
    model->layers[model->layer_index].input = NULL;

    model->layer_index++;
}

// Creates a modle, pretty self explanitory
Model* create_model(int num_layers) {
    Model* model = (Model *) malloc(sizeof(Model));
    if (model == NULL) {
        fprintf(stderr, "Unable to allocate enough memory when creating a model");
        exit(EXIT_FAILURE);
    }
    model->layer_index = 0;
    model->layers = (Layer *) calloc(num_layers, sizeof(Layer));
    if (model->layers == NULL) {
        fprintf(stderr, "Unable to allocate enough memory when creating initializing %d layers\n", num_layers);
        exit(EXIT_FAILURE);
    }
    model->layer_size = num_layers;

    return model;
}


Tensor* forward(Model* model, Tensor* input) {
    Tensor* x = input;

    for (int i = 0; i < model->layer_size; i++) {
        Layer* cur_layer = &model->layers[i];
        Tensor* next;

        // If the current layer already has an input, free it and create a new tensor with the values of the input
        if (cur_layer->input) {
            free_tensor(cur_layer->input);
        }
        cur_layer->input = create_tensor(x->shape, x->ndim, true);
        for (int j = 0; j < x->shape[0] * x->shape[1]; j++) {
            cur_layer->input->data[j] = x->data[j];
        }       
        // Depending on the layer type, do the operation and save it to next
        switch (cur_layer->layer_type) {
            case LINEAR_LAYER:
                next = matmul(x, cur_layer->weights);
                for (int j = 0; j < next->shape[1]; j++) {
                    next->data[j] += cur_layer->bias->data[j];
                }

                break;
            case RELU_LAYER:
                next = relu(x);
                break;
            case SOFTMAX_LAYER:
                next = softmax(x);
                break;
            default:
                // Shouldn't reach here but just in case
                fprintf(stderr,"INVALID INPUT FOR LAYER DURING FPASS %d", cur_layer->layer_type);
                exit(EXIT_FAILURE);
                break;
        }


        // set the output of the current layer to next (output of this layer) and set the input of the next layer to "next"
        cur_layer->output = next;
        // Also, free input to next layer if it isn't the input to the model and return the final output
        if (i > 0) {
            free_tensor(x);
        }
        x = next;
    }

    return x;
}

void backwards(Model* model, Tensor* pred, Tensor* act) {
    int last_layer = model->layer_size -1;

    // Calculate the grad of the loss where the input is the input of the last layer, pred, expected

    cross_entropy_softmax_backwards(pred, pred, act);

    // Current gradient to mult with is the input (.grad)
    Tensor* cur_grad = pred;
    for (int i = last_layer; i >= 0; i--) {
        // Current layer is the last layer, it is also what will be used for the backprop
        Layer* cur_layer = &model->layers[i];

        switch (cur_layer->layer_type) {
            // Skip softmax already done
            case SOFTMAX_LAYER:
                break;
            case LINEAR_LAYER:
                // input the current layer input and weights, as well as the cur grad to matmul
                matmul_d(cur_layer->input, cur_layer->weights, cur_grad);

                // Update bias weights by sum
                for (int j = 0; j < cur_layer->bias->total_size; j++) {
                    double sum = 0.0;
                    for (int batch = 0; batch < cur_grad->shape[0]; batch++) {
                        sum += cur_grad->grad[batch * cur_layer->bias->total_size + j];
                    }
                    cur_layer->bias->grad[j] += sum;
                }


                // Current grad is the input (.grad)
                cur_grad = cur_layer->input;
                break;
            case RELU_LAYER:
                // Relu is pretty self explainitory 
                relu_d(cur_layer->input, cur_grad);
                cur_grad = cur_layer->input;
                break;
            default:
                fprintf(stderr, "INVALID INPUT FOR BACKWARDS: %d", cur_layer->layer_type);
                exit(EXIT_FAILURE);

        }


    }
}

// Free all the layers if they exist which is important cuz segfault
void free_layer(Layer *layer) {
    if (layer->bias) {
        free_tensor(layer->bias);
    }
    if (layer->input) {
        free_tensor(layer->input);
    }
    if (layer->weights) {
        free_tensor(layer->weights);
    }
}


void free_model(Model* model) {

    for (int layer = 0; layer < model->layer_size; layer++) {
        free_layer(&model->layers[layer]);
    }
    free(model->layers);
    free(model);
}
