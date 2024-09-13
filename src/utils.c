#include "../include/utils.h"
#include "../include/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// CEL, assumes softmaxed already 
double cross_entropy_loss(Tensor* y_pred, Tensor* y_act) {
    int batch_size = y_pred->shape[0];
    if (y_act->shape[1] != 1) {
        fprintf(stderr, "Actual must be of shape (n, 1)");
        exit(EXIT_FAILURE);
    }
    if (batch_size != y_act->total_size) {
        fprintf(stderr, "Invalid dims for cross entropy loss. Predicted: %d Acutal: %ld \n", batch_size, y_act->total_size);
        exit(EXIT_FAILURE);
    }
    int size = y_pred->shape[1];
    double loss = 0.0;

    for (int b = 0; b < batch_size; b++) {

        for (int i = 0; i < size; i++) {
            int true_class = y_act->data[b];
            double pred = y_pred->data[b * size + true_class];
            // Make sure to not log of 0, otherwise KABLOOM 
            loss -= log(fmax(pred, 1e-7));
        }
    }

    // This is to make sure the loss wasn't 0 because if it was then the afformentioned kabloom occured, not needed anymore but is a relic of my pain
    /*if (loss < 0) {*/
        /*printf("Loss: %.10f\n", loss);*/
        /*print_tensor(y_pred);*/
        /*print_tensor(y_act);*/
        /*exit(EXIT_FAILURE);*/
    /*}*/
    return loss / batch_size;;
}

// deriv of cross entropy loss and softmax because it is MCUH less computationaly intensive
void cross_entropy_softmax_backwards(Tensor* input, Tensor* output, Tensor* actual) {
    int batch_size = input->shape[0];
    int size = input->shape[1];

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            // If it predicted right, subtract one, otherwise keep it the same. This intuitively checks out, also normalize for batch size
            double targ = (i == (int) actual->data[b]) ? 1.0 : 0.0;
            input->grad[b * size + i] = (output->data[b * size + i] - targ) / batch_size;
        }
    }
}



//TODO Adam optim?!?!
// Pretty simple just the classic SGD, changes the all linear layers
void SGD_step(Model* model, double learning_rate) {
    for (int i = 0; i < model->layer_size; i++) {
        Layer* layer = &model->layers[i];
        if (layer->layer_type == LINEAR_LAYER) {
            for (int j = 0; j < layer->weights->total_size; j++) {
                layer->weights->data[j] -= layer->weights->grad[j] * learning_rate;
            }


            for (int j = 0; j < layer->bias->total_size; j++) {
                layer->bias->data[j] -= layer->bias->grad[j] * learning_rate;
                
            }
        }
    }

}

// Zeros grad using memset, though I have heard it is unsafe
void zero_grad(Model* model) {
    for (int i = 0; i < model->layer_size; i++) {
        Layer* layer = &model->layers[i];
        if (layer->weights) {
            memset(layer->weights->grad, 0.0, layer->weights->total_size * sizeof(double));
        }
        if (layer->bias) {
            memset(layer->bias->grad, 0.0, layer->bias->total_size * sizeof(double));
        }
    }
}
