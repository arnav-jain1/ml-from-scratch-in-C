#include "../include/tensor.h"
#include "../include/model.h"
#include "../include/utils.h"
#include "../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>







int main() {
    srand(time(NULL));


    int dims = 2;
    int input_shape[] = {10,4};

    Model* model = create_model(6);
    add_layer(model, LINEAR_LAYER, 4, 100);
    add_layer(model, RELU_LAYER, 100, 100);
    add_layer(model, LINEAR_LAYER, 100, 100);
    add_layer(model, RELU_LAYER, 100, 100);
    add_layer(model, LINEAR_LAYER, 100, 4);
    add_layer(model, SOFTMAX_LAYER, 4, 4);
    int shape_act[] = {10, 1};
    Tensor* y_act = create_tensor(shape_act, 2, true);
    for (int i = 0; i < y_act->total_size; i++) {
        y_act->data[i] = 2;
    }
    int epochs = 200;
    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* input = create_tensor_rand(input_shape, dims, true);
        Tensor* pred = forward(model, input);



        double loss = cross_entropy_loss(pred, y_act);

        printf("Loss: %.9f\n", loss);

        backwards(model, pred, y_act);
        SGD_step(model, .001);
        /*zero_grad(model);*/

        free_tensor(input);
        free_tensor(pred);
    }

    free_model(model);
    free_tensor(y_act);

}
