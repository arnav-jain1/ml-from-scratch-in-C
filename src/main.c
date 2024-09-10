#include "../include/tensor.h"
#include "../include/model.h"
#include "../include/utils.h"
#include "../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH_SIZE 16

typedef struct {
    int count;
    Tensor* inputs;
    Tensor* actual;
} Dataset;

Dataset* create_dataset(int num_datapoints, int size_per_point) {
    Dataset* dataset = malloc(sizeof(Dataset));
    if (dataset == NULL) {
        fprintf(stderr, "Failed to allocate memory for dataset\n");
        return NULL;
    }

    dataset->count = num_datapoints;
    dataset->inputs = malloc(sizeof(Tensor));
    if (!dataset->inputs) {
        fprintf(stderr, "Failed to allocate memory for inputs tensor\n");
        free(dataset);
        return NULL;
    }

    int shape[] = {num_datapoints, size_per_point};
    dataset->inputs = create_tensor(shape, 2, false);
    shape[0] = 1;
    shape[1] = num_datapoints;
    dataset->actual = create_tensor(shape, 2, false);

    return dataset;
}
void free_dataset(Dataset* dataset) {
    free_tensor(dataset->inputs);
    free_tensor(dataset->actual);
    free(dataset);
}
void MNIST_dataset(const char* filename, Dataset* dataset) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        free_dataset(dataset);
        exit(EXIT_FAILURE);
    }


    char line[8192];
    int datapoint_index = 0;
    
    while (fgets(line, sizeof(line), file) && datapoint_index < dataset->count) {
        char* input = strtok(line, ";");
        char* actual = strtok(NULL, "\n");

        char* token = strtok(input, ",");
        for (int i = 0; i < 784 && token != NULL; i++) {
            dataset->inputs->data[datapoint_index * 784 + i] = atof(token);
            token = strtok(NULL, ",");
        }

        dataset->actual->data[datapoint_index] = (double) (atoi(actual));

        datapoint_index++;
        
    }
    



    fclose(file);
}
/*int main() {*/
    /*srand(time(NULL));*/

    /*Dataset* dataset = create_dataset(86184, 784);*/
    /*MNIST_dataset("data/train_dataset.txt", dataset);*/



    /*int dims = 2;*/
    /*int input_shape[] = {BATCH_SIZE,784};*/

    /*Model* model = create_model(6);*/
    /*add_layer(model, LINEAR_LAYER, 784, 500);*/
    /*add_layer(model, RELU_LAYER, 500, 500);*/
    /*add_layer(model, LINEAR_LAYER, 500, 100);*/
    /*add_layer(model, RELU_LAYER, 100, 100);*/
    /*add_layer(model, LINEAR_LAYER, 100, 10);*/
    /*add_layer(model, SOFTMAX_LAYER, 10, 10);*/
    /*int shape_act[] = {BATCH_SIZE, 1};*/


    /*int epochs = 200;*/
    /*for (int epoch = 0; epoch < epochs; epoch++) {*/
        /*Tensor* input = create_tensor_rand(input_shape, dims, false);*/
        /*Tensor* pred = forward(model, input);*/



        /*double loss = cross_entropy_loss(pred, y_act);*/

        /*printf("Loss: %.9f\n", loss);*/

        /*backwards(model, pred, y_act);*/
        /*SGD_step(model, .001);*/
        /*zero_grad(model);*/

        /*free_tensor(input);*/
        /*free_tensor(pred);*/
    /*}*/

    /*free_model(model);*/

    /*free_dataset(dataset);*/

/*}*/

int main() {
    srand(time(NULL));
    
    // Load dataset
    Dataset* dataset = create_dataset(86184, 784);
    MNIST_dataset("data/train_dataset.txt", dataset);

    // Create model
    Model* model = create_model(6);
    add_layer(model, LINEAR_LAYER, 784, 500);
    add_layer(model, RELU_LAYER, 500, 500);
    add_layer(model, LINEAR_LAYER, 500, 100);
    add_layer(model, RELU_LAYER, 100, 100);
    add_layer(model, LINEAR_LAYER, 100, 10);
    add_layer(model, SOFTMAX_LAYER, 10, 10);

    int epochs = 2;
    int num_batches = dataset->count / BATCH_SIZE;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int batch = 0; batch < num_batches; batch++) {
            // Create input tensor for the batch
            int input_shape[] = {BATCH_SIZE, 784};
            Tensor* input = create_tensor(input_shape, 2, false);
            
            // Create actual output tensor for the batch
            int actual_shape[] = {BATCH_SIZE, 1};
            Tensor* y_act = create_tensor(actual_shape, 2, false);

            // Fill input and y_act tensors with data from the dataset
            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = batch * BATCH_SIZE + i;
                for (int j = 0; j < 784; j++) {
                    input->data[i * 784 + j] = dataset->inputs->data[idx * 784 + j];
                }
                y_act->data[i] = dataset->actual->data[idx];
            }

            // Forward pass
            Tensor* pred = forward(model, input);

            // Compute loss
            double loss = cross_entropy_loss(pred, y_act);
            total_loss += loss;

            // Backward pass
            backwards(model, pred, y_act);

            // Update weights
            SGD_step(model, 0.01);

            // Zero gradients
            zero_grad(model);

            // Free temporary tensors
            free_tensor(input);
            free_tensor(y_act);
            free_tensor(pred);
        }

        // Print average loss for the epoch
        printf("Epoch %d, Average Loss: %.9f\n", epoch + 1, total_loss / num_batches);
    }

    // Free model and dataset
    free_model(model);
    free_dataset(dataset);

    return 0;
}
