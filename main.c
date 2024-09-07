#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
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

// Never used enum so comments for myself:
// Essentially like a macro to make the code easier to read, to be used in a switch 
// Sequential after declared (else starts with 0)
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

void matmul_d(Tensor* t1, Tensor* t2, Tensor* prev_layer_grad) {
    int m = t1->shape[0]; // i iterates over rows of t1, prev
    int n = t1->shape[1]; // j iterates over rows of t1, cols of t2
    int p = t2->shape[1]; // k iterates over cols of t2, prev
    
    // Prev times t2^T
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


    // t1^T times prev
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

void relu_d(Tensor* input, Tensor* grad_output) {
    for (int i = 0; i < input->total_size; i++) {
        input->grad[i] = grad_output->grad[i] * (input->data[i] > 0);
    }
}

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

double cross_entropy_loss(Tensor* y_pred, Tensor* y_act) {
    int batch_size = y_pred->shape[0];
    int size = y_pred->shape[1];
    double loss = 0.0;

    for (int b = 0; b < batch_size; b++) {

        for (int i = 0; i < size; i++) {
            int true_class = y_act->data[b];
            double pred = y_pred->data[b * size + true_class];
            loss -= log(fmax(pred, 1e-7));
        }
    }

    if (loss < 0) {
        printf("Loss: %.10f\n", loss);
        print_tensor(y_pred);
        print_tensor(y_act);
        exit(EXIT_FAILURE);
    }
    return loss / batch_size;;
}

void cross_entropy_softmax_backwards(Tensor* input, Tensor* output, Tensor* actual) {
    int batch_size = input->shape[0];
    int size = input->shape[1];

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            double targ = (i == (int) actual->data[b]) ? 1.0 : 0.0;
            input->grad[b * size + i] = (output->data[b * size + i] - targ) / batch_size;
        }
    }
}


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

Tensor* forward(Model* model, Tensor* input) {
    Tensor* x = input;

    for (int i = 0; i < model->layer_size; i++) {
        Layer* cur_layer = &model->layers[i];
        Tensor* next;

        if (cur_layer->input) free_tensor(cur_layer->input);
        cur_layer->input = create_tensor(x->shape, x->ndim, true);
        for (int j = 0; j < x->shape[0] * x->shape[1]; j++) {
            cur_layer->input->data[j] = x->data[j];
        }       
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
                fprintf(stderr,"INVALID INPUT FOR LAYER DURING FPASS %d", cur_layer->layer_type);
                exit(EXIT_FAILURE);
                break;
        }


        cur_layer->output = next;
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
        // Current layer is the last layer
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
                relu_d(cur_layer->input, cur_grad);
                cur_grad = cur_layer->input;
                break;
            default:
                fprintf(stderr, "INVALID INPUT FOR BACKWARDS: %d", cur_layer->layer_type);
                exit(EXIT_FAILURE);

        }


    }
}

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
