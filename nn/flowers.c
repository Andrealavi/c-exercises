/**
 * In this example we implement a simple neural network for classifying
 * flowers based on the length and the width of the sepals and petals.
 * Data are obtained from the famous Iris dataset.
 *
 * This example aims at understanding the basic concepts of neural networks,
 * just as the XOR example in xor.c. The main difference this time is due to
 * network complexity (we have two hidden layers instead of one) and
 * the usage of an external dataset.
 *
 * To compile the code use the following command:
 * clang -O3 -o flowers ../csv_parser/csv_parser.c flowers.c
 */

#include "../csv_parser/csv_parser.h"

#include <time.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Definition of network parameters.
//
// This time we consider a feed-forward network with
// two hidden layers. Probably a single hidden layer]
// with more neurons would have done the job, as well,
// but this way we can see how the code changes when multiple
// hidden layers are used (spoiler: not much).
#define NN_INPUT 4
#define NN_HIDDEN_1 8
#define NN_HIDDEN_2 4
#define NN_OUTPUT 3

// Definition of training parameters.
//
// In this example the learning rate is smaller than the one
// used in the XOR example.
// A learning rate of 0.1 was too big and made the loss
// go stale in a bad local minimum.
#define LR 0.01f
#define EPOCHS 5000

// Definition of the network struct.
//
// As in the XOR example the definition is hard coded in the struct
// to make things simpler.
// The only difference with the XOR example is that we have added
// a second hidden layer, with the corresponding weights and biases
// and an array for storing raw logits, before applying softmax
// activation function.
typedef struct {
    float input[NN_INPUT];
    float f_hidden[NN_HIDDEN_1];
    float s_hidden[NN_HIDDEN_2];
    float raw_logits[NN_OUTPUT];
    float output[NN_OUTPUT];

    float weights_ih[NN_INPUT * NN_HIDDEN_1];
    float bias_ih[NN_HIDDEN_1];
    float weights_hh[NN_HIDDEN_1 * NN_HIDDEN_2];
    float bias_hh[NN_HIDDEN_2];
    float weights_ho[NN_HIDDEN_2 * NN_OUTPUT];
    float bias_ho[NN_OUTPUT];
} NeuralNetwork;

// Helper to generate a random float between -0.5 and 0.5
#define RAND_FLOAT() (( (float)rand() / (float)RAND_MAX ) - 0.5f)

// Initializes the network.
//
// It initializes the network with random weights and biases values that
// go from -0.5 to 0.5.
// This way we avoid problems in the training process as each parameter
// will have a different random value.
void nn_init(NeuralNetwork *nn) {
    for (int i = 0; i < NN_INPUT * NN_HIDDEN_1; i++) nn->weights_ih[i] = RAND_FLOAT();
    for (int i = 0; i < NN_HIDDEN_1; i++) nn->bias_ih[i] = RAND_FLOAT();
    for (int i = 0; i < NN_HIDDEN_1 * NN_HIDDEN_2; i++) nn->weights_hh[i] = RAND_FLOAT();
    for (int i = 0; i < NN_HIDDEN_2; i++) nn->bias_hh[i] = RAND_FLOAT();
    for (int i = 0; i < NN_HIDDEN_2 * NN_OUTPUT; i++) nn->weights_ho[i] = RAND_FLOAT();
    for (int i = 0; i < NN_OUTPUT; i++) nn->bias_ho[i] = RAND_FLOAT();
}

// ReLU activation function implementation.
//
// The ReLU activation function simply returns the value
// if it is greater than 0.
//
// There are several variants of the ReLU function
// where the threshold value is changed.
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ReLU derivative.
//
// The derivative of the ReLU function is straightforward.
// It is 1 if the value of x is greater than 0; it is 0 otherwise.
float relu_der(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Implementation of the softmax activation function.
//
// The softmax function is used to convert the output of the neural
// network into probabilities and it is used especially for classification
// tasks where the number of output neurons is more than 1.
//
// It has the following formula: softmax(x) = e^x / (sum_{i=0}^n e^{x_i})
// where x_i represent a logit value from the output layer.
//
// This implementation takes into account the numerical instability of
// the softmax function, therefore it uses e^{x_i - max}, where max
// is the greatest logit among all of the logits in the output layer.
void softmax(float *input, float *output, const int size) {
    // Finds max for numerical stability.
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Computes the summation in the denominator.
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    if (sum > 0) {
        // Completes the formula by diving for the sum.
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    } else {
        // If the result is numerically unstable,
        // we just consider a uniform distribution.
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / size;
        }
    }
}

// Forward pass of the network.
//
// Since the network is a simple feed-forward net,
// we just compute Y = x * W + b for each layer and then apply
// the activation function.
//
// We use ReLU activation for all layers with the exception of the output
// where we use softmax activation.
void nn_forward(NeuralNetwork *nn, float* input) {
    // Copy input as nn->input is used in the backward pass.
    memcpy(nn->input, input, sizeof(float) * NN_INPUT);

    // Computation of the first hidden layer values.
    for (int i = 0; i < NN_HIDDEN_1; i++) {
        float sum = nn->bias_ih[i];
        for (int j = 0; j < NN_INPUT; j++) {
            sum += input[j] * nn->weights_ih[j * NN_HIDDEN_1 + i];
        }

        nn->f_hidden[i] = relu(sum);
    }

    // Computation of the second hidden layer values.
    for (int i = 0; i < NN_HIDDEN_2; i++) {
        float sum = nn->bias_hh[i];
        for (int j = 0; j < NN_HIDDEN_1; j++) {
            sum += nn->f_hidden[j] * nn->weights_hh[j * NN_HIDDEN_2 + i];
        }

        nn->s_hidden[i] = relu(sum);
    }

    // Computation of the output layer values.
    for (int i = 0; i < NN_OUTPUT; i++) {
        nn->raw_logits[i] = nn->bias_ho[i];
        for (int j = 0; j < NN_HIDDEN_2; j++) {
            nn->raw_logits[i] += nn->s_hidden[j] * nn->weights_ho[j * NN_OUTPUT + i];
        }
    }

    softmax(nn->raw_logits, nn->output, NN_OUTPUT);
}


void nn_backward(NeuralNetwork *nn, float *input, float *target, float lr) {
    nn_forward(nn, input);

    // Output deltas
    float output_deltas[NN_OUTPUT];
    for (int i = 0; i < NN_OUTPUT; i++) {
        output_deltas[i] = nn->output[i] - target[i];
    }

    //hidden 2 deltas
    float hidden_2_deltas[NN_HIDDEN_2];
    for (int i = 0; i < NN_HIDDEN_2; i++) {
        hidden_2_deltas[i] = 0.0f;
        for (int j = 0; j < NN_OUTPUT; j++) {
            hidden_2_deltas[i] += output_deltas[j] * nn->weights_ho[i * NN_OUTPUT + j];
        }

        hidden_2_deltas[i] *= relu_der(nn->s_hidden[i]);;
    }

    // hidden 1 deltas
    float hidden_1_deltas[NN_HIDDEN_1];
    for (int i = 0; i < NN_HIDDEN_1; i++) {
        hidden_1_deltas[i] = 0.0f;
        for (int j = 0; j < NN_HIDDEN_2; j++) {
            hidden_1_deltas[i] += hidden_2_deltas[j] * nn->weights_hh[i * NN_HIDDEN_2 + j];
        }

        hidden_1_deltas[i] *= relu_der(nn->f_hidden[i]);
    }

    // weights and biases update
    for (int i = 0; i < NN_OUTPUT; i++) {
        for (int j = 0; j < NN_HIDDEN_2; j++) {
            nn->weights_ho[j * NN_OUTPUT + i] -= LR * output_deltas[i] * nn->s_hidden[j];
        }

        nn->bias_ho[i] -= LR * output_deltas[i];
    }

    for (int i = 0; i < NN_HIDDEN_2; i++) {
        for (int j = 0; j < NN_HIDDEN_1; j++) {
            nn->weights_hh[j * NN_HIDDEN_2 + i] -= LR * hidden_2_deltas[i] * nn->f_hidden[j];
        }

        nn->bias_hh[i] -= LR * hidden_2_deltas[i];
    }

    for (int i = 0; i < NN_HIDDEN_1; i++) {
        for (int j = 0; j < NN_INPUT; j++) {
            nn->weights_ih[j * NN_HIDDEN_1 + i] -= LR * hidden_1_deltas[i] * nn->input[j];
        }

        nn->bias_ih[i] -= LR * hidden_1_deltas[i];
    }
}

// Function used to encode the last column of the dataset into
// a valid output for the neural network.
//
// We use one-hot encoding for this purpose.
// One-hot encoding simply allocates n numbers, one for each class
// and places a 1 at the index that corresponds to the class
// expressed in our string.
//
// This implementation is probably inefficient since we could have used
// just three bit instead of an array of floats, but it simple to understand
// and to use in our network.
void encode(char *species, float *encoding) {
    encoding[0] = 0.0f;
    encoding[1] = 0.0f;
    encoding[2] = 0.0f;

    if (strcmp(species, "Iris-setosa") == 0) encoding[0] = 1.0f;
    else if (strcmp(species, "Iris-versicolor") == 0) encoding[1] = 1.0f;
    else if (strcmp(species, "Iris-virginica") == 0) encoding[2] = 1.0f;
}

// This function decode the output of the network.
// It looks for the max in the output layer and
// returns the string correspondent to the species related to the max index.
char *decode(float *output) {
    float max = output[0];
    int max_idx = 0;

    for (int i = 1; i < NN_OUTPUT; i++) {
        if (output[i] > max) {
            max = output[i];
            max_idx = i;
        }
    }

    if (max_idx == 0) return "Setosa";
    else if (max_idx == 1) return "Versicolor";
    else if (max_idx == 2) return "Virginica";

    return "Unknown";
}

// Computes the loss of the network using cross entropy loss.
//
// Cross entropy is often used when we use a softmax as activation function.
//
// It is computed with the following formula E[-log(q)],
// where E is the expected value and q is the result of the softmax function
// In our case the formula is much simpler, because the probabilities will be
// influenced by the actual outcome and will be all zeroes with the exception
// of the value related to the expected output class.
//
// E.g. if the flower is a Setosa, target will be {1, 0, 0}, therefore
// the other terms of the sum can be avoided.
//
// In this case, I decided to implement it with the real formula in order
// to propose the complete implementation. Since our example is trivial,
// it won't affect the performance.
float loss(float *output, float *target) {
    float sum = 0.0f;

    for (int i = 0; i < NN_OUTPUT; i++) {
        sum -= target[i] * logf(output[i]);
    }

    return sum;
}

// Perform network training.
//
// We follow the typical learning process by iterating over the dataset
// for a number of epochs given by EPOCHS (5000 in our case).
// For each example we perform the forward pass and then adjust the parameters
// using the backpropagation algorithm.
void nn_train(NeuralNetwork *nn, CsvParser *parser) {
    for (int k = 0; k < EPOCHS; k++) {
        float total_loss = 0.0f;

        for (int i = 0; i < parser->num_rows; i++) {
            float input[NN_INPUT];
            for (int j = 0; j < NN_INPUT; j++) {
                input[j] = atof(parser->data[i][j]);
            }

            nn_forward(nn, input);

            float target[NN_OUTPUT];
            encode(parser->data[i][NN_INPUT], target);


            nn_backward(nn, input, target, LR);

            total_loss += loss(nn->output, target);
        }

        if (k % 1000 == 0) {
            printf("Epoch %d/%d --> Loss: %f\n", k, EPOCHS, total_loss / parser->num_rows);
        }
    }
}

// Here we create and train the network using the iris dataset.
// Then we test the results to check if the training went good.
//
// We do not split the dataset into training and testing as our purpose is
// just to create a functioning network.
// We are interested in seeing how the loss decreases over time
// and that the network is able to correctly classify examples of the dataset.
int main(void) {
    srand(time(NULL));

    NeuralNetwork nn;
    nn_init(&nn);

    // Here you have to put your dataset file
    // The iris dataset can be found at https://archive.ics.uci.edu/dataset/53/iris
    CsvParser *dataset = csv_parser_create("iris.data.csv", ',', 0);
    csv_parser_parse(dataset);

    nn_train(&nn, dataset);

    printf("Testing network\n\n");
    for (int i = 0; i <  dataset->num_rows; i++) {
        float input[NN_INPUT];

        for (int j = 0; j < NN_INPUT; j++) {
            input[j] = atof(dataset->data[i][j]);
        }

        nn_forward(&nn, input);
        char *out = decode(nn.output);

        printf("Computed: %s\tActual: %s\n", out, dataset->data[i][NN_INPUT]);
    }

    return 0;
}
