/**
 * In this example we implement a simple neural network for approximating
 * the XOR function.
 *
 * This example aims at understanding the basic concepts of neural networks
 * and the backpropagation algorithm.
 *
 * So, despite its triviality, it is a good starting point for learning
 * how to implement backpropagation, without being on the shoulders of
 * frameworks like PyTorch and the like.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Definition of the network parameters.
//
// The function we need to approximate (XOR) does not require a complex network,
// therefore we define a simple feed-forward neural network with just two
// neurons in the hidden layer.
#define NN_INPUT 2
#define NN_HIDDEN 2
#define NN_OUTPUT 1

// Definition of the network learning parameters.
//
// The learning rate is set to be 0.1 and it is used,
// when updating weights and biases in order to make the step smaller
// and avoid to converge too fast.
//
// The number of training epochs serves the purpose
#define LR 0.1f
#define EPOCHS 10000

// Neural network struct definition.
//
// The following definition is hardcoded since the example is trivial.
// It would require a more structured approach for more complex functions
// to approximate.
//
// In order to properly define the network we need:
// - input layer
// - hidden layer
// - output layer
//
// - weights from input to hidden layers
// - weights from hidden to output layers
//
// - biases for the hidden layer
// - biases for the output layer
//
// In classification tasks we would probably need two layers for the output:
// one for the raw logits (output values before activation function) and
// one for the actual probabilities of each class (output values after
// activation function)
typedef struct {
    float input[NN_INPUT];
    float hidden[NN_HIDDEN];
    float output[NN_OUTPUT];

    float weights_ih[NN_INPUT * NN_HIDDEN];
    float weights_ho[NN_OUTPUT * NN_HIDDEN];

    float bias_ih[NN_HIDDEN];
    float bias_ho[NN_OUTPUT];
} NeuralNetwork;

// Function that computes the sigmoid activation function
//
// The sigmoid function has the following formula sig(x) = \frac{1}{1 + e^{-x}}
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Function that computes the sigmoid function derivative.
//
// sig'(x) = sig(x) * (1 - sig(x))
//
// We suppose to receive as input the value of an already computed sigmoid
// function, therefore we just use x and not sigmoid(x).
// We do this because when computing the forward pass we do not save the values
// of a layer before activation is applied.
float sigmoid_der(float x) {
    return x * (1.0f - x);
}

// Helper to generate a random float between -0.5 and 0.5
//
// This function is used to randomly generate weights and biases values.
float random_float() {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

// This function initialize the network parameters with randomly generated
// values from the random_float function.
//
// By randomly initializing parameters we assure that each of them has
// a different starting point. This is important because, when parameters are
// all equally initialized, the learning process does not occur properly, as
// each parameter will lead to the same output and receive the same update.
void nn_init(NeuralNetwork *nn) {
    // Initialize input-to-hidden weights and biases
    for (int i = 0; i < NN_INPUT * NN_HIDDEN; i++) {
        nn->weights_ih[i] = random_float();
    }
    for (int i = 0; i < NN_HIDDEN; i++) {
        nn->bias_ih[i] = random_float();
    }

    // Initialize hidden-to-output weights and biases
    for (int i = 0; i < NN_HIDDEN * NN_OUTPUT; i++) {
        nn->weights_ho[i] = random_float();
    }
    for (int i = 0; i < NN_OUTPUT; i++) {
        nn->bias_ho[i] = random_float();
    }
}

// Function that implements the forward pass of the network
//
// Since this is a simple feed-forward network the forward is straightforward.
// for each of the two layers, input and output, we have to compute the linear
// transformation and then apply the activation function (sigmoid in this case).
//
// So, the main computations are:
//
// hidden = sigmoid(input * weights_ih + bias_ih)
// output = sigmoid(hidden * weights_ho + bias_ho)
//
void nn_forward(NeuralNetwork *nn, float *input) {
    // Here we store the value of the input neurons
    // in the neural network struct so that we can
    // re-use these values when computing the backpropagation
    for (int i = 0; i < NN_INPUT; i++) {
        nn->input[i] = input[i];
    }

    // Here we compute the value of the hidden layer neurons
    for (int j = 0; j < NN_HIDDEN; j++) {
        float sum = nn->bias_ih[j];

        for (int i = 0; i < NN_INPUT; i++) {
            sum += input[i] * nn->weights_ih[i * NN_HIDDEN + j];
        }

        nn->hidden[j] = sigmoid(sum);
    }

    // Here we compute the value of the output layer neurons.
    //
    // In our case, we could have simplified the computations,
    // since NN_OUTPUT is equal to 1 but this way the function will remain valid
    // if we change the value of NN_OUTPUT
    for (int j = 0; j < NN_OUTPUT; j++) {
        float sum = nn->bias_ho[j];

        for (int i = 0; i < NN_HIDDEN; i++) {
            sum += nn->hidden[i] * nn->weights_ho[i * NN_OUTPUT + j];
        }

        nn->output[j] = sigmoid(sum);
    }
}

// Function that computes the loss.
//
// The chosen loss is the half the squared error.
// Actually, we don't use the function but it is useful to still have it
// in order to better explain its derivative
float loss(float x, float y) {
    return 0.5f * powf(x - y, 2.0f);
}

// Function that computes the loss derivative.
//
// Since L = 1/2 * (x - y)^2, its derivative will be
// L' = (x - y)
float loss_der(float x, float y) {
    return x - y;
}

// Function that implements the backpropagation algorithm for our network.
//
// This is the most complex and important function, as it updates the network
// parameters to minimize the loss.
//
// Intuitively, the backpropagation algorithm works by moving the information
// given by the loss function to previous layers, so that they can use it
// to update themselves. Specifically, what we want to compute is the gradient
// of the loss with respect to each specific component of the network.
// Each gradient gives us information about how much the loss is affected by
// a change in a specific component of the network.
//
// For our specific network, we need to compute:
// - gradient of the loss w.r.t. hidden to output weights and biases
// - gradient of the loss w.r.t. input to hidden weights and biases
//
// To perform our computation, we will also need the gradient of the loss w.r.t
// the output and the hidden layers,
// since we use the chain rule to compute them.
void nn_backprop(NeuralNetwork *nn, float *c_out) {
    // Computation Output and hidden gradients
    float output_delta[NN_OUTPUT];
    float hidden_delta[NN_HIDDEN];

    // Here we compute the output gradient
    //
    // To compute the output gradient we use the chain rule. We, therefore,
    // multiply the gradient of the loss with respect to the activation values
    // by the gradient of the activation function with respect to the output
    // value obtained before the activation.
    // Thus, we multiply the loss derivative by the sigmoid derivative.
    //
    // dL/dOut = dL/dAct * dAct/dOut
    for (int i = 0; i < NN_OUTPUT; i++) {
        output_delta[i] = loss_der(nn->output[i], c_out[i]) * sigmoid_der(nn->output[i]);
    }

    // Here we compute the hidden gradient
    //
    // Each neuron of the hidden layer is dependant on each of the neurons of
    // the output layer (in this case just one), so to compute the gradient of
    // each neuron of the hidden layer we have to take into account
    // the contribution of each output neuron.
    // We then have to multiply by the weight,
    // that connect the hidden layer to the output layer. This last term is
    // the result of the gradient of the pre-activation with respect to
    // the hidden neurons. Finally we have to multiply by the derivative of
    // the sigmoid function for each neuron of the hidden layer.
    //
    // In summary the gradient is computed with the following formula:
    // dL/dHid = dL/dW * dW/dHid = dL/dOut * dOut/dHidAct * dHidAct/dHidPreAct
    for (int i = 0; i < NN_HIDDEN; i++) {
        hidden_delta[i] = 0.0f;

        for (int j = 0; j < NN_OUTPUT; j++) {
            hidden_delta[i] += output_delta[j] * nn->weights_ho[i * NN_OUTPUT + j];
        }

        hidden_delta[i] *= sigmoid_der(nn->hidden[i]);
    }

    // Weights and Bias and Update

    // Update of the Hidden to Output weights and biases
    //
    // The update formula is given by the Gradient Descent Algorithm
    // and it is:
    //
    // newValue = oldValue - learning_rate * gradient
    //
    // For the weights that connect the hidden layer with the output layer,
    // the gradient is computed by multiplying the output gradient with
    // the value of each of the hidden layer neuron.
    //
    // dL/dW = dL/dOut * dOut/dW
    for (int i = 0; i < NN_OUTPUT; i++) {
        for (int j = 0; j < NN_HIDDEN; j++) {
            nn->weights_ho[j * NN_OUTPUT + i] -= LR * output_delta[i] * nn->hidden[j];
        }

        // Update of the bias
        //
        // In this case the gradient computation is trivial, as the derivative
        // of the pre-activation output with respect to the bias is simply one.
        //
        // dL/dB = dL/dOut * dOut/dB = dL/dOut * 1 = dL/dOut
        nn->bias_ho[i] -= LR * output_delta[i];
    }

    // Update of the Input to Hidden weights and biases
    //
    // The same reasoning applied in the previous update is still valid.
    //
    // This case is different just in the used terms. In fact, we use
    // the hidden layer deltas and the neurons of the input layer
    for (int i = 0; i < NN_HIDDEN; i++) {
        for (int j = 0; j < NN_INPUT; j++) {
            nn->weights_ih[j * NN_HIDDEN + i] -= LR * hidden_delta[i] * nn->input[j];
        }

        nn->bias_ih[i] -= LR * hidden_delta[i];
    }
}

// Function that trains the neural network
//
// In this trivial example of the XOR function, the training function is simple
// but its structure follows the general structure of a training function for
// neural network.
//
// The pipeline is the following:
// - we feed an example to the network
// - we compute the loss and then perform backpropagation
//
// Usually the network is trained on the same dataset multiple times, called
// epochs. This way the network will be able to learn the underlying patterns
// in data.
void nn_train(NeuralNetwork *nn) {
    // XOR truth table
    float in1[] = {0.0f, 0.0f, 1.0f, 1.0f};
    float in2[] = {0.0f, 1.0f, 0.0f, 1.0f};
    float out[4][NN_OUTPUT] = {{0}, {1}, {1}, {0}};

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0; // Track loss per epoch
        for (int i = 0; i < 4; i++) {
            float input[2] = {in1[i], in2[i]};
            nn_forward(nn, input);
            total_loss += loss(nn->output[0], out[i][0]); // Add to total loss
            nn_backprop(nn, out[i]);
        }

        // Print progress every 1000 epochs
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / 4.0f);
        }
    }
}

int main(void) {
    srand(time(NULL)); // Seed the random number generator

    float in[] = {0.0f, 1.0f};
    NeuralNetwork nn;
    nn_init(&nn);

    nn_train(&nn);

    // Test the trained network
    printf("\n--- Testing Trained Network ---\n");
    float inputs[4][NN_INPUT] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float targets[4] = {0, 1, 1, 0};

    for (int i = 0; i < 4; i++) {
        nn_forward(&nn, inputs[i]);
        printf("Input: [%.0f, %.0f] -> Output: %f (Target: %.0f)\n",
               inputs[i][0], inputs[i][1], nn.output[0], targets[i]);
    }

    return 0;
}
