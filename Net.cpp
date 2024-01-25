#include "net.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace std;

typedef struct Connection Connection;
typedef struct Neuron Neuron;

struct Connection {
    float weight;
    float delta;
};

struct Neuron {
    Connection* inputConnection;
    float output;
};

struct NeuralNetwork_ {
    int training;
    int loadedInput;
    int loadedEpectedOutput;
    float input[INPUT_NEURON];
    float expectedOutput[OUTPUT_NEURON];

    Neuron* layers[HIDDEN_LAYER + 1];
};


//------Private------//
float activationFunction(const float x) {
    assert(!isnan(x));
    // if (x <= -15.0f) return 0;
    // if (x >= 15.0f) return 1;
    return 1.0f / (1.0f + expf(-x));
}

float derivateActivationFunction(const float x) {
    assert(!isnan(x));
    // if (x == 0) return -15;
    // if (x == 1) return 15;
    return x * (1.0f - x);
}

// float calcError(NeuralNetwork nn) {
//     float error = 0.0f;
//     const Neuron* outputLayer = nn->layers[HIDDEN_LAYER];
//     for (unsigned i = 0; i < OUTPUT_NEURON; i++) {
//         const float diff = nn->expectedOutput[i] - outputLayer[i].output;
//         error += 0.5f * (diff * diff);
//     }
//     return error;
// }

void backPropagate(NeuralNetwork nn) {
    if (nn == nullptr) return;
    unsigned sizeOldLayer = 0;
    for (int i = HIDDEN_LAYER; i >= 0; i--) {
        const unsigned sizeConnections = 0 == i ? INPUT_NEURON : HIDDEN_LAYER_NEURONS;
        const unsigned sizeLayer = HIDDEN_LAYER == i ? OUTPUT_NEURON : HIDDEN_LAYER_NEURONS;

        for (unsigned y = 0; y < sizeLayer; y++) {
            const Neuron* n = &nn->layers[i][y];
            for (unsigned z = 0; z < sizeConnections; z++) {
                Connection* c = &n->inputConnection[z];

                if (i == HIDDEN_LAYER) {
                    c->delta = n->output - nn->expectedOutput[y];
                } else {
                    c->delta = 0.0f;
                    for (unsigned x = 0; x < sizeOldLayer; x++) {
                        const Connection* conn = &nn->layers[i + 1][x].inputConnection[y];
                        c->delta += conn->delta * conn->weight;
                    }
                }

                c->delta *= derivateActivationFunction(n->output);
                c->weight -= LEARNING_RATE * (c->delta * (0 == i
                                                              ? nn->input[z]
                                                              : nn->layers[i - 1][z].output));
            }
        }
        sizeOldLayer = sizeLayer;
    }
}

int setInput(NeuralNetwork nn, const float val) {
    if (nn == nullptr || nn->loadedInput >= INPUT_NEURON) return 0;
    nn->input[nn->loadedInput++] = val;
    return 1;
}

int setExpectedOutput(NeuralNetwork nn, const float val) {
    if (nn == nullptr || nn->loadedEpectedOutput >= OUTPUT_NEURON) return 0;
    nn->input[nn->loadedEpectedOutput++] = val;
    return 1;
}


//------------------//

NeuralNetwork newFromBlackNeuralNetwork() {
    const auto nn = static_cast<NeuralNetwork>(malloc(sizeof(struct NeuralNetwork_)));
    if (nn != nullptr) {
        nn->loadedInput = 0;
        nn->loadedEpectedOutput = 0;
        nn->training = 1;
        for (unsigned i = 0; i <= HIDDEN_LAYER; i++) {
            const unsigned sizeConnections = 0 == i ? INPUT_NEURON : HIDDEN_LAYER_NEURONS;
            const unsigned sizeLayer = HIDDEN_LAYER == i ? OUTPUT_NEURON : HIDDEN_LAYER_NEURONS;

            auto* n = static_cast<Neuron *>(calloc(sizeLayer, sizeof(Neuron)));
            if (n == nullptr) return nullptr;

            for (unsigned y = 0; y < sizeLayer; y++) {
                auto* c = static_cast<Connection *>(calloc(sizeConnections, sizeof(Connection)));
                if (c == nullptr) return nullptr;
                for (unsigned z = 0; z < sizeConnections; z++) {
                    c[z].weight = RANDOM() - 0.5f;
                }
                n[y].inputConnection = c;
            }
            nn->layers[i] = n;
        }
    }
    return nn;
}

NeuralNetwork newFromFileNeuralNetwork(char* pathFile) {
    FILE* f = fopen(pathFile, "r");
    const auto nn = static_cast<NeuralNetwork>(malloc(sizeof(struct NeuralNetwork_)));
    if (nn != nullptr && f != nullptr) {
        nn->loadedInput = 0;
        nn->loadedEpectedOutput = 0;
        nn->training = 1;
        for (unsigned i = 0; i <= HIDDEN_LAYER; i++) {
            const unsigned sizeConnections = 0 == i ? INPUT_NEURON : HIDDEN_LAYER_NEURONS;
            const unsigned sizeLayer = HIDDEN_LAYER == i ? OUTPUT_NEURON : HIDDEN_LAYER_NEURONS;

            auto* n = static_cast<Neuron *>(calloc(sizeLayer, sizeof(Neuron)));
            if (n == nullptr) return nullptr;

            for (unsigned y = 0; y < sizeLayer; y++) {
                auto* c = static_cast<Connection *>(calloc(sizeConnections, sizeof(Connection)));
                if (c == nullptr) return nullptr;
                for (unsigned z = 0; z < sizeConnections; z++) {
                    fscanf(f, "%f,%f;\n", &c[z].weight, &c[z].delta);
                }
                n[y].inputConnection = c;
            }
            nn->layers[i] = n;
        }
    }
    return nn;
}


void feedForward(NeuralNetwork nn, float* output) {
    if (nn == nullptr) return;
    nn->loadedInput = 0;
    nn->loadedEpectedOutput = 0;

    for (unsigned i = 0; i <= HIDDEN_LAYER; i++) {
        const unsigned sizeConnections = 0 == i ? INPUT_NEURON : HIDDEN_LAYER_NEURONS;
        const unsigned sizeLayer = HIDDEN_LAYER == i ? OUTPUT_NEURON : HIDDEN_LAYER_NEURONS;

        for (unsigned y = 0; y < sizeLayer; y++) {
            float sum = 0.0f;
            Neuron* n = &nn->layers[i][y];

            for (unsigned z = 0; z < sizeConnections; z++) {
                sum += n->inputConnection[z].weight * (0 == i ? nn->input[z] : nn->layers[i - 1][z].output);
            }
            n->output = activationFunction(sum);
            if (output != nullptr && i == HIDDEN_LAYER)
                output[y] = n->output;
        }
    }

    if (nn->training)
        backPropagate(nn);
}

void freeNN(NeuralNetwork nn) {
    if (nn == nullptr) return;
    for (unsigned i = 0; i <= HIDDEN_LAYER; i++) {
        const unsigned sizeLayer = HIDDEN_LAYER == i ? OUTPUT_NEURON : HIDDEN_LAYER_NEURONS;

        for (unsigned y = 0; y < sizeLayer; y++) {
            free(nn->layers[i][y].inputConnection);
        }
        free(nn->layers[i]);
    }
}

int loadValue(NeuralNetwork nn, const float val) {
    if (nn == nullptr) return 0;
    if (nn->loadedInput < INPUT_NEURON) {
        return setInput(nn, val);
    }
    return setExpectedOutput(nn, val);
}

int saveNeuralNetwork(NeuralNetwork nn, char* pathFile) {
    FILE* f = fopen(pathFile, "w");
    if (nn == nullptr || f == nullptr) return 0;
    for (unsigned i = 0; i <= HIDDEN_LAYER; i++) {
        const unsigned sizeConnections = 0 == i ? INPUT_NEURON : HIDDEN_LAYER_NEURONS;
        const unsigned sizeLayer = HIDDEN_LAYER == i ? OUTPUT_NEURON : HIDDEN_LAYER_NEURONS;

        for (unsigned y = 0; y < sizeLayer; y++) {
            for (unsigned z = 0; z < sizeConnections; z++) {
                fprintf(f, "%f,%f;\n", nn->layers[i][y].inputConnection->weight,
                        nn->layers[i][y].inputConnection->delta);
            }
        }
    }
    return 1;
}
