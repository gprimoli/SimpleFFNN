#ifndef NET_H
#define NET_H

#define INPUT_NEURON 187
#define OUTPUT_NEURON 5
#define HIDDEN_LAYER 2
#define HIDDEN_LAYER_NEURONS 25
#define EPOCHS 50
#define LEARNING_RATE 0.8f

#define RANDOM() (((float)rand())/RAND_MAX)

// #include <Arduino.h>

typedef struct NeuralNetwork_* NeuralNetwork;

NeuralNetwork newFromBlackNeuralNetwork();
NeuralNetwork newFromFileNeuralNetwork(char *pathFile);

void feedForward(NeuralNetwork nn, float* output);
void freeNN(NeuralNetwork nn);
int loadValue(NeuralNetwork nn, float val);
int saveNeuralNetwork(NeuralNetwork nn, char *pathFile);

inline int getNumberInput() { return INPUT_NEURON; }
inline int getNumberOutput() { return OUTPUT_NEURON; }
inline int getNumberInputPlusOutput() { return INPUT_NEURON + OUTPUT_NEURON; }

#endif //NET_H


// #ifdef ARDUINO
// #else
// #endif
