#include "Net.h"

int main() {
    // NeuralNetwork nn = newFromBlackNeuralNetwork();
    NeuralNetwork nn = newFromFileNeuralNetwork("C:/Users/Gennaro/CLionProjects/SimpleFFNN/file.txt");

    saveNeuralNetwork(nn, "C:/Users/Gennaro/CLionProjects/SimpleFFNN/file1.txt");

    freeNN(nn);
}
