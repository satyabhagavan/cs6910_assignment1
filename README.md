
## Neaural networks class:
class object takes following as the input :
inputSize, hiddenLayers, outputSize, sizeOfHiddenLayers, batchSize, learningRate, initialisationType, optimiser, epochs, activationFunc, weightDecay, lossFunc, dataset.

possible optimisation functions are 
- sgd
- momentum based gradient descent
- nesterov accelerated gradient descent
- rmsprop
- adam
- nadam
Possible Activation functions are
- Tanh
- Sigmoid
- Relu
Possible weight inistilaisers are
- Random
- Xavior

One can define their own no.of hidden layers, batch size, and no.of neaurons per layer, L2 regularisation coefficient(weight decay), epochs, and dataset from "mnist" or "fashion_mnist". 

