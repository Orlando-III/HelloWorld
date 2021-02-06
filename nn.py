import numpy as np

np.random.seed(0)


class Layer:

    def __init__(self, n_inputs, n_neurons):
        # Allocates the initial state of the neurons
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = []

    def forward(self, inputs):
        # Shows if the neuron fires and how much then assigns it to self.output
        self.output = np.dot(inputs, self.weights) + self.biases


class Network:

    def __init__(self, layers):
        # Generates Network with given neurons or pre-made layers
        if isinstance(layers[0], int):
            self.layers = []
            for i in range(len(layers)-1):
                self.layers.append(Layer(layers[i], layers[i + 1]))
        else:
            self.layers = layers

    def __iter__(self):
        # Initializes iterative index
        self.index = 0
        return self

    def __next__(self):
        # Returns current iteration and increases iteration index
        if self.index < len(self.layers):
            a = self.index
            self.index += 1
            return self.layers[a]
        else:
            raise StopIteration

    def forward_prop(self, data):
        # Gets output of the neural network from input data
        b = [data]
        j = 0
        for i in self:
            j += 1
            i.forward(b[-1])
            if j == len(self.layers):
                ex_val = np.exp(i.output)
                norm_val = ex_val / np.sum(ex_val)
                b.append(norm_val)
            else:
                b.append(np.maximum(0, i.output))
        return b[-1]
