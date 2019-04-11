from math import exp
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from timeit import timeit

def sigmoid(x):
    return 1 / (1 + exp(-x))
sigmoid = np.vectorize(sigmoid)

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
d_sigmoid = np.vectorize(d_sigmoid)

learning_rate = 0.001
verbose = False

class Layer():
    
    def __init__(self, input_size, output_size, weights=None, bias=None):
        self.weights = np.random.randn(output_size, input_size)
        # self.bias = np.random.randn(output_size)
        self.input = np.zeros(input_size)
        self.output = np.zeros(output_size)
        self.inter = np.zeros(output_size)
        self.grad_output = np.zeros(output_size)
        self.grad_inter = np.zeros(output_size)
        self.grad_weights = np.zeros((output_size, input_size))
        # Bias

    def __repr__(self):
        return "<Layer {} -> {}>".format(self.weights.shape[1], self.weights.shape[0])

    def compute(self, input):
        self.input = input
        self.inter = self.weights @ input #+ self.bias
        self.output = sigmoid(self.inter)
        return self.output

    def compute_gradient(self, input=None, next_layer=None, expected=None):
        if expected is not None:
            self.grad_output = self.output - expected
        else:
            self.grad_output = next_layer.weights.T @ next_layer.grad_inter
        self.grad_inter = d_sigmoid(self.inter) * self.grad_output
        if input is None:
            input = self.input
        self.grad_weights =  np.outer(self.grad_inter, input)

class MultiLayerPerceptron():
    layers = []

    def __init__(self, layer_sizes):
        for k in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[k], layer_sizes[k+1]))

    def __repr__(self):
        return "<MultiLayerPerceptron {}>".format(self.layers)

    @staticmethod
    def load(filename):
        array = np.load(open(filename, 'rb'))
        sizes = [array[0].shape[1]]
        for layer in array:
            sizes.append(layer.shape[0])
        mlp = MultiLayerPerceptron(sizes)
        for k, layer in enumerate(array):
            mlp.layers[k].weights = layer
        return mlp

    def save(self, filename):
        array = [layer.weights for layer in self.layers]
        np.save(open(filename, 'wb'), array)

    def error(self, output, expected):
        return 1/2 * np.sum((expected - output)**2)

    def frontprop(self, input) -> np.ndarray:
        for layer in self.layers:
            input = layer.compute(input)
        return input

    def backprop(self, input: np.ndarray, expected):
        self.frontprop(input)
        self.layers[-1].compute_gradient(expected=expected)
        for i in range(len(self.layers)-2, 0, -1):
            self.layers[i].compute_gradient(next_layer=self.layers[i+1])
        self.layers[0].compute_gradient(input=input, next_layer=self.layers[1])

    def fit(self):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.grad_weights

    def test(self, test_space, test_iterations=100):
        errors = []
        success = 0
        for j in range(test_iterations):
            input, expected = random.choice(test_space)
            output = self.frontprop(input)
            errors.append(self.error(output, expected))
            if np.argmax(output) == np.argmax(expected):
                success += 1
        return sum(errors)/100, success/100

    def learn(self, space, test_space, iterations=100, test_iterations=100, epochs=100, save=None):
        mean_errors = []
        mean_accuracy = []
        begin = time.time()
        for i in range(epochs):
            if save: self.save(save)
            # Print ETA
            elapsed = time.time() - begin
            total = int(epochs/i * elapsed) if i > 0 else 0
            remaining = int(total - elapsed)
            text = "\r[Learning] {:3.2f}% ~ ert ".format(i*100/epochs)
            # if remaining//3600 > 0:
            text += "{:2d} h ".format(remaining//3600)
            remaining = remaining % 3600
            # if remaining//60 > 0:
            text += "{:2d} m ".format(remaining//60)
            remaining = remaining % 60
            text += "{:2d} s".format(remaining)
            if verbose: print(text, end="")
            # Train
            for j in range(iterations):
                input, expected = random.choice(space)
                self.backprop(input, expected)
                self.fit()
            # Test
            error, accuracy = self.test(test_space, test_iterations)
            mean_errors.append(error)
            mean_accuracy.append(accuracy)
        return mean_errors, mean_accuracy

            


if __name__ == "__main__":
    # XOR
    # mlp = MultiLayerPerceptron([2, 4, 4, 1])
    # space = [
    #     (np.array([0, 0]), np.array([0])),
    #     (np.array([0, 1]), np.array([1])),
    #     (np.array([1, 0]), np.array([1])),
    #     (np.array([1, 1]), np.array([0])),
    # ]
    # mean_errors = []
    # begin = time.time()
    # for i in range(100):
    #     errors = []
    #     elapsed = time.time() - begin
    #     remaining = int(100/i * elapsed - elapsed) if i> 0 else 0
    #     print(i, "{}% ~ {} min {} sec left".format(i, remaining//60, remaining%60), end='\r')
    #     for j in range(1600):
    #         input, expected = random.choice(space)
    #         mlp.backprop(input, expected)
    #         mlp.fit()
    #     for input, expected in space:
    #         errors.append(mlp.error(mlp.frontprop(input), expected))
    #     mean_errors.append(sum(errors, 0)/4)
    # for input, expected in space:
    #     print(mlp.frontprop(input), expected)
            
    # plt.plot(mean_errors)
    # plt.show()

    # MNIST
    print("[Initializing] Perceptron: ", end="")
    # mlp = MultiLayerPerceptron([28*28, 32, 16, 10])
    mlp = MultiLayerPerceptron.load('mnist.npy')
    print("OK")
    print("[Loading] Training set: ", end="")
    space = np.load(open("data/train.npy", "rb"))
    space = [(input.reshape(28*28), expected) for input, expected in space]
    print("OK    Testing set: ", end="")
    test_space = np.load(open("data/test.npy", "rb"))
    test_space = [(input.reshape(28*28), expected) for input, expected in test_space]
    print("OK")

    mean_errors, mean_accuracy = mlp.learn(space, test_space, iterations=1000, epochs=100, save='mnist.npy')
            
    image, expected = random.choice(space)
    input = np.reshape(image, 28*28)
    print(mlp.frontprop(input), expected)
    print(np.argmax(mlp.frontprop(input)), "=>", np.argmax(expected))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(mean_errors)
    plt.ylabel("error")
    plt.subplot(212)
    plt.plot(mean_accuracy)
    plt.ylabel("accuracy")
    plt.axis([0, len(mean_accuracy), 0, 1])
    plt.show()