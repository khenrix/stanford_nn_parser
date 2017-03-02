import Numpy as np
import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

class NN():
    hidden_amount = 3
    hidden_weights = {}
    hidden_features = {}

    def __init__(self, activation):
        temp = {x:rnd.randint(-1,1) for x in activation} #{0:rnd, 1:rnd, 2:rnd}
        self.hidden_weights = {y:temp for y in activation} #{0:{0:rnd, 1:rnd, 2:rnd}, 1:{0:rnd, 1:rnd, 2:rnd}, 2: {0:rnd, 1:rnd, 2:rnd}}
        self.hidden_features = {z:0 for z in activation} #{0:0, 1:0, 2:0}

    def hidden_function(activation):
        return np.tanh(activation)

    def predict(self, activation):
        output = {temp:0 for temp in self.hidden_features} #{0:0, 1:0, 2:0}

        for feature in self.hidden_features:
            self.hidden_features[feature] = self.hidden_function(feature)
            for weight in self.hidden_weights[feature]:
                output[weight] += self.hidden_weights[feature][weight] * self.hidden_features[feature]

        return max(output, key = lambda c: (output[c], c))
