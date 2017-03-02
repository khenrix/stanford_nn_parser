import numpy as np
import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

class NN():
    hidden_amount = 3
    hidden_weights = {}
    hidden_features = {}

    def __init__(self, activation):
        temp = {x:rnd.randint(-1,1) for x in activation} #{0:rnd, 1:rnd, 2:rnd}
        self.hidden_weights = {y:0 for y in activation} #{0:{0:rnd, 1:rnd, 2:rnd}, 1:{0:rnd, 1:rnd, 2:rnd}, 2: {0:rnd, 1:rnd, 2:rnd}}
        print(self.hidden_weights)
        self.hidden_features = {z:0 for z in activation} #{0:0, 1:0, 2:0}

    def hidden_function(self,activation):
        return np.tanh(activation)

    def predict(self, activation,candidates):
        output = {}
        for c in candidates:
            for feature in self.hidden_features:
                self.hidden_features[feature] = self.hidden_function(feature)+1
                #print(self.hidden_weights[c].setdefault(feature,0.0))
                #print(feature)
                output[c] = output.setdefault(c, 0.0)+ self.hidden_weights[c].setdefault(feature,0.0)* self.hidden_features[feature]
                #print(output)
        
        if(max(output, key = lambda c: (output[c], c)) in candidates):
            #print(output)
            #print(max(output, key = lambda c: (output[c], c)))
            return max(output, key = lambda c: (output[c], c))
        else:
            return 0
        
    
    def update(self,p,gold):
        #print(self.hidden_weights)
        if p != gold:
            for feature in [0,1,2]:
                #print(feature)
                #print(self.hidden_weights[p][word])
                #print(feature)
                self.hidden_weights[p][feature] -= 1

                self.hidden_weights[gold][feature] += 1

        return p
