import numpy as np
import tagger

import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

class NN():

    weights_word = {}
    weights_tags = {}
    weights_label = {}
    tagger = None

    def __init__(self, tagger):
        self.tagger = tagger

    def hidden_function(self, w, t, l, bias=1, func='cube'):
        # Get feature representation of elements
        word = self.embedd(w,'w')
        tag = self.embedd(t,'t')
        label = self.embedd(l,'l')

        # Calcaulate hidden layer
        data = word*self.weights_word[w] + tag+self.weights_tags[t] + label*self.weights_label[l] + bias

        return self.activation_function(data,func)

    def activation_function(self, data, func):
        if func == 'cube':
            return data^3
        elif func == 'tanh':
            return np.tanh(data)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-data))
        elif func == 'identity':
            return data

    def embedd(self, element, type):
        # Represent element as a d-dimensional vector
        # TODO: What is the format ? What is the embedding?
        feature = 0

        if type == 'w':
            # Transform Word
            feature = 0
        elif type == 't':
            # Transform Tag
            feature = 0
        elif type == 'l':
            # Transform Arc-Label
            feature = 0

        return feature

    def create_sets(self, buffer, stack):
        Sw, St, Sl = [], [], [0 for elem in buffer]

        # Get the top 3 words on stack, if less then 3 exist, add Null instead
        if len(stack) < 3:
            Sw = stack
            St = self.tagger.tag(Sw)
            while len(Sw) < 3:
                Sw.append('NULL')
                St.append('NULL')
        else:
            Sw = stack[:-4:-1] # "Pop" 3 elements from stack
            St = self.tagger.tag(Sw)

        # Get the top 3 words on the buffer
        if len(buffer) < 3:
            Sw.extend(buffer)
            St.extend(self.tagger.tag(Sw))
            while len(Sw) < 6:
                Sw.append('NULL')
                St.append('NULL')
        else:
            Sw.extend(buffer[0:2])
            St.extend(self.tagger.tag(Sw[3:6]))

        return Sw, St, Sl


    def predict(self, buffer, stack):

        Sw, St, Sl = self.create_sets()
        h = self.hidden_function()

    def update(self, buffer, stack):
