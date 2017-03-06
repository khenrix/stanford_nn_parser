import numpy as np
import tagger

import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

class NN():

    weights_word = {}
    weights_tags = {}
    weights_label = {}
    weights_hidden = {}
    tagger = None

    def __init__(self, tagger):
        self.tagger = tagger

    def hidden_function(self, w, t, l, bias=1, func='cube'):
        # Get feature representation of elements
        word = self.embedd(w,'w')
        tag = self.embedd(t,'t')
        label = self.embedd(l,'l')

        # Calcaulate hidden layer
        # TODO: Currently only handeling one element , need to handle the whole feature vector
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
        # TODO: Implement lab 5, words space model, word2vec
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

    def create_sets(self, buffer, stack, pdt):
        Sw, St, Sl = [], [], []
        # TODO: Tag Arc Labels? How ? another tagger?
        # PDT: Predicted Dependency Tree
        # Get the top 3 words on stack, if less then 3 exist, add Null instead
        if len(stack) < 3:
            Sw = stack
            St = self.tagger.tag(Sw)
            while len(Sw) < 3:
                Sw.append('NULL')
                St.append('NULL')
        else:
            Sw = stack[:-4:-1]  # "Pop" 3 elements from stack
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

        # Get the first and second leftmost / rightmost children of the top two words on the stack
        x = [buffer.index(s) for s in stack[-2:]]
        for elem in x:
            leftmost = pdt.index(elem)
            rightmost = len(pdt) - pdt[::-1].index(elem) - 1

            if leftmost == elem:
                leftmost = 'NULL'
            if rightmost == elem:
                rightmost = 'NULL'

            Sw.append(buffer[leftmost])
            Sw.append(buffer[rightmost])

        # Get the leftmost of leftmost / rightmost of right- most children of the top two words on the stack
        for elem in x:
            right = len(pdt) - pdt[::-1].index(elem) - 1
            leftmost = pdt.index(right)

            if leftmost == right:
                leftmost = 'NULL'

            Sw.append(buffer[leftmost])

        return self.embedd(Sw, 'w'), self.embedd(St, 't'), self.embedd(Sl, 'l')

    def predict_move(self, buffer, stack, pdt):
        """
        1 Create sets for words, tags and arc-labels
            Word set, Sw: [s1,s2,s3,b1,b2,b3,l(s1),r(s1),l(s2),r(s2),l(r(s1)),l(r(s2))]
            s = stack, b = buffer, l() = left child, r() = right child
            Tags set, St: Corresponding tags (POS) for words in word set
            Ex: St = {NN,NNP, NNS,DT,JJ,...}
            Arc-labels, Al: Arc-labels for each arc between words. Ex.
            Ex: Sl = {amod,tmod,nsubj,csubj,dobj,...}

        2 Translate (embedd) the sets into a as vectors in a high-dimensional vector space
        3 Calculate hidden layer with chosen activation function.
        4 Calculate activation of hidden layer
        """
        # Currently isch pseudocode

        Sw, St, Sl = self.create_sets(buffer, stack, pdt)
        h = self.hidden_function(Sw, St, Sl)
        # TODO: Handle whole feature vector instead of one word.
        #CALCULATE hidden_feature * weight_hidden 
        return 0


    """
    Parsing section
    Currently in the same file
    TODO: Move it to parser file when done
    """
    def valid_moves(self, i, stack, pred_tree):

        valid_moves = []
        if i < len(pred_tree):
            valid_moves.append(0)

        if len(stack)>=2:
            valid_moves.append(1)
            valid_moves.append(2)

        return valid_moves

    def move(self, i, stack, pred_tree, move):
        # SH - Shift
        if move == 0:
            stack.append(i)
            i += 1
        # left
        elif move == 1:
            ind = stack.pop(-2)
            pred_tree[ind] = stack[-1]
        # right
        elif move == 2:
            ind = stack.pop(-1)
            pred_tree[ind] = stack[-1]

        return i, stack, pred_tree

    def predict(self, words):

        buffer = words
        stack = []
        pdt = [0 for word in buffer]
        i = 0
        while True:
            valid_moves = self.valid_moves(i,stack,pdt)
            if not valid_moves:
                break
            predicted_move = self.predict_move(buffer, stack, pdt)
            i, stack, pdt = self.move(i, stack, pdt, predicted_move)

    # def update(self, buffer, stack):


