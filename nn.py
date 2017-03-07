from gensim.models import word2vec
import numpy as np
import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

class NN():

    weights_word = {}
    weights_tags = {}
    weights_label = {}
    weights_hidden = {}
    tagger = None

    def __init__(self, tagger):
        """

        Args:
            tagger: Tagger for Part of Speech (POS)
            wsm: Word Space Model used for embedding text (word2Vec)

        Returns:

        """
        self.tagger = tagger
        #self.wsm = word2vec.Word2Vec.load(fname)

    def hidden_function(self, w, t, l, bias=1, func='cube'):
        # Get feature representation of elements
        word = self.embedd(w, 'w')
        tag = self.embedd(t, 't')
        label = self.embedd(l, 'l')

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

    def embedd(self, set):
        # Represent element as a d-dimensional vector
        feature = []
        #for element in set:
           #feature.append(self.wsm[element])

        return feature

    def create_sets(self, buffer, stack, pdt):
        Sw, St, Sl = [], [], []
        # TODO: Tag Arc Labels? How ? another tagger?
        # TODO: This can be done so much fancier, ugly code
        # PDT: Predicted Dependency Tree
        # Get the top 3 words on stack, if less then 3 exist, add Null instead
        if len(stack) < 3:
            Sw = [buffer[elem] for elem in stack]
            while len(Sw) < 3:
                Sw.append('NULL')
        else:
            Sw = [buffer[elem] for elem in stack[:-4:-1]]  # "Pop" 3 elements from stack

        # Get the top 3 words on the buffer
        if len(buffer) < 3:
            Sw.extend(buffer)
            St.extend(self.tagger.tag(Sw))
            while len(Sw) < 6:
                Sw.append('NULL')
        else:
            Sw.extend(buffer[0:2])

        # Get the first and second leftmost / rightmost children of the top two words on the stack
        x = stack[-2:]
        for elem in x:
            if elem in pdt:
                leftmost = pdt.index(elem)
                rightmost = len(pdt) - pdt[::-1].index(elem) - 1

                if leftmost == elem:
                    Sw.append('NULL')
                else:
                    Sw.append(buffer[leftmost])
                if rightmost == elem or rightmost == leftmost:
                    Sw.append('NULL')
                else:
                    Sw.append(buffer[rightmost])
            else:
                Sw.append('NULL')
                Sw.append('NULL')

        # Get the leftmost of leftmost / rightmost of right- most children of the top two words on the stack
        for elem in x:
            if elem in pdt:
                right = len(pdt) - pdt[::-1].index(elem) - 1
                if right in pdt:
                    leftmost = pdt.index(right)

                    if leftmost == right:
                        Sw.append('NULL')
                    else:
                        Sw.append(buffer[leftmost])
                else:
                     Sw.append('NULL')
            else:
                Sw.append('NULL')

        return self.embedd(Sw), St, Sl  # self.embedd(Sw, 'w'), self.embedd(St, 't'), self.embedd(Sl, 'l')

    def predict(self, buffer, stack, pdt):
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

        Sw, St, Sl = self.create_sets(buffer, stack, pdt)
        print(Sw)
        #h = self.hidden_function(Sw, St, Sl)
        # TODO: Handle whole feature vector instead of one word.
        #CALCULATE hidden_feature * weight_hidden
        #return 0

    # def update(self, buffer, stack):


