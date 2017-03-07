from gensim.models import word2vec

import numpy as np
import random as rnd

# This class contains the Neural Network replacing the perceptron for the classifier.

class NN:

    weights_word = {}
    weights_tags = {}
    weights_label = {}
    weights_hidden = {}
    tagger = None

    def __init__(self, model_path):
        """

        Args:
            tagger: Tagger for Part of Speech (POS)
            wsm: Word Space Model used for embedding text (word2Vec)

        Returns:

        """
        self.wsm = word2vec.Word2Vec.load(model_path)

    def hidden_function(self, w, t, l, bias=1, func='cube'):
        # Get feature representation of elements
        word = self.embedd(w)
        tag = self.embedd(t)
        label = self.embedd(l)

        # Calcaulate hidden layer
        # TODO: Currently only handeling one element , need to handle the whole feature vector
        data = word*self.weights_word[w] + tag+self.weights_tags[t] + label*self.weights_label[l] + bias

        return self.activation_function(data, func)

    def activation_function(self, data, func):
        if func == 'cube':
            return data^3
        elif func == 'tanh':
            return np.tanh(data)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-data))
        elif func == 'identity':
            return data

    def embedd(self, set, type='w'):
        # Represent element as a d-dimensional vector
        feature = []
        if type == 'w':
            for element in set:
                if element in self.wsm:
                    feature.append(self.wsm[element])
                else:
                    # https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ
                    feature_size = 50
                    random_feature = [np.random.uniform(-0.25, 0.25, feature_size) for i in range(0,feature_size)]
                    # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
                    feature.append(random_feature)
        else:
            feature = 0

        return feature

    def get_children(self, parent, pdt, buffer, type='b'):
        leftmost = 'NULL'
        rightmost = 'NULL'
        left_pos = 'NULL'
        right_pos = 'NULL'

        if parent in pdt:
            left_pos = pdt.index(parent)
            right_pos = len(pdt) - pdt[::-1].index(parent) - 1

            if left_pos != parent:
                leftmost = buffer[left_pos]
            else:
                left_pos = 'NULL'
            # TODO: If rightmost == leftmost , should i handle it this way?
            if right_pos != parent and right_pos != leftmost:
                rightmost = buffer[right_pos]
            else:
                right_pos = 'NULL'

        if type == 'b':
            return leftmost, left_pos, rightmost, right_pos
        elif type == 'l':
            return leftmost, left_pos
        elif type == 'r':
            return rightmost, right_pos

    def top_three(self, buffer, stack, buffer_pos, tags):
        Sw, St = [], []

        # Get the top 3 words on stack, if less then 3 exist, add Null instead
        if len(stack) < 3:
            Sw = [buffer[elem] for elem in stack]
            St = [tags[elem] for elem in stack]
            while len(Sw) < 3:
                Sw.append('NULL')
                St.append('NULL')
        else:
            Sw = [buffer[elem] for elem in stack[:-4:-1]]  # "Pop" 3 elements from stack
            St = [tags[elem] for elem in stack[:-4:-1]]

        # Get the top 3 words on the buffer
        for i in range(0,3):
            if buffer_pos+i < len(buffer)-1:
                Sw.append(buffer[buffer_pos+i])
                St.append(tags[buffer_pos+i])
            else:
                Sw.append('NULL')
                St.append('NULL')

        return Sw, St

    def first_two_children(self, x, buffer, pdt, tags, arc_tags):
        # There probably is a nicer way of finding the two first occurances...
        Sw, St, Sl = [], [], []
        for elem in x:
            i = 0
            for word in buffer:
                if elem == buffer.index(word) and i < 2:
                    i += 1
                    left_child, pos_l, right_child, pos_r = self.get_children(elem, pdt, buffer, 'b')
                    Sw.extend([left_child, right_child])

                    if pos_l == 'NULL':
                        St.append(pos_l)
                        Sl.append(pos_l)
                    else:
                        St.append(tags[pos_l])
                        Sl.append(arc_tags[pos_l])

                    if pos_r == 'NULL':
                        St.append('NULL')
                        Sl.append('NULL')
                    else:
                        St.append(tags[pos_r])
                        Sl.append(arc_tags[pos_r])

            for missing in range(i*2,4):
                Sw.append('NULL')
                St.append('NULL')
                Sl.append('NULL')

        return Sw, St, Sl

    def leftmost_children(self, x, buffer, pdt, tags, arc_tags):
        Sw, St, Sl = [], [], []

        for elem in x:
            left_child, pos_l, right_child, pos_r = self.get_children(elem, pdt, buffer, 'b')
            left_child_child, pos_l = self.get_children(left_child, pdt, buffer, 'l')
            right_child_child, pos_l = self.get_children(elem, pdt, buffer, 'r')

            Sw.extend([left_child_child, right_child_child])

            if pos_l == 'NULL':
                St.append(pos_l)
                Sl.append(pos_l)
            else:
                St.append(tags[pos_l])
                Sl.append(arc_tags[pos_l])

            if pos_r == 'NULL':
                St.append('NULL')
                Sl.append('NULL')
            else:
                St.append(tags[pos_r])
                Sl.append(arc_tags[pos_r])

        if len(x) == 1:  # Ugly quick fix
            Sw.extend(['NULL']*2)
            St.extend(['NULL']*2)
            Sl.extend(['NULL']*2)

        return Sw, St, Sl


    def create_sets(self, buffer, stack, pdt, buffer_pos, tags, arc_tags):
        Sw, St, Sl = [], [], []
        # TODO: This can be done so much fancier, ugly code
        # PDT: Predicted Dependency Tree

        # Step 1 - Top three elements from stack and buffer

        Sw, St = self.top_three(buffer, stack, buffer_pos, tags)

        # Step 2 - Get the first and second leftmost / rightmost children of the top two words on the stack

        x = stack[-2:]

        if not x: # Ugly quick fix
            Sw.extend(['NULL']*12)
            St.extend(['NULL']*12)
            Sl.extend(['NULL']*12)

        temp_w, temp_t, temp_l = self.first_two_children(x, buffer, pdt, tags, arc_tags)
        Sw.extend(temp_w)
        St.extend(temp_t)
        Sl.extend(temp_l)

        if len(x) == 1:  # Ugly quick fix
            Sw.extend(['NULL']*4)
            St.extend(['NULL']*4)
            Sl.extend(['NULL']*4)

        # Step 3 - Get the leftmost of leftmost / rightmost of right- most children of the top two words on the stack

        temp_w, temp_t, temp_l = self.leftmost_children(x, buffer, pdt, tags, arc_tags)
        Sw.extend(temp_w)
        St.extend(temp_t)
        Sl.extend(temp_l)

        return Sw, St, Sl

    def predict(self, buffer, stack, pdt, buffer_pos, tags, arc_tags):
        """
        1 Create sets for words, tags and arc-labels
            Word set, Sw: [s1,s2,s3,b1,b2,b3,l(s1),r(s1),l(s2),r(s2),l(r(s1)),l(r(s2))]
            s = stack, b = buffer, l() = left child, r() = right child
            Tags set, St: Corresponding tags (POS) for words in word set
            Ex: St = {NN,NNP, NNS,DT,JJ,...}
            Arc-labels, Al: Arc-labels for each arc between words. Ex.
            Ex: Sl = {amod,tmod,nsubj,csubj,dobj,...

        2 Translate (embedd) the sets into a as vectors in a high-dimensional vector space
        3 Calculate hidden layer with chosen activation function.
        4 Calculate activation of hidden layer
        """

        Sw, St, Sl = self.create_sets(buffer, stack, pdt, buffer_pos, tags, arc_tags)
        #h = self.hidden_function(Sw, St, Sl)
        print(Sw, '\n', St, '\n', Sl, '\n')
        # TODO: Handle whole feature vector instead of one word.
        #CALCULATE hidden_feature * weight_hidden
        #return 0

    # def update(self, buffer, stack):


