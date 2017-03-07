import classifier
import nn

class Parser():
    """A transition-based dependency parser.
    
    This parser implements the arc-standard algorithm for dependency parsing.
    When being presented with an input sentence, it first tags the sentence for
    parts of speech, and then uses a multi-class perceptron classifier to
    predict a sequence of *moves* (transitions) that construct a dependency
    tree for the input sentence. Moves are encoded as integers as follows:
    """
    moves = {0:"SH", 1:"LA", 2:"RA"}
    """
    At any given point in the predicted sequence, the state of the parser can
    be specified by: the index of the first word in the input sentence that
    the parser has not yet started to process; a stack holding the indices of
    those words that are currently being processed; and a partial dependency
    tree, represented as a list of indices such that `tree[i]` gives the index
    of the head (parent node) of the word at position `i`, or 0 in case the
    corresponding word has not yet been assigned a head.
    
    Attributes:
        tagger: A part-of-speech tagger.
        classifier: A multi-class perceptron classifier used to predict the
            next move of the parser.
    """

    def __init__(self, tagger, arc_tagger, model_path):
        self.tagger = tagger
        self.arc_tagger = arc_tagger
        self.classifier = classifier.Classifier()
        self.nn = nn.NN(model_path)  # TODO: Create arc tagger


    def parse(self, words):
        # return super().parse(words)
        """Parses a sentence.
        Args:
            words: The input sentence, a list of words.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        buffer = words
        stack = []
        pdt = [0]*len(words)

        tags = self.tagger.tag(words)
        x=0

        while (True):
            valid_moves = self.valid_moves(x,stack,pdt)
            if not valid_moves:
                break
            feature = self.features(words,tags,x,stack,pdt)
            predicted_move = self.classifier.predict(feature, valid_moves)
            x,stack,pdt = self.move(x,stack,pdt,predicted_move)

        return tags, pdt


    def valid_moves(self, i, stack, pred_tree):
        # return super().valid_moves(i, stack, pred_tree)
        """Returns the valid moves for the specified parser configuration.*
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser configuration.
        """
        valid_moves = []
        # There are elements left in the buffer.
        if(i < len(pred_tree)):
            valid_moves.append(0)
        # There are more than two elements + root element in the stack
        if(len(stack)>=2):
            valid_moves.append(1)
            valid_moves.append(2)
        # There are only 1 element + the root element in the stack
        elif(len(stack)==2):
            valid_moves.append(2)

        return valid_moves


    def move(self, i, stack, pred_tree, move):
        # return super().move(i, stack, pred_tree, move)
        """Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple containing
            the index of the new first unprocessed word, stack, and partial
            dependency tree.
        """
        i = i
        stack = stack
        pred_tree=pred_tree
        # shift
        if(move == 0):
            stack.append(i)
            i+=1
        # left
        elif(move == 1):
            ind = stack.pop(-2)
            pred_tree[ind] = stack[-1]
        # right
        elif(move == 2):
            ind = stack.pop(-1)
            pred_tree[ind] = stack[-1]

        return i, stack, pred_tree


    def update(self, words, gold_tags, gold_arclabels, gold_tree):
        # return super().update(words, gold_tags, gold_tree)
        """Updates the move classifier with a single training example.
        
        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input sentence.
            gold_tree: The gold-standard tree for the sentence.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        buffer = words
        stack = []
        pdt = [0]*len(words)

        tags = self.tagger.update(words, gold_tags)
        arc_tags = self.arc_tagger.update(words, gold_arclabels)  # TODO: This is wrong, currently only looking at words , not pdt and words

        x=0
        while (True):
            g_move = self.gold_move(x,stack,pdt,gold_tree)
            if g_move is None:
                break
            feature = self.features(words,tags,x,stack,pdt)
            self.classifier.update(feature,g_move)
            self.nn.predict(buffer, stack, pdt, x, gold_tags, gold_arclabels)
            x, stack, pdt = self.move(x,stack,pdt,g_move)


        return tags, pdt


    def gold_move(self, i, stack, pred_tree, gold_tree):
        """Returns the gold-standard move for the specified parser
        configuration.
        
        The gold-standard move is the first possible move from the following
        list: LEFT-ARC, RIGHT-ARC, SHIFT. LEFT-ARC is possible if the topmost
        word on the stack is the gold-standard head of the second-topmost word,
        and all words that have the second-topmost word on the stack as their
        gold-standard head have already been assigned their head in the
        predicted tree. Symmetric conditions apply to RIGHT-ARC. SHIFT is
        possible if at least one word in the input sentence still requires
        processing.
        
        Args:
            i: The index of the first unprocessed word.
           stack: The stack of words that are currently being processed.
            pred_tree: The partial dependency tree.
            gold_tree: The gold-standard dependency tree.
        
        Returns:
            The gold-standard move for the specified parser configuration, or
            None if no move is possible.
        """

        pdt = pred_tree
        gold_tree=gold_tree
        valid_moves = self.valid_moves(i,stack,pdt)
        try:
            if(len(stack)>=2):
                stack_top = stack[-1]
                stack_sec = stack[-2]
                if(1 in valid_moves and gold_tree[stack_sec] == stack_top):
                    heads = [x for x,word in enumerate(gold_tree) if word == stack_sec]
                    valid=True
                    for i in heads:
                        if(pdt[i]!=stack_sec):
                            valid = False
                    if(valid):
                        return 1
                if(2 in valid_moves and gold_tree[stack_top] == stack_sec):
                    heads = [x for x,word in enumerate(gold_tree) if word == stack_top]
                    valid=True
                    for i in heads:
                        if(pdt[i]!=stack_top):
                            valid = False
                    if(valid):
                        return 2
            if(0 in valid_moves and i < len(pred_tree)):
                return 0
            else:
                return None
        except IndexError:
            return None



    def features(self, words, tags, i, stack, parse):
        """Extracts features for the specified parser configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            A feature vector for the specified configuration.

        """
        features = list()
        if(i<len(words)):
            features.append((0,words[i]))
            features.append((1,tags[i]))
        else:
            features.append((0,'<EOS>'))
            features.append((1,'<EOS>'))

        if(len(stack) >= 2):
            features.append((2,words[stack[-1]]))
            features.append((3,tags[stack[-1]]))

            features.append((4,words[stack[-2]]))
            features.append((5,tags[stack[-2]]))

        elif(len(stack) == 1):
            features.append((2,words[stack[-1]]))
            features.append((3,tags[stack[-1]]))

            features.append((4,'<Empty>'))
            features.append((5,'<Empty>'))
        else:
            features.append((2,'<Empty>'))
            features.append((3,'<Empty>'))

            features.append((4,'<Empty>'))
            features.append((5,'<Empty>'))

        return features


    def finalize(self):
        """Averages the weight vectors."""
        self.tagger.finalize()
        self.classifier.finalize()
        self.arc_tagger.finalize()
