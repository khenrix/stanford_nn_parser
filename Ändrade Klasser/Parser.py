import classifier
class Parser():
    moves = {0:"SH",1:"LA",2:"RA"}
    """A transition-based dependency parser.

   This parser implements the arc-standard algorithm for dependency parsing.
   When being presented with an input sentence, it first tags the sentence for
   parts of speech, and then uses a multi-class perceptron classifier to
   predict a sequence of *moves* (transitions) that construct a dependency
   tree for the input sentence. Moves are encoded as integers as follows:

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
    def __init__(self,tagger):
        self.tagger = tagger
        self.classifier = classifier.Classifier()

    def parse(self, words):
        """Parses a sentence.
        Args:
            words: The input sentence, a list of words.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        pred_tree = []

        for j in range(0,len(words)):
            pred_tree.append(0)

        pred_tags = self.tagger.tag(words)
        stack = []
        i = 0
        while self.valid_moves(i,stack,pred_tree) != []:
            x = self.features(words,pred_tags,i,stack,pred_tree)
            candidates = self.valid_moves(i,stack,pred_tree)
            move = self.classifier.predict(x,candidates)

            temp_list = self.move(i,stack,pred_tree,move)
            i = temp_list[0]
            stack = temp_list[1]
            pred_tree = temp_list[2]

        return (pred_tags,pred_tree)

    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser configuration.*
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser configuration.
        """
        valid_moves = []
        if i<len(pred_tree):
            valid_moves.append(0)

        if len(stack)>=2:
            valid_moves.append(1)
            valid_moves.append(2)

        return valid_moves

    def move(self, i, stack, pred_tree, move):
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

        if(move == 0):
            stack.append(i);
            i = i+1;
        if(move == 1):
            headIndex = stack.pop();
            tempIndex = stack.pop();
            pred_tree[tempIndex] = headIndex;
            stack.append(headIndex);
        if(move == 2):
            tempIndex = stack.pop();
            headIndex = stack.pop();
            pred_tree[tempIndex] = headIndex;
            stack.append(headIndex);

        return (i,stack,pred_tree);


    def update(self, words, gold_tags, gold_tree):
        #return super().update(words, gold_tags, gold_tree)
        """Updates the move classifier with a single training example.
        
        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input sentence.
            gold_tree: The gold-standard tree for the sentence.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        pred_tree = []

        for j in range(0,len(words)):
            pred_tree.append(0)

        self.tagger.update(words,gold_tags)
        pred_tags = self.tagger.tag(words)
        stack = []
        i = 0
        while self.valid_moves(i, stack, pred_tree) != []:
            x = self.features(words,pred_tags,i,stack,pred_tree)
            gold_move = self.gold_move(i,stack,pred_tree,gold_tree)
            self.classifier.update(x,gold_move)

            temp_list = self.move(i,stack,pred_tree,gold_move)
            i = temp_list[0]
            stack = temp_list[1]
            pred_tree = temp_list[2]

        return (pred_tags,pred_tree)

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

        left = False;
        right = False;

        validmoves = self.valid_moves(i,stack,pred_tree)

        if (1 in validmoves and stack[-1] == gold_tree[stack[-2]]):
            left = True;
            for i in range(0,len(gold_tree)):
                if(gold_tree[i] == stack[-2] and pred_tree[i] == 0):
                    left = False;

        if (2 in validmoves and stack[-2] == gold_tree[stack[-1]]):
            right = True;
            for i in range(0,len(gold_tree)):
                if(gold_tree[i] == stack[-1] and pred_tree[i] == 0):
                    right = False;

        if(left):
            return 1;
        if(right):
            return 2;
        else:
            return 0;

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
