class Tagger():
    def __init__(self, tags):
        """Creates a new tagger that uses the specified tag set."""
        self.count = 1
        self.classes = tags
        self.weights = {}
        self.acc = {}
        for tag_class in self.classes: 
            self.weights.update({tag_class:{}})
            self.acc.update({tag_class:{}})    


    def tag(self, words):
        """Tags the specified words, returning a tagged sentence."""
        pred_tags = []
        tagged_words = []
        for i in range(0,len(words)):
            f = self.get_features(words,i,pred_tags)
            p = self.predict(f)
            pred_tags.append(p)
            temp = (words[i],p)
            tagged_words.append(temp)

        return [tag[1] for tag in tagged_words]

    def predict(self, features):
        scores = {}
        for tag_class in self.classes:      
            scores.update({tag_class:0})     
        for tag in self.classes:
            for f in features:
                if f not in self.weights[tag]:
                    self.weights[tag].update({f:0})
                    self.acc[tag].update({f:0})
                scores.update({tag:scores[tag]+self.weights[tag][f]})
        
        return max(scores, key=lambda tag: scores[tag])


    def update(self, words, gold_tags):
        pred_tags = []    
        for i in range(0,len(words)):
            f = self.get_features(words,i,pred_tags)
            p = self.predict(f)
            pred_tags.append(p)
            y = gold_tags[i]
            
            if(p != y):
                for feature in f:
                    if(feature in self.weights[p]): 
                        self.weights[p][feature] -= 1
                        self.acc[p][feature] -= self.count
                    
                    if(feature in self.weights[y]):
                        self.weights[y][feature] += 1
                        self.acc[y][feature] += self.count
        self.count += 1
        return pred_tags


    def get_features(self, tokens, i, pred_tags):
        features = set();
        word = tokens[i]
        prev = "BOS"
        prev2 = "BOS"
        if(len(pred_tags)>0):
            prev = pred_tags[-1]
        if(len(pred_tags)>1):
            prev2 = pred_tags[-2]

        def add(name, *args):
            features.add('+'.join((name,) + tuple(args)))
        #adjust to fit the language beeing tagged.
        suffix_len = 4

        #Code below:
        # Try adding the described feature, catch error where the feature is BOS/EOS.
        # Where suffix is longer than word, add the whole word as suffix feature. If word is BOS/EOS we add that instead.
        try:
            add('i suffix', word[-suffix_len:])
        except IndexError:
            add('i suffix',word)
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', word)
        add('i-1 tag+i word', prev, word)
        
        try:
            add('i-1 word', tokens[i-1])
        except IndexError:
             add('i-1 word', "BOS")
        try:
            add('i-1 suffix', tokens[i-1][-suffix_len:])
        except IndexError:
            try:
                add('i-1 suffix',tokens[i-1])
            except IndexError:
                add('i-1 suffix',"BOS")
        
        try:
            add('i-2 word', tokens[i-2])
        except IndexError:
            add('i-2 word', "BOS")
        try:
            add('i-2 suffix', tokens[i-2][-suffix_len:])
        except IndexError:
            try:
                add('i-2 suffix', tokens[i-2])
            except IndexError:
                 add('i-2 suffix',"BOS")
        
        try:
            add('i+1 word', tokens[i+1])
        except IndexError:
            add('i+1 word', "EOS")
        try:
            add('i+1 suffix', tokens[i+1][-suffix_len:])
        except IndexError:
            try:
                add('i+1 suffix',  tokens[i+1])
            except IndexError:
                add('i+1 suffix',"EOS")
        
        try:
            add('i+2 word', tokens[i+2])
        except IndexError:
            add('i+2 word', "EOS")
        try:
            add('i+2 suffix', tokens[i+2][-suffix_len:])
        except IndexError:
            try:
                add('i+2 suffix', tokens[i+2])
            except IndexError:
                add('i+2 suffix', "EOS")
        

        return features;


    def finalize(self):
        for k in self.classes:
            for word in self.weights[k]:
                self.weights[k][word] = self.weights[k][word] - self.acc[k][word]/self.count
