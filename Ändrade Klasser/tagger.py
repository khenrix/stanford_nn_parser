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
        
        return tagged_words

    def predict(self, features):
        scores = {}
        for tag_class in self.classes:      
            scores.update({tag_class:0})     
        for tag in self.classes:
            for f in features:
                if(f in self.weights[tag]):
                    scores.update({tag:scores[tag]+self.weights[tag][f]})
                else:
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



    def get_features(self, tokens, i, pred_tags):
        """Extracts the feature list for the specified configuration."""
        features = list();

        x = len(pred_tags)

        features.append(tokens[i])
        features.append(tokens[i])
        features.append(tokens[i])
        features.append(tokens[i])

        if (x > 0):
            features.append(pred_tags[i - 1])
            features.append(tokens[i - 1])
            features.append((pred_tags[i - 1], tokens[i - 1]))
            features.append((tokens[i - 1], tokens[i]))
        else:
            features.append("BOSTAG")
            features.append("BOS")
            features.append(("BOSTAG", "BOS"))
            features.append(("BOS", tokens[i]))

        if (i == len(tokens) - 1):
            features.append(("EOS", tokens[i]))
        else:
            features.append((tokens[i + 1], tokens[i]))

        return features;


    def finalize(self):
        for k in self.classes:
            for word in self.weights[k]:
                self.weights[k][word] = self.weights[k][word] - self.acc[k][word]/self.count
