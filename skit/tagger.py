class Tagger():
    def __init__(self, tags):
        """Creates a new tagger that uses the specified tag set."""
        self.count = 0
        self.tags = tags
        self.accumulator = {}
        self.weights = {}
        for tag in tags:
            self.accumulator.setdefault(tag, {})
            self.weights.setdefault(tag, {})

    def tag(self, words):
        """Tags the specified words, returning a tagged sentence."""
        wordsScores = list()
        pred_tags = list()
        i = 0
        for word in words:
            features = self.get_features(words, i, pred_tags)

            tag = self.predict(features)
            wordsScores.append((word, tag))
            pred_tags.append(tag)
            i += 1
        return wordsScores

    def predict(self, features):
        scores = {}
        # maps classes to scores
        for tag in self.tags:
            scores[tag] = 0
            for e in range(0, len(features)):
                feature = features[e]
                if feature in self.weights[tag]:
                    scores[tag] += self.weights[tag][feature]

                else:
                    self.weights[tag][feature] = 0
                    self.accumulator[tag][feature] = 0

        return max(scores, key=lambda tag: scores[tag])

    def train(self, tagged_sentences, report_progress=True):
        """Trains this tagger on the specified gold-standard data."""
        self.weights = dict()
        self.accumulator = dict()
        for tag in self.tags:
            self.weights[tag] = {}
            self.accumulator[tag] = {}

        self.count = 1
        for e in range(0, 1):

            for sentence in tagged_sentences:
                i = 0
                k = 0
                pred_tags = list()
                words = list()

                for tupel in sentence:
                    words.append(tupel[0])

                for word in words:

                    features = self.get_features(words, k, pred_tags)

                    p = self.predict(features)
                    goldTag = sentence[i][1]

                    if p != goldTag:
                        for feature in features:
                            self.weights[goldTag][feature] += 1
                            self.weights[p][feature] -= 1

                            self.accumulator[p][feature] -= self.count
                            self.accumulator[goldTag][feature] += self.count

                    i += 1

                    self.count += 1
                    k += 1
                    pred_tags.append(p)

        
        # Averaging          
        for tag in self.tags:
            for word in self.weights[tag]:
                self.weights[tag][word] -= self.accumulator[tag][word] / self.count

    def update(self, words, gold_tags):
        pred_tags = list()
        i = 0
        k = 0
        for word in words:
            features = self.get_features(words, k, pred_tags)

            p = self.predict( features)
            goldTag = gold_tags[i]

            if p != goldTag:
                for feature in features:
                    self.weights[goldTag][feature] += 1
                    self.weights[p][feature] -= 1

                    self.accumulator[p][feature] -= self.count
                    self.accumulator[goldTag][feature] += self.count

            i += 1

            self.count += 1
            k += 1
            pred_tags.append(p)

        return pred_tags



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
        for tag in self.tags:
            for word in self.weights[tag]:
                self.weights[tag][word] -= self.accumulator[tag][word] / self.count