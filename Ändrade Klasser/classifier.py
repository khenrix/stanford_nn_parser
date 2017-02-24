from bokeh.resources import EMPTY

training_data = None # nlp1.load_data("/home/TDDE09/labs/nlp1/review_polarity.train.json")
test_data = None # nlp1.load_data("/home/TDDE09/labs/nlp1/review_polarity.Parser.json")


class Classifier():

    def __init__(self):
        """Initialises a new classifier."""
        self.classes = []
        self.weights = {}
        self.accumilator = {}
        self.count = 1

    def predict(self, feature,candidates=None):
        scores = {} # maps classes to scores
        occur = {}
        for f in feature:
            occur[f] = occur.setdefault(f, 0) + 1

        for c in self.weights.keys():
            for f in occur:
                scores[c] = scores.setdefault(c, 0.0) + self.weights[c].setdefault(f, 0.0) * occur[f]
                self.accumilator[c].setdefault(f, 0.0)
        return max(scores, key=lambda c: (scores[c], c))

    def update(self, feature, gold):
        if gold not in self.classes:
            self.classes.append(gold)
            self.weights[gold] = {}
            self.accumilator[gold] = {}

        p = self.predict(feature)

        if p!=gold:
            for word in feature:
                self.weights[p][word] -= 1
                self.accumilator[p][word] -=self.count

                self.weights[gold][word] += 1
                self.accumilator[gold][word] +=self.count

        self.count+=1
        return p

    def finalize(self):
        #Averaging
        for c in self.accumilator.keys():
            for f in self.accumilator[c]:
                self.weights[c][f] = self.weights[c].setdefault(f, 0.0) - self.accumilator[c].setdefault(f, 0.0) / self.count