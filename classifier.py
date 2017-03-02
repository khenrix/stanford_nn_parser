import nn

class Classifier():

    def __init__(self):
        """Initialises a new classifier."""
        self.classes = []
        self.weights = {}
        self.accumilator = {}
        self.count = 1
        self.nn = None

    def predict(self, feature,candidates=None):
        if candidates == None:
            candidates = self.weights.keys()
        scores = {} # maps classes to scores
        occur = {}
        for f in feature:
            occur[f] = occur.setdefault(f, 0) + 1

        for c in candidates:
            for f in occur:
                scores[c] = scores.setdefault(c, 0.0) + self.weights[c].setdefault(f, 0.0) * occur[f]
                self.accumilator[c].setdefault(f, 0.0)
                
        if self.nn is None:
            self.nn = nn.NN(scores)
            
        output = self.nn.predict(scores)
        return (max(scores, key=lambda c: (scores[c], c)),output)

    def update(self, feature, gold):
        if gold not in self.classes:
            self.classes.append(gold)
            self.weights[gold] = {}
            self.accumilator[gold] = {}

        p = self.predict(feature)
        #self.nn.update(feature, p[1], gold)
        
        if p[0] != gold:
            for word in feature:
                self.weights[p[0]][word] -= 1
                self.accumilator[p[0]][word] -=self.count

                self.weights[gold][word] += 1
                self.accumilator[gold][word] +=self.count

        self.count+=1
        return p

    def finalize(self):
        # Averaging
        for c in self.accumilator.keys():
            for f in self.accumilator[c]:
                self.weights[c][f] = self.weights[c].setdefault(f, 0.0) - self.accumilator[c].setdefault(f, 0.0) / self.count
