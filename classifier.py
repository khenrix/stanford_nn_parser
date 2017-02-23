

training_data = None # nlp1.load_data("/home/TDDE09/labs/nlp1/review_polarity.train.json")
test_data = None #nlp1.load_data("/home/TDDE09/labs/nlp1/review_polarity.test.json")


class MyPerceptronClassifier():

    def __init__(self):
        """Initialises a new classifier."""
        super(self).__init__()

    def predict(self, review):
        scores = {'pos': 0, 'neg': 0} # maps classes to scores
        for c in ['pos', 'neg']:
            for word in review:
                if word in self.weights[c]:
                     scores[c] += self.weights[c][word] #* x[f]
                else:
                    self.weights[c].update({word:0})
                    self.accumilator[c].update({word:0})
        return max(scores, key=lambda c: scores[c])


    @classmethod
    def update(cls, data, n_epochs=1):
        mpc = cls()
        mpc.weights ={'pos':{}, 'neg' : {}}
        mpc.accumilator= {'pos':{}, 'neg':{}}
        count=1
        for e in range(0,n_epochs):
            for review in data:
                p = mpc.predict(review[0])
                y=review[1]

                if p!=y:
                    for word in review[0]:
                        mpc.weights[p][word] -= 1
                        mpc.weights[y][word] += 1
                        mpc.accumilator[p][word]-=count
                        mpc.accumilator[y][word]+=count
                count+=1
        #Averaging
        for c in ['pos','neg']:
            for word in mpc.weights[c]:
                mpc.weights[c][word] -= mpc.accumilator[c][word]/count

        return mpc