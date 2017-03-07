from gensim.models import word2vec


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        with open(self.dirname, encoding="utf-8") as fp:
            buffer = []

            for line in fp:
                line = line.rstrip() # strip off the trailing newline
                if not line.startswith('#'):
                    if len(line) == 0:
                        words = ['<ROOT>'] + [element[1] for element in buffer]
                        if len(words) > 1:
                            #print(words)
                            yield ['<ROOT>'] + [element[1] for element in buffer]
                        buffer = []
                    else:
                        columns = line.split()
                        if columns[0].isdigit(): # skip range tokens
                            buffer.append(columns)


class WSM:

    def __init__(self):
        self.model = word2vec.Word2Vec()

    def create_model(self, path):
        sentences = MySentences(path)  # a memory-friendly iterator
        self.model = word2vec.Word2Vec(sentences)
        print(self.model['This'])
        self.model.save('models/wsm_en')