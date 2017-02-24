import sys
import getopt
import re

def conllu(fp):
    # for tempLine in fp.readline():
    tempLine = fp.readline()
    returnList = list()
    # print(tempLine)
    while(True):
        if(tempLine[0] == "#"):
            tempLine = fp.readline()
        wordList = tempLine.split( )
        returnList.append(wordList)
        tempLine = fp.readline()

        if(tempLine[0] == "#"):
            del returnList[-1]
            theReturnList = returnList
            returnList = list()
            yield theReturnList


def trees(fp):
    """Reads trees from an input source.
    
    Args:
        fp: A file pointer.
    Yields:
        Triples of the form words, tags, heads where: words is the list of
        words of the tree (including the pseudo-word <ROOT> at position 0),
        tags is the list of corresponding tags, and heads is the list of
        head indices (one head index per word in the tree).
    """

    for tree in conllu(fp):
        # print(tree)

        pos = list()
        word = list()
        Head = list()
        bigList = list()

        word.append('<ROOT>')
        pos.append('<ROOT>')
        Head.append(0)
        # print(tree)
        for tokens in tree:
            # print(tokens)
            if(len(tokens)>0):
                word.append(tokens[1])
                pos.append(tokens[3])
                Head.append(int(tokens[6]))

        bigList.append(word)
        bigList.append(pos)
        bigList.append(Head)

        yield bigList



def evaluate(train_file, test_file, par):
    n_examples = 1   # Set to None to train on all examples

    with open(train_file) as fp:
        for i, (words, gold_tags, gold_tree) in enumerate(trees(fp)):
            # print(words)
            par.update(words, gold_tags, gold_tree)
            print("\rUpdated with sentence #{}".format(i))
            if n_examples and i >= n_examples:
                print("dataReader Break")
                break

    par.finalize()

    acc_k = acc_n = 0
    uas_k = uas_n = 0
    with open(test_file) as fp:
        for i, (words, gold_tags, gold_tree) in enumerate(trees(fp)):
            pred_tags, pred_tree = par.parse(words)
            print(pred_tags)
            print(gold_tags)
            acc_k += sum(int(g == p) for g, p in zip(gold_tags, pred_tags)) - 1
            acc_n += len(words) - 1
            uas_k += sum(int(g == p) for g, p in zip(gold_tree, pred_tree)) - 1
            uas_n += len(words) - 1
            print("\rParsing sentence #{}".format(i))
            if(i >= 2000):
                break
        print("")
    print("Tagging accuracy: {:.2%}".format(acc_k / acc_n))
    print("Unlabelled attachment score: {:.2%}".format(uas_k / uas_n))

