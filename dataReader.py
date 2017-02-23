import sys
import getopt
import re

def main():
	print("Start")
	with open("/home/johli160/Desktop/Project/en-ud-train.conllu") as fp:
		print(next(trees(fp)));
		print(next(trees(fp)));



def conllu(fp):
	returnList = list();
	while(True):	
		tempLine = fp.next();
		if(tempLine[0] == "#"):
			break;
		#wordList = re.sub("[^\w]", " ",  tempLine).split()
		wordList = tempLine.split( );
		print(wordList)
		returnList.append(wordList);		
	if(len(returnList) > 1):	
		del returnList[-1]
	yield returnList

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
    tree =  conllu(fp);
    
    for tree in conllu(fp):
        pos = list();
        word = list();
        Head = list();
        bigList = list();
        
        pos.append('<ROOT>');
        word.append('<ROOT>');
        Head.append(0);
        
        for tokens in tree:
            pos.append(tokens[1])
            word.append(tokens[3])
            Head.append(int(float(tokens[6])))
            
        bigList.append(pos);
        bigList.append(word);
        bigList.append(Head);
    
        yield bigList;

def evaluate():
n_examples = 2000   # Set to None to train on all examples

parser = Parser()
with open("/home/TDDE09/labs/nlp4/en-ud-train-projective.conllu") as fp:
    for i, (words, gold_tags, gold_tree) in enumerate(trees(fp)):
        parser.update(words, gold_tags, gold_tree)
        print("\rUpdated with sentence #{}".format(i), end="")
        if n_examples and i >= n_examples:
            break
    print("")
parser.finalize()

acc_k = acc_n = 0
uas_k = uas_n = 0
with open("/home/TDDE09/labs/nlp4/en-ud-dev.conllu") as fp:
    for i, (words, gold_tags, gold_tree) in enumerate(trees(fp)):
        pred_tags, pred_tree = parser.parse(words)
        acc_k += sum(int(g == p) for g, p in zip(gold_tags, pred_tags)) - 1
        acc_n += len(words) - 1
        uas_k += sum(int(g == p) for g, p in zip(gold_tree, pred_tree)) - 1
        uas_n += len(words) - 1
        print("\rParsing sentence #{}".format(i), end="")
    print("")
print("Tagging accuracy: {:.2%}".format(acc_k / acc_n))
print("Unlabelled attachment score: {:.2%}".format(uas_k / uas_n))

	

if __name__ == "__main__":
    main()
