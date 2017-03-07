import parser
import tagger
import dataReader
import wsm

def main():
    sv_tags = ["<ROOT>", "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]
    en_tags = ["<ROOT>", "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    arc_tags = ['<ROOT>', 'name', 'nsubjpass', 'dobj', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse','det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj','list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']

    sv_train_file = "treebanks/sv_train.conllu"
    sv_test_file = "treebanks/sv_dev.conllu"
    en_train_file = "treebanks/en_train.conllu"
    en_test_file = "treebanks/en_dev.conllu"

    #model = wsm.WSM()
    #model.create_model(en_train_file)

    model_path = 'models/wsm_en'
    myTagger = tagger.Tagger(en_tags)
    arc_tagger = tagger.Tagger(arc_tags)
    myParser = parser.Parser(myTagger, arc_tagger ,model_path)
    dataReader.evaluate(en_train_file, en_test_file, myParser)
    dataReader.evaluate(sv_train_file, sv_test_file, myParser)


if __name__ == '__main__':
    main()
