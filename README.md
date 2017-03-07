# Dependency Parser using Neural Networks

This is a continuation of a project that was part of Natural Language Processing (NLP) course at Linköping University.  
In the project we used a Averaged perceptron to classify which transition to use (SHIFT, LEFT-ARC or RIGHT-ARC).  
Instead of using the Multi-class perceptron as a classifier I now want to replace it with a neural network.  
The parser will be based on the parser created by Danqi Chen and Christopher Manning at Stanford University, 2014.  

[A Fast and Accurate Dependency Parser Using Neural Networks](http://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)  

**Further reading:**  
* [Neural Network Dependency Parser, Stanford](http://nlp.stanford.edu/software/nndep.shtml)  
* [Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes4.pdf)  

**Motivation:**   
Using stanfords parser they were able to get a UAS of 92%, LAS of 91% and a parsing speed of 1013 sentences/second.
I wanted to learn how to I could achieve similar results and in the process learn more about Natural language processing.

***

**Structure**  

* Parser - Transition-based dependency parser
* Tagger_POS - Part-of-speech (POS) tagger
* Tagger_arc - Arc-label tagger
* Classifier - Neural Network
* Corpus - English, Conllu format

***
 
**Project structure before any modifcations were made**  

* Parser - Transition-based dependency parser
* Tagger - Part-of-speech (POS) tagger
* Classifier - Multi-class perceptron classifier
* Corpus - English, Conllu format

**Features**  
*The structure of the features used in different parts*

* Tagging features - 
* Classifier features -
* Parsing features - 

**Accuracy**  
*Trained on 4307 sentences*

* Unlabeled Attachment Score (UAS): 60.77%
* Tagging Accuracy: 90.97 %

