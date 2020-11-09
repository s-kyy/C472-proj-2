# Natural Language Prediction of Twitter Posts with Naïve Bayes Classifier & N-grams
COMP 472 Project #2

Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)

GitHub:https://github.com/s-kyy/lang-classifier-for-twitter-posts

## About
Predict if the language of a twitter post is one of the size Latin-based languages: Basque, Catalan, Galician, Spanish, English or Portuguese. An N-gram will be generated based on some hyperparamters (see below). The performance of each model (defined by the hyperparameters) will differ for the same training corpus. 

## How to use the program
1. Navigate on terminal to directory of main.py
2. Run main.py with the following inputs: "python main.py V n d trainingFileName testingFileName P BYOM"

### Hyperparamters
  
 V : vocabulary type [0: lowercase, 1: upper + lowercase, 2: isalpha()]  
 n : ngram size [1 : unigram, 2 : bigram, 3: trigram]  
 d : smoothing factor, float > 0.0  
 P : [0 : don't include priors, 1: include priors] *For the purpose of analysis in Report; 1 by default.*  
 BYOM : [0 : don't use BYOM, 1: use BYOM] *0 by default*  

### Note 
To see the F1 metric outputted to the evaluation text file, we made a new branch "F1" that has the code. This was for the sake being able to do the analysis in our report. The submitted code is the same as master. 
