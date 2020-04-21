# C472-proj-2
COMP 472 : Naive Bayes Classifier of languages of Twitter messeages (NLP)
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)

GitHub:https://github.com/s-kyy/C472-proj-2

1. Navigate on terminal to directory of main.py
2. Run main.py with the following inputs: "python main.py V n d trainingFileName testingFileName P BYOM"
  
 V : vocabulary type [0: lowercase, 1: upper + lowercase, 2: isalpha()]  
 n : ngram size [1 : unigram, 2 : bigram, 3: trigram]  
 d : smoothing factor, float > 0.0  
 P : [0 : don't include priors, 1: include priors] *For the purpose of analysis in Report; 1 by default.*  
 BYOM : [0 : don't use BYOM, 1: use BYOM] *0 by default*  

Note: To see the F1 metric outputted to the evaluation text file, we made a new branch "F1" that has the code. This was for the sake being able to do the analysis in our report. The submitted code is the same as master. 
