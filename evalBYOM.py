import io
import os
import re
import math
import numpy as np
import pandas as pd
from BYOM import BYOM

"""
Project 2: Naive Bayes Classifier
COMP 472 NN
DUE: April 5th, 2020
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)
Purpose:
    The following code will evaluate our word length based BYOM model (of object BYOM) based on a given test file

Variables
    languages -- list, 6 strings of length 2 representing total language classes modelled
    resultsDF --pandas Dataframe, 6x6 matrix with rows and columns labelled by languages initialized with floating point zeroes

Methods
    computeAcc() -- returns Total examples predicted correctly over total examples in the test file
    computePrecision(TP, TP_FP) -- For a given class, returns True Positives over True Positives + False Positives
    computeRecall(TP, TP_FN) -- For a given class, returns True Positives over True Positives + False Negatives
    computeF1(P, R, B=1) -- returns F1 measurement (see below for formula)
    rowcolSums() -- append two columns in resultsDF containing sum of each row and col
    computeMacroF1() --  return the sum of F1's from each class and divide by the number of classes
    computeWeightedF1() -- return the sum of F1's from each class multiplied by the number of examples labelled as that class
                            divide by the total number of examples in test file
    testLine(ngram, tweet) -- return a tuple of length 2 containing the predicted language (str) and its score (log10 float in scientific notation)
    evaluate(ngram) -- return output text files trace and eval with the help of above methods


Outputs
    A folder called output will be created and two text files can be found within.
    The trace file (ie trace_V_n_d.txt) contains the output of our ngram model's predictions for each example (line) found in the test file. 
    The evaluation file (ie eval_V_n_d.txt) contains the accuracy of our ngram model, precision and recall for each language class, macro F1 measurement 
        as well as the weighted F1 measurment of the language classes combined. 

"""
# TP: True Positive, classified correctly
# FP: False Positive, classified incorrectly as current class
# FN: False Negative, classified incorrectly as another class
#  P: Precision (TP/(TP+FP))
#  R: Recall (TP/(TP+FN))

languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt'] # 6 classes in total

# Initialize dataframe of counts (as floats for downstream computations)
resultsDF = pd.DataFrame(0.0, columns = languages, index= languages)
        # index/rows : predicted classifications; sum the row to calculate Precision
        # columns    : correct / labelled classifications; sum the col to calculate Recall

def computeAcc():
    TP = 0.0
    for lang in languages:
        TP += resultsDF.loc[lang,lang]
    return (TP / computeTotalExamples())

def computePrecision(TP, TP_FP):
    if TP_FP == 0:
        return 0.0
    else:
        return (TP / TP_FP)

def computeRecall(TP, TP_FN):
    if TP_FN == 0:
        return 0.0
    else:
        return (TP / (TP_FN))

def computeF1(P, R, B=1):
    if P == 0 or R == 0:
        return 0.0
    else:
        return ( ( (B**2 + 1)*P*R ) / (B**2*P + R) )
            # { (B^2 + 1)PR }
            # --------------
            # { (B^2)P + R } 

def computeTotalExamples():
    return resultsDF.drop(['rowsums', 'colsums'], axis = 1).values.sum()

def rowcolSums():
    resultsDF['rowsums'] = resultsDF.sum(axis = 1)
    resultsDF['colsums'] = resultsDF.drop('rowsums', axis = 1).sum(axis = 0)
    print(resultsDF)
    return

def computeMacroF1():
    # Average of F1 values from each class divided by the number of classes
    nominator = 0

    for lang in languages:
        P = computePrecision(resultsDF.loc[lang,lang], resultsDF.loc[lang,'rowsums'])
        R = computeRecall(resultsDF.loc[lang,lang], resultsDF.loc[lang,'colsums'])
        if P != 0 or R != 0:
            nominator += computeF1(P, R)
        # if either P or R = 0, then nominator will not be incremented

    return (nominator / 6) 

def computeWeightedF1():
    # Weights determined by number of examples from a given class (columns)
    nominator = 0

    for lang in languages:
        P = computePrecision(resultsDF.loc[lang,lang], resultsDF.loc[lang,'rowsums'])
        R = computeRecall(resultsDF.loc[lang,lang], resultsDF.loc[lang,'colsums'])
        if P != 0 or R != 0:
            nominator += computeF1(P, R) * resultsDF.loc[lang,'colsums'] 
        # if either P or R = 0, then nominator will not be incremented

    denominator = computeTotalExamples()
    return (nominator/denominator)

def testLine(byom, tweet):
    results = ['', 0.0] # predicted language and score
    wordLengthArray = []
    averageWordLengths = 0.0

    totalTrainingExamples = sum(byom.languageCounter.values())

    # Initialize scores with priors
    if byom.prior == False:
        scores = {  'eu':0.0,
                    'ca':0.0, 
                    'gl':0.0, 
                    'es':0.0, 
                    'en':0.0, 
                    'pt':0.0}
    else:
        scores = {  'eu': math.log10(byom.languageCounter['eu']/totalTrainingExamples), 
                    'ca': math.log10(byom.languageCounter['ca']/totalTrainingExamples), 
                    'gl': math.log10(byom.languageCounter['gl']/totalTrainingExamples), 
                    'es': math.log10(byom.languageCounter['es']/totalTrainingExamples), 
                    'en': math.log10(byom.languageCounter['en']/totalTrainingExamples), 
                    'pt': math.log10(byom.languageCounter['pt']/totalTrainingExamples)}
    #print(scores)
    # Compute scores for each language 
    for lang in languages:
        words = re.split(' ', tweet)
        for word in words:
            for char in word:
                # V = 0, lowercase
                if char.lower() in byom.vocabulary:
                    continue
                else:
                    word.replace(char, '')
                # V = 1,2 upper + lower + isalpha
                if char in byom.vocabulary:
                    continue
                else:
                    word.replace(char, '')
            # Either the word doesnt have any valid chars or its too long eg(a random assortment of chars, not a word)
            if(len(word) != 0 and len(word) < 51):
                wordLengthArray.append(len(word))
                if (scores[lang] != 0.0):
                    if (byom.getConditionalProbability('count', lang, len(word)) == 0.0):
                        scores[lang] = 0.0
                    else:
                        scores[lang] += math.log10(byom.getConditionalProbability('count', lang, len(word)))
        # It's possible a tweet doesn't have any valid words
        if wordLengthArray:
            averageWordLengths = int(round(np.mean(wordLengthArray), 0))
            if (scores[lang] != 0.0):
                    if (byom.getConditionalProbability('mean', lang, averageWordLengths) == 0.0):
                        scores[lang] = 0.0
                    else:
                        scores[lang] += math.log10(byom.getConditionalProbability('mean', lang, averageWordLengths))
            stdvWordLengths = int(round(np.std(wordLengthArray), 0))
            if (scores[lang] != 0.0):
                    if (byom.getConditionalProbability('dev', lang, stdvWordLengths) == 0.0):
                        scores[lang] = 0.0
                    else:
                        scores[lang] += math.log10(byom.getConditionalProbability('dev', lang, stdvWordLengths))
        wordLengthArray = []
    # End of score computation

    # Store the language that obtained the highest score.
    results[0] = max(scores, key=scores.get) # get lang with max score
    results[1] = format(max(scores.values()), ".3E") # get max value among all scores
        # score is in log10 scientific notation (rounded to 3 decimals)
    # print(scores)
    # print(results)
    return results
    #End of testLine() function

def evalBYOM(byom):
    traceFileName = str(os.getcwd()+'/output/'+\
                                'traceBYOM_' + str(byom.vocabularyType) +\
                                '_' + str(byom.smoothing) + '.txt')
                                # EX: "trace_V_n_d.txt" in output folder
    os.makedirs(os.path.dirname(traceFileName), exist_ok=True)
    evalFileName = str(os.getcwd()+'/output/'+\
                                'evalBYOM_'+ str(byom.vocabularyType) +\
                                '_' + str(byom.smoothing) + '.txt')
                                # EX: "eval_V_n_d.txt" in output folder
    os.makedirs(os.path.dirname(evalFileName), exist_ok=True)

    # Erase original content of trace file and eval file
    with open(traceFileName, 'w') as trace:
        trace.write('')
    with open(evalFileName, 'w') as eval:
        eval.write('')

    with open(byom.trainingFile, 'r', encoding="utf-8") as train:
        # Example: 439764933119868928	bordatxiki	eu	Oidek mami!!! Jajajjjajaja puta graxioxa

        for line in train:
            params = re.split(r'\t+', line)
            label = params[2] 

            score_results = testLine(byom, params[3])
            resultsDF.loc[score_results[0],label] += 1

            # append results to trace
            with open(traceFileName, 'a', encoding="utf-8") as trace:
                if (score_results[0] == params[2]):
                    trace.write(params[0] + '  ' + score_results[0] + '  ' + score_results[1] + '  ' + label + '  ' + 'correct\n')
                else:
                    trace.write(params[0] + '  ' + score_results[0] + '  ' + score_results[1] + '  ' + label + '  ' + 'wrong\n')
                # "tweetID--classification--score--label--correct/wrong"; - = space
    print("trace file completed")

    # Measure Performance of ngram model
    rowcolSums()

    acc = computeAcc() # Accuracy overall

    precisions = {} # Precision for each language
    recalls = {} # Recall for each language

    for lang in languages:
        precisions[lang] = computePrecision(resultsDF.loc[lang,lang], resultsDF.loc[lang,'rowsums'])
        recalls[lang] = computeRecall(resultsDF.loc[lang,lang], resultsDF.loc[lang,'colsums'])

    macroF1  = computeMacroF1() # Average F1 amongst all language classes
    weightF1 = computeWeightedF1() # Average F1 amongst all languages weighted on frequency of each language occuring in the test data. 

    # Append evaluation measurements to evaluation file
    with open(evalFileName, 'a', encoding="utf-8") as eval:
        eval.write(str(acc) + '\n' +
                    str(precisions['eu']) + '  ' + str(precisions['ca']) + '  ' + str(precisions['gl']) + '  ' + 
                    str(precisions['es']) + '  ' + str(precisions['en']) + '  ' + str(precisions['pt']) + '\n' +
                    str(recalls['eu']) + '  ' + str(recalls['ca']) + '  ' + str(recalls['gl']) + '  ' + 
                    str(recalls['es']) + '  ' + str(recalls['en']) + '  ' + str(recalls['pt']) + '  ' + '\n' +
                    str(macroF1) + '  ' + str(weightF1))
    print("evaluation file completed")

            