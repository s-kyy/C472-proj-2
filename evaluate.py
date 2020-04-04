import io
import os
import re
import math
import pandas as pd
from NGram import NGram

"""
Project 2: Naive Bayes Classifier
COMP 472 NN
DUE: April 5th, 2020
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)
Purpose:
    The following code will evaluate an ngram model (of object NGram) based on a given test file

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

def testLine(ngram, tweet):
    results = ['', 0.0] # predicted language and score

    totalTrainingExamples = sum(ngram.languageCounter.values())

    # Initialize scores with priors, default is True (include priors)
    if ngram.prior == False:
        scores = {  'eu':0.0,
                    'ca':0.0, 
                    'gl':0.0, 
                    'es':0.0, 
                    'en':0.0, 
                    'pt':0.0}
    else:
        scores = {  'eu': math.log10(ngram.languageCounter['eu']/totalTrainingExamples), 
                    'ca': math.log10(ngram.languageCounter['ca']/totalTrainingExamples), 
                    'gl': math.log10(ngram.languageCounter['gl']/totalTrainingExamples), 
                    'es': math.log10(ngram.languageCounter['es']/totalTrainingExamples), 
                    'en': math.log10(ngram.languageCounter['en']/totalTrainingExamples), 
                    'pt': math.log10(ngram.languageCounter['pt']/totalTrainingExamples)}
    #print(scores)
    # Compute scores for each language 
    for lang in languages:
        # Unigram: ngram = 1, 
        if ngram.ngramSize == 1:
            for character in tweet:
                # Vocabulary = 0 (all lowercase)
                if ngram.vocabularyType == '0':
                    if character.lower() in ngram.vocabulary:
                        scores[lang] += math.log10(ngram.getConditionalProbability(lang, character))
                # Vocabulary = 1 (Upper and lowercase)

                if character in ngram.vocabulary:
                    scores[lang] += math.log10(ngram.getConditionalProbability(lang, character))

        # Bigram: ngram = 2,
        if ngram.ngramSize == 2:
            previousChar = None
            for character in tweet:
                # If we've reached an invalid char for V = 0, the next bigram does not count
                if ngram.vocabularyType == '0':
                    if character.lower() not in ngram.vocabulary:
                        previousChar = None
                        continue
                    character = character.lower()
                # If we've reached an invalid char for V = [1,2], the next bigram does not count
                else: 
                    if character not in ngram.vocabulary:
                        previousChar = None
                        continue
                if previousChar is not None:
                    # Vocabulary = 2 (isalpha())
                    if previousChar in ngram.frequencyTable[lang]:
                        scores[lang] += math.log10(ngram.getConditionalProbability(lang, previousChar+character))
                    # Vocabulary = 0 (all lowercase)
                    # Vocabulary = 1 (Upper and lowercase)
                    scores[lang] += math.log10(ngram.getConditionalProbability(lang, previousChar+character))
                previousChar = character
        # Trigram: ngram = 3
        if ngram.ngramSize == 3:
            firstChar = None
            secondChar = None
            for character in tweet:
                # If we've reached an invalid char for V = 0, the next trigram does not count
                if ngram.vocabularyType == '0':
                    if character.lower() not in ngram.vocabulary:
                        firstChar = None
                        secondChar = None
                        continue
                    character = character.lower()
                # If we've reached an invalid char for V = 0, the next trigram does not count
                else:
                    if character not in ngram.vocabulary:
                        firstChar = None 
                        secondChar = None
                        continue
                # Continue building the trigram
                if firstChar is None: 
                    firstChar = character
                    continue
                if secondChar is None:
                    secondChar = character
                    continue
                # Vocabulary = 2 (isalpha())
                # Vocabulary = 0 (all lowercase)
                # Vocabulary = 1 (Upper and lowercase)
                scores[lang] += math.log10(ngram.getConditionalProbability(lang, firstChar+secondChar+character))
                firstChar = secondChar
                secondChar = character
    # End of score computation
   
    # Store the language that obtained the highest score.
    results[0] = max(scores, key=scores.get) # get lang with max score
    results[1] = format(max(scores.values()), ".3E") # get max value among all scores
        # score is in log10 scientific notation (rounded to 3 decimals)
    # print(scores)
    # print(results)
    return results
    #End of testLine() function


def evaluate(ngram):
    traceFileName = str(os.getcwd()+'/output/'+\
                                'trace_' + str(ngram.vocabularyType) +\
                                '_' + str(ngram.ngramSize) +\
                                '_' + str(ngram.smoothing) + '.txt')
                                # EX: "trace_V_n_d.txt" in output folder
    os.makedirs(os.path.dirname(traceFileName), exist_ok=True)
    evalFileName = str(os.getcwd()+'/output/'+\
                                'eval_'+ str(ngram.vocabularyType) +\
                                '_' + str(ngram.ngramSize) +\
                                '_' + str(ngram.smoothing) + '.txt')
                                # EX: "eval_V_n_d.txt" in output folder
    os.makedirs(os.path.dirname(evalFileName), exist_ok=True)

    # Erase original content of trace file and eval file
    with open(traceFileName, 'w') as trace:
        trace.write('')
    with open(evalFileName, 'w') as eval:
        eval.write('')

    # Open the testing file
    with open(ngram.testingFile, 'r', encoding="utf-8") as train:
        # Parse the first line
        for line in train:
            if line == "\n":
                break
            params = []
            params = re.split(r'\t+', line)
            label = params[2] 

            score_results = testLine(ngram, params[3])
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
    # End of evaluate() function
    