import sys
import os
from NGram import NGram
from BYOM import BYOM
import evaluate

"""
Project 2: Naive Bayes Classifier
COMP 472 NN
DUE: Mar 29th, 2020
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)
Purpose:
This is the ngram model, it contains the frequency table and the conditional probability table to
build scores with a traing set.

Methods:
  verifyArgs() -- This verifies all input arguments from the command line

When running on the command line, use the following form:
    python main.py V SizeOfNGram SmoothingFactor TrainingFile, TestingFile usePriors useBYOM
    eg: python main.py 1 3 0.1 training-tweets.txt test-tweets-given.txt 1 1

"""


def verifyArgs():
    if (not os.path.isfile(sys.argv[4])):
        print('Training file does not exist, terminating program...')
        sys.exit()

    if(not os.path.isfile(sys.argv[5])):
        print('Testing file does not exist, terminating program...')
        sys.exit()

    if sys.argv[1] not in ['0', '1', '2']:
        print('Ivalid vocabulary type, terminating program...')
        sys.exit()

    if sys.argv[2] not in ['1', '2', '3']:
        print('Ivalid ngram size, terminating program...')
        sys.exit()

    try:
        float(sys.argv[3])
    except ValueError:
        print('Ivalid smoothing factor, terminating program...')
        sys.exit()

    if float(sys.argv[3]) > 1 or float(sys.argv[3]) < 0:
        print('Ivalid smoothing factor, terminating program...')
        sys.exit()

    if sys.argv[7] not in ['0', '1']:
        print('Invalid BYOM value, terminating program...')
        sys.exit()


# Main Method
if __name__ == "__main__":
    verifyArgs()
    if sys.argv[7] == '1':
        byom = BYOM(sys.argv[1],
                    float(sys.argv[3]), sys.argv[4], sys.argv[5])

        byom.initialize()

    else:
        ngram = NGram(sys.argv[1], int(sys.argv[2]),
                      float(sys.argv[3]), sys.argv[4], sys.argv[5])

        ngram.initialize()

        evaluate.evaluate(ngram)
