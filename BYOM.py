import re
import io
import numpy as np

"""
Project 2: Naive Bayes Classifier
COMP 472 NN
DUE: April 5th, 2020
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)
Purpose:
This is the Build Your Own Model, it contains the necessary functions to build a probability 
table based on the length of words in tweets, the mean and the standard deviation of the
word length across a tweet. Besides the ordering of letters, another characterstic of 
any language is the length of the words that it uses. For example, on average the length of
english words is much lower than the length of german words because of the nature of nouns in german. This
experiment would analyze if these differences are obvious in other languages as well.
Reference: https://en.wikipedia.org/wiki/German_nouns


Object Variables
    vocabularyType (specifies which letters are in our vocabulary)
    smoothing (specifies the smoothing factor for empty entries)
    trainingFile (specifies the name of the training file)
    testingFile (specifies the name of the testing file)
    vocabulary (specifies all valid characters in our vocabulary)
    frequencyTable (contains the frequencies of each ngram)
    conditionalProbabilityTable (contains the conditional probs of all ngrams)
    languageCounter (specifies how many tweets come from each language)

Methods:
  constructVocabulary() -- This creates all of the letters that can be used with the vocabulary type.
  constructFrequencyTable() -- This constructs all of the entries for type of metrics used.
  populateFrequencyTable() -- This reads a training set and saves the frequencies of each metric.
  createConditionalProbabilityTable() -- This translates the frequency table to a conditional probability table.
  initialize() -- This calls all of the methods required to set up the metric models.
  getConditionalProbability(metric, lang, number) -- This takes the language and the metric and the count value, and returns the probability for that metric


Example Formats of frequency & conditional probability tables
 - count: self.frequencyTable['count']['es'][6] = integer value
 - mean: self.frequencyTable['mean']['es'][6] = integer value
 - deviation: self.frequencyTable['dev']['es'][6] = integer value

 - count: self.conditionalProbabilityTable['count']['es'][6] = integer value
 - mean: self.conditionalProbabilityTable['mean']['es'][6] = integer value
 - deviation: self.conditionalProbabilityTable['dev']['es'][6] = integer value
"""


class BYOM:
    def __init__(self, vocabularyType, smoothing, trainingFile, testingFile, prior = True):
        self.vocabularyType = vocabularyType
        self.trainingFile = trainingFile
        self.testingFile = testingFile
        self.smoothing = smoothing
        self.vocabulary = {}
        self.prior = prior
        self.frequencyTable = {}
        self.conditionalProbabilityTable = {}
        self.languageCounter = {}

    def constructVocabulary(self):

        if self.vocabularyType == '0':
            # chr(97) = a
            for x in range(26):
                chrIndex = 97 + x
                self.vocabulary[chr(chrIndex)] = True

        if self.vocabularyType == '1':
            # chr(65) = A and chr(97) = a
            for x in range(26):
                chrIndex = 65 + x
                self.vocabulary[chr(chrIndex)] = True
            for x in range(26):
                chrIndex = 97 + x
                self.vocabulary[chr(chrIndex)] = True

        if self.vocabularyType == '2':
            # Taken (with modifications) from Prof. Kosseim Moodle Q&A Page
            # Her moodle page says 116766 but the exact same code below returns 125419 (out-of-date?) using Python 3.7.0
            count = 0
            # unicode = 17 planes of 2**16 symbols
            for codepoint in range(17 * 2**16):
                ch = chr(codepoint)
                if ch.isalpha():
                    self.vocabulary[ch] = True
                    count = count + 1
            # End of Prof. Kosseim Moodle Q&A Page

    def constructFrequencyTable(self):
        frequencyTable = {}
        languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
        frequencyTable['count'] = {}
        frequencyTable['mean'] = {}
        frequencyTable['dev'] = {}
        # Loop over vocab 1, 2, or 3 times depending on ngram size to build freq table
        for lang in languages:
            frequencyTable['count'][lang] = {}
            frequencyTable['mean'][lang] = {}
            frequencyTable['dev'][lang] = {}
            # No ordinary word is longer than 50 characters
            for x in range(1, 51):
                frequencyTable['count'][lang][x] = self.smoothing
                frequencyTable['mean'][lang][x] = self.smoothing
                frequencyTable['dev'][lang][x] = self.smoothing
            # It's possible the deviation is 0 if all words are same length
            frequencyTable['dev'][lang][0] = self.smoothing

        self.frequencyTable = frequencyTable

    def populateFrequencyTable(self):
        with open(self.trainingFile, 'r', encoding="utf-8") as train:
            # Example: 439764933119868928	bordatxiki	eu	Oidek mami!!! Jajajjjajaja puta graxioxa
            wordLengthArray = []
            averageWordLengths = 0.0

            for line in train:
                params = re.split(r'\t+', line)
                language = params[2]
                if language in self.languageCounter:
                    self.languageCounter[language] += 1
                else:
                    # First time we've encountered the language, add it to frequency and prob table
                    self.languageCounter[language] = 1

                tweet = params[3]
                words = re.split(' ', tweet)
                for word in words:
                    for char in word:
                        # V = 0, lowercase
                        if char.lower() in self.vocabulary:
                            continue
                        else:
                            word.replace(char, '')
                        # V = 1,2 upper + lower + isalpha
                        if char in self.vocabulary:
                            continue
                        else:
                            word.replace(char, '')
                    # Either the word doesnt have any valid chars or its too long eg(a random assortment of chars, not a word)
                    if(len(word) != 0 and len(word) < 51):
                        wordLengthArray.append(len(word))
                        self.frequencyTable['count'][language][len(word)] += 1
                # It's possible a tweet doesn't have any valid words
                if wordLengthArray:
                    averageWordLengths = np.mean(wordLengthArray)
                    self.frequencyTable['mean'][language][int(
                        round(averageWordLengths, 0))] += 1
                    stdvWordLengths = np.std(wordLengthArray)
                    self.frequencyTable['dev'][language][int(
                        round(stdvWordLengths, 0))] += 1
                wordLengthArray = []

    def createConditionalProbabilityTable(self):
        self.conditionalProbabilityTable['count'] = {}
        self.conditionalProbabilityTable['mean'] = {}
        self.conditionalProbabilityTable['dev'] = {}
        for lang in self.frequencyTable['count']:
            self.conditionalProbabilityTable['count'][lang] = {}
            self.conditionalProbabilityTable['mean'][lang] = {}
            self.conditionalProbabilityTable['dev'][lang] = {}
            denominatorCount = sum(
                self.frequencyTable['count'][lang].values())
            denominatorMean = sum(
                self.frequencyTable['mean'][lang].values())
            denominatorDev = sum(
                self.frequencyTable['dev'][lang].values())
            for num in self.frequencyTable['count'][lang]:
                numeratorCount = self.frequencyTable['count'][lang][num]
                numeratorMean = self.frequencyTable['mean'][lang][num]
                numeratorDev = self.frequencyTable['dev'][lang][num]
                conditionalProbCount = numeratorCount/denominatorCount
                conditionalProbMean = numeratorMean/denominatorMean
                conditionalProbDev = numeratorDev/denominatorDev

                self.conditionalProbabilityTable['count'][lang][num] = conditionalProbCount
                self.conditionalProbabilityTable['mean'][lang][num] = conditionalProbMean
                self.conditionalProbabilityTable['dev'][lang][num] = conditionalProbDev
            # the dev metric has one extra row for 0
            self.conditionalProbabilityTable['dev'][lang][0] = self.frequencyTable['dev'][lang][0]/denominatorDev

    def getConditionalProbability(self, check, lang, num):
        return self.conditionalProbabilityTable[check][lang][num]

    def initialize(self):

        self.constructVocabulary()

        self.constructFrequencyTable()

        self.populateFrequencyTable()

        self.createConditionalProbabilityTable()

        for countType in self.conditionalProbabilityTable:
            for lang in self.conditionalProbabilityTable[countType]:
                for num in self.conditionalProbabilityTable[countType][lang]:
                    print(countType + ' ' + lang + ' ' + str(num) +
                          ': ' + str(self.conditionalProbabilityTable[countType][lang][num]))
