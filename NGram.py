import io
import re

""" 
Project 2: Naive Bayes Classifier
COMP 472 NN 
DUE: Mar 29th, 2020
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)
Purpose:
This is the ngram model, it contains the frequency table and the conditional probability table to
build scores with a traing set.

Object Variables
    vocabularyType (specifies which letters are in our vocabulary)
    ngramSize (specifies the size of each ngram)
    smoothing (specifies the smoothing factor for empty entries)
    trainingFile (specifies the name of the training file)
    testingFile (specifies the name of the testing file)
    vocabulary (specifies all valid characters in our vocabulary)
    frequencyTable (contains the frequencies of each ngram)
    languageCounter (specifies how many tweets come from each language)

Methods:
  constructVocabulary() -- This creates all of the letters that can be used with the vocabulary type.
  constructFrequencyTable() -- This constructs all of the entries for type of ngram used.
  populateFrequencyTable() -- This reads a training set and saves the frequencies of each ngram.
  smoothFrequencyTable() -- This adds a smoothing factor to all of the entries that are 0.
  createConditionalProbabilityTable() -- This translates the frequency table to a conditional probability table.
  initialize() -- This calls all of the methods required to set up the ngram models.

Example Formats of frequency table
 - unigram: self.frequencyTable['en']['a'] = integer value
 - bigram: self.frequencyTable['es']['a']['b'] = integer value
 - trigram: self.frequencyTable['fr']['f']['d']['e'] = integer value
"""


class NGram:
    def __init__(self, vocabularyType, ngramSize, smoothing, trainingFile, testingFile):
        self.vocabularyType = vocabularyType
        self.ngramSize = ngramSize
        self.smoothing = smoothing
        self.trainingFile = trainingFile
        self.testingFile = testingFile
        self.vocabulary = {}
        self.frequencyTable = {}
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
        # Loop over vocab 1, 2, or 3 times depending on ngram size to ubild freq table
        for x in self.vocabulary:
            frequencyTable[x] = 0
            if self.ngramSize > 1:
                frequencyTable[x] = {}
                for y in self.vocabulary:
                    frequencyTable[x][y] = 0
                    if self.ngramSize > 2:
                        frequencyTable[x][y] = {}
                        for z in self.vocabulary:
                            frequencyTable[x][y][z] = 0
        self.frequencyTable = frequencyTable

    def populateFrequencyTable(self):
        with open(self.trainingFile, 'r', encoding="utf-8") as train:
            innerFrequencyTable = self.frequencyTable
            self.frequencyTable = {}
            # Example: 439764933119868928	bordatxiki	eu	Oidek mami!!! Jajajjjajaja puta graxioxa
            for line in train:
                params = re.split(r'\t+', line)
                language = params[2]
                if(language in self.frequencyTable):
                    self.languageCounter[language] += 1
                else:
                    # First time we've encountered the language, add it to frequency table
                    self.frequencyTable[language] = innerFrequencyTable
                    self.languageCounter[language] = 1

                tweet = params[3]
                if self.ngramSize == 1:
                    for character in tweet:
                        # If V = 1, we only care about lowercase chars
                        if self.vocabularyType == '0':
                            if character.lower() in self.vocabulary:
                                self.frequencyTable[language][character.lower(
                                )] += 1
                        else:
                            if character in self.vocabulary:
                                self.frequencyTable[language][character] += 1
                if self.ngramSize == 2:
                    previousChar = None
                    for character in tweet:
                        if self.vocabularyType == '0':
                            # If we've reached an invalid char, the next bigram does not count
                            if character.lower() not in self.vocabulary:
                                previousChar = None
                                continue
                            character = character.lower()
                        else:
                            # If we've reached an invalid char, the next bigram does not count
                            if character not in self.vocabulary:
                                previousChar = None
                                continue
                        # If we cant build a bigram dont increment, save char and go next
                        if previousChar is not None:
                            self.frequencyTable[language][previousChar][character] += 1
                        previousChar = character

                if self.ngramSize == 3:
                    firstChar = None
                    secondChar = None
                    for character in tweet:
                        if self.vocabularyType == '0':
                            # If we've reached an invalid char, the next trigram does not count
                            if character.lower() not in self.vocabulary:
                                firstChar = None
                                secondChar = None
                                continue
                            character = character.lower()
                        else:
                            # If we've reached an invalid char, the next trigram does not count
                            if character not in self.vocabulary:
                                firstChar = None
                                secondChar = None
                                continue
                        # We cant build a trigram yet, store this char and go to next char
                        if firstChar is None:
                            firstChar = character
                            continue
                        if secondChar is None:
                            secondChar = character
                            continue

                        self.frequencyTable[language][firstChar][secondChar][character] += 1
                        # We save the two previous chars and move on to next char
                        firstChar = secondChar
                        secondChar = character

    def smoothFrequencyTable(self):
        # This is just looping over all entries and adding 1 if they're 0.
        if self.ngramSize == 1:
            for lang in self.frequencyTable:
                for character in self.frequencyTable[lang]:
                    if self.frequencyTable[lang][character] == 0:
                        self.frequencyTable[lang][character] += self.smoothing
        if self.ngramSize == 2:
            for lang in self.frequencyTable:
                for firstChar in self.frequencyTable[lang]:
                    for secondChar in self.frequencyTable[lang][firstChar]:
                        if self.frequencyTable[lang][firstChar][secondChar] == 0:
                            self.frequencyTable[lang][firstChar][secondChar] += self.smoothing
        if self.ngramSize == 3:
            for lang in self.frequencyTable:
                for firstChar in self.frequencyTable[lang]:
                    for secondChar in self.frequencyTable[lang][firstChar]:
                        for thirdChar in self.frequencyTable[lang][firstChar][secondChar]:
                            if self.frequencyTable[lang][firstChar][secondChar][thirdChar] == 0:
                                self.frequencyTable[lang][firstChar][secondChar][thirdChar] += self.smoothing

    def createConditionalProbabilityTable(self):
        pass

    def initialize(self):

        self.constructVocabulary()

        # If you uncomment this make sure your console supports utf-8 or it will error
        # for x in self.vocabulary:
        #     print(x)

        self.constructFrequencyTable()

        # for x in self.frequencyTable:
        #     print(x)
        #     for y in self.frequencyTable[x]:
        #         print(x+y)
        #         for z in self.frequencyTable[x][y]:
        #             print(x+y+z)

        self.populateFrequencyTable()

        # for lang in self.frequencyTable:
        #     for x in self.frequencyTable[lang]:
        #         for y in self.frequencyTable[lang][x]:
        #             for z in self.frequencyTable[lang][x][y]:
        #                 if(self.frequencyTable[lang][x][y][z] > 0):
        #                     print()
        #                     print(x+y+z + ' --- ' +
        #                           str(self.frequencyTable[lang][x][y][z]))

        self.smoothFrequencyTable()

        # for lang in self.frequencyTable:
        #     for x in self.frequencyTable[lang]:
        #         for y in self.frequencyTable[lang][x]:
        #             for z in self.frequencyTable[lang][x][y]:
        #                 if(self.frequencyTable[lang][x][y][z] > 0):
        #                     print()
        #                     print(x+y+z + ' --- ' +
        #                           str(self.frequencyTable[lang][x][y][z]))
