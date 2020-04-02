import io
import re

"""
Project 2: Naive Bayes Classifier
COMP 472 NN
DUE: April 5th, 2020
Samantha Yuen (40033121), Andrew Marcos (40011252), Michael Gagnon (40030481)
Purpose:
This is the ngram model, it contains the frequency table and the conditional probability table to
build scores with a training set.

Object Variables
    vocabularyType (specifies which letters are in our vocabulary)
    ngramSize (specifies the size of each ngram)
    smoothing (specifies the smoothing factor for empty entries)
    trainingFile (specifies the name of the training file)
    testingFile (specifies the name of the testing file)
    vocabulary (specifies all valid characters in our vocabulary)
    frequencyTable (contains the frequencies of each ngram)
    conditionalProbabilityTable (contains the conditional probs of all ngrams)
    languageCounter (specifies how many tweets come from each language)
    totalRowCount (this is to keep track of the isalpha vocab probabilities,
    since we're building it dynamically we need to keep track of how many ngrams appear in each row so we can find the denominator)

Methods:
  constructVocabulary() -- This creates all of the letters that can be used with the vocabulary type.
  constructFrequencyTable() -- This constructs all of the entries for type of ngram used.
  populateFrequencyTable() -- This reads a training set and saves the frequencies of each ngram.
  smoothFrequencyTable() -- This adds a smoothing factor to all of the entries that are 0.
  createConditionalProbabilityTable() -- This translates the frequency table to a conditional probability table.
  initialize() -- This calls all of the methods required to set up the ngram models.
  createIsAlphaProbabilityTable() -- This creates the dynamic probability table for the isalhpa vocabulary
  getConditionalProbability(lang, ngram) -- This takes the language and the ngram, and returns (or creates) the probability for that ngram


Example Formats of frequency & conditional probability tables
 - unigram: self.frequencyTable['en']['a'] = integer value
 - bigram: self.frequencyTable['es']['a']['b'] = integer value
 - trigram: self.frequencyTable['fr']['f']['d']['e'] = integer value

 - unigram: self.conditionalProbabilityTable['en']['a'] = integer value
 - bigram: self.conditionalProbabilityTable['es']['a']['b'] = integer value
 - trigram: self.conditionalProbabilityTable['fr']['f']['d']['e'] = integer value
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
        self.conditionalProbabilityTable = {}
        self.languageCounter = {}
        self.totalRowCount = 0

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
            # We need to keep track of how many ngrams we have not seen to calculate probabilites
            # count = count ** self.ngramSize
            self.totalRowCount = count

    def constructFrequencyTable(self):
        frequencyTable = {}
        # if we have the isalpha vocab, we cannot hardcode the freq table (it's too big!), it needs to be dynamic
        if self.vocabularyType != '2':
            # Loop over vocab 1, 2, or 3 times depending on ngram size to build freq table
            for x in self.vocabulary:
                frequencyTable[x] = self.smoothing
                if self.ngramSize > 1:
                    frequencyTable[x] = {}
                    for y in self.vocabulary:
                        frequencyTable[x][y] = self.smoothing
                        if self.ngramSize > 2:
                            frequencyTable[x][y] = {}
                            for z in self.vocabulary:
                                frequencyTable[x][y][z] = self.smoothing
        self.frequencyTable = frequencyTable

    def populateFrequencyTable(self):
        with open(self.trainingFile, 'r', encoding="utf-8") as train:
            innerFrequencyTable = self.frequencyTable
            self.frequencyTable = {}
            # Example: 439764933119868928	bordatxiki	eu	Oidek mami!!! Jajajjjajaja puta graxioxa
            for line in train:
                params = re.split(r'\t+', line)
                language = params[2]
                if language in self.frequencyTable:
                    self.languageCounter[language] += 1
                else:
                    # First time we've encountered the language, add it to frequency and prob table
                    self.conditionalProbabilityTable[language] = {}
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
                        elif self.vocabularyType == '2':
                            if character in self.vocabulary:
                                # Since this vocab type is dynamic, we need to make sure it exists first.
                                if character in self.frequencyTable[language]:
                                    self.frequencyTable[language][character] += 1
                                else:
                                    self.frequencyTable[language][character] = self.smoothing + 1
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
                            # We need to build this vocab dynamically, check if it exists first
                            if self.vocabularyType == '2':
                                if previousChar not in self.frequencyTable[language]:
                                    self.frequencyTable[language][previousChar] = {
                                    }
                                    self.frequencyTable[language][previousChar][character] = self.smoothing
                                elif character not in self.frequencyTable[language][previousChar]:
                                    self.frequencyTable[language][previousChar][character] = self.smoothing
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
                            # We need to build this vocab dynamically, check if it exists first
                        if self.vocabularyType == '2':
                            if firstChar not in self.frequencyTable[language]:
                                self.frequencyTable[language][firstChar] = {
                                }
                                self.frequencyTable[language][firstChar][secondChar] = {
                                }
                                self.frequencyTable[language][firstChar][secondChar][character] = self.smoothing
                            elif secondChar not in self.frequencyTable[language][firstChar]:
                                self.frequencyTable[language][firstChar][secondChar] = {
                                }
                                self.frequencyTable[language][firstChar][secondChar][character] = self.smoothing
                            elif character not in self.frequencyTable[language][firstChar][secondChar]:
                                self.frequencyTable[language][firstChar][secondChar][character] = self.smoothing
                        self.frequencyTable[language][firstChar][secondChar][character] += 1
                        # We save the two previous chars and move on to next char
                        firstChar = secondChar
                        secondChar = character

    def createConditionalProbabilityTable(self):
        if self.vocabularyType == '2':
            self.createIsAlphaProbabilityTable()
        # Else do the regular translation of the frequency table
        else:
            if self.ngramSize == 3:
                for lang in self.frequencyTable:
                    for first in self.frequencyTable[lang]:
                        self.conditionalProbabilityTable[lang][first] = {}
                        for second in self.frequencyTable[lang][first]:
                            self.conditionalProbabilityTable[lang][first][second] = {
                            }

                            denominator = sum(
                                self.frequencyTable[lang][first][second].values()) 
                            for third in self.frequencyTable[lang][first][second]:
                                numerator = self.frequencyTable[lang][first][second][third]
                                conditionalProb = numerator/denominator
                                self.conditionalProbabilityTable[lang][first][second][third] = conditionalProb
            if self.ngramSize == 2:
                for lang in self.frequencyTable:
                    for first in self.frequencyTable[lang]:
                        self.conditionalProbabilityTable[lang][first] = {}

                        denominator = sum(
                            self.frequencyTable[lang][first].values()) 
                        for second in self.frequencyTable[lang][first]:
                            numerator = self.frequencyTable[lang][first][second]
                            conditionalProb = numerator/denominator
                            self.conditionalProbabilityTable[lang][first][second] = conditionalProb
            if self.ngramSize == 1:
                for lang in self.frequencyTable:
                    denominator = sum(
                        self.frequencyTable[lang].values()) 
                    for first in self.frequencyTable[lang]:
                        numerator = self.frequencyTable[lang][first]
                        conditionalProb = numerator/denominator
                        self.conditionalProbabilityTable[lang][first] = conditionalProb


    def createIsAlphaProbabilityTable(self):
        # Since this ngram model has a lot of empty entries, we need to compute the denominator dynamically.
        # We need to find the amount of characters that are not entered, and multiply by smoothing factor
        # and add this to the sum() of the appeared characters. This will give us the denom for each row.
        if self.ngramSize == 3:
            for lang in self.frequencyTable:
                for first in self.frequencyTable[lang]:
                    self.conditionalProbabilityTable[lang][first] = {}
                    for second in self.frequencyTable[lang][first]:
                        self.conditionalProbabilityTable[lang][first][second] = {
                        }
                        actualRowCount = len(
                            self.frequencyTable[lang][first][second])
                        amountOfUnseenChars = self.totalRowCount - actualRowCount
                        denominator = sum(
                            self.frequencyTable[lang][first][second].values()) + (amountOfUnseenChars * self.smoothing)
                        for third in self.frequencyTable[lang][first][second]:
                            numerator = self.frequencyTable[lang][first][second][third]
                            conditionalProb = numerator/denominator
                            self.conditionalProbabilityTable[lang][first][second][third] = conditionalProb
        if self.ngramSize == 2:
            for lang in self.frequencyTable:
                for first in self.frequencyTable[lang]:
                    self.conditionalProbabilityTable[lang][first] = {}
                    actualRowCount = len(
                        self.frequencyTable[lang][first])
                    amountOfUnseenChars = self.totalRowCount - actualRowCount
                    denominator = sum(
                        self.frequencyTable[lang][first].values()) + (amountOfUnseenChars * self.smoothing)
                    for second in self.frequencyTable[lang][first]:
                        numerator = self.frequencyTable[lang][first][second]
                        conditionalProb = numerator/denominator
                        self.conditionalProbabilityTable[lang][first][second] = conditionalProb
        if self.ngramSize == 1:
            for lang in self.frequencyTable:
                actualRowCount = len(
                    self.frequencyTable[lang])
                amountOfUnseenChars = self.totalRowCount - actualRowCount
                denominator = sum(
                    self.frequencyTable[lang].values()) + (amountOfUnseenChars * self.smoothing)
                for first in self.frequencyTable[lang]:
                    numerator = self.frequencyTable[lang][first]
                    conditionalProb = numerator/denominator
                    self.conditionalProbabilityTable[lang][first] = conditionalProb

    def getConditionalProbability(self, lang, ngram):
        # This has to dynamically create probabilities for entries not in the model for isalpha
        # This is using the same logic as the createIsAlphaProbabilityTable, where if a ngram
        # does not appear in the model then we need to generates its probability dynamically.
        if self.ngramSize == 3:
            if ngram[0] in self.conditionalProbabilityTable[lang]:
                if ngram[1] in self.conditionalProbabilityTable[lang][ngram[0]]:
                    if ngram[2] in self.conditionalProbabilityTable[lang][ngram[0]][ngram[1]]:
                        return self.conditionalProbabilityTable[lang][ngram[0]][ngram[1]][ngram[2]]
                    actualRowCount = len(
                        self.frequencyTable[lang][ngram[0]][ngram[1]])
                    amountOfUnseenChars = self.totalRowCount - actualRowCount
                    denominator = sum(
                        self.frequencyTable[lang][ngram[0]][ngram[1]].values()) + (amountOfUnseenChars * self.smoothing)
                    return self.smoothing / denominator
                return self.smoothing / (self.totalRowCount * self.smoothing)
            return self.smoothing / (self.totalRowCount * self.smoothing)
        if self.ngramSize == 2:
            if ngram[0] in self.conditionalProbabilityTable[lang]:
                if ngram[1] in self.conditionalProbabilityTable[lang][ngram[0]]:
                    return self.conditionalProbabilityTable[lang][ngram[0]][ngram[1]]
                actualRowCount = len(
                    self.frequencyTable[lang][ngram[0]])
                amountOfUnseenChars = self.totalRowCount - actualRowCount
                denominator = sum(
                    self.frequencyTable[lang][ngram[0]].values()) + (amountOfUnseenChars * self.smoothing)
                return self.smoothing / denominator
            return self.smoothing / (self.totalRowCount * self.smoothing)
        if self.ngramSize == 1:
            if ngram[0] in self.conditionalProbabilityTable[lang]:
                return self.conditionalProbabilityTable[lang][ngram[0]]
            actualRowCount = len(
                self.frequencyTable[lang])
            amountOfUnseenChars = self.totalRowCount - actualRowCount
            denominator = sum(
                self.frequencyTable[lang].values()) + (amountOfUnseenChars * self.smoothing)
            return self.smoothing / denominator

    def initialize(self):

        self.constructVocabulary()

        # If you uncomment this make sure your console supports utf-16 or it will error
        #for x in self.vocabulary:
         #    print(x)

        self.constructFrequencyTable()

        #for x in self.frequencyTable:
         #    print(x)
          #   for y in self.frequencyTable[x]:
           #      print(x+y)
            #     for z in self.frequencyTable[x][y]:
             #        print(x+y+z)

        self.populateFrequencyTable()

        self.createConditionalProbabilityTable()

        #print(self.frequencyTable)

        #print(sum(self.frequencyTable['es'].values()))

        #print(len(self.frequencyTable['es']))

        #print(self.conditionalProbabilityTable)

        #print(self.getConditionalProbability('es', 'ç'))

        #print(self.getConditionalProbability('es', 'cao'))

        #for lang in self.frequencyTable:
         #    for x in self.frequencyTable[lang]:
          #       for y in self.frequencyTable[lang][x]:
           #          for z in self.frequencyTable[lang][x][y]:
            #             if(self.frequencyTable[lang][x][y][z] > 0):
             #                print()
              #               print(x+y+z + ' --- ' +
               #                    str(self.frequencyTable[lang][x][y][z]))
