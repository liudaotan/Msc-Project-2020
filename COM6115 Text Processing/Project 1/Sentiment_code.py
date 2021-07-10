#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=1

#------------- Function Definitions ---------------------

#%%
def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):
    
    random.seed(46)
    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1
  
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#%%
#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0


    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary): 
                dictionary[word] = 1
            if sentiment=="positive" : 
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive): 
                    freqPositive[word] = 1 
                else:
                    freqPositive[word] += 1 
            else:
                negWordsTot+=1
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary: 
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative): 
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------
#%%
def performances(correct,total,correctpos,totalpospred,totalpos,correctneg,totalnegpred,totalneg,dataName):
    accuracy = correct/total
    precision_pos = correctpos/totalpospred
    recall_pos = correctpos/totalpos
    precision_neg = correctneg/totalnegpred
    recall_neg = correctneg/totalneg
    F1_pos = (2*precision_pos*recall_pos)/(precision_pos+recall_pos)
    F1_neg =(2*precision_neg * recall_neg)/(precision_neg + recall_neg)
    
    print('----------------------------------------')
    print(' The performance of NaiveBayes on the %s'%dataName)
    print('(1) Accuracy : %0.2f'%accuracy)   
    print('(2) Precision(positive): %0.2f, Recall(positive): %0.2f'%(precision_pos,recall_pos))
    print('(3) Precision(negative): %0.2f, Recall(negative): %0.2f'%(precision_neg,recall_neg))
    print('(4) F1 score(positive): %0.2f, F1 score(negative): %0.2f'%(F1_pos,F1_neg)) 

#%%
#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset

def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results (you do not need them)
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW) # 

        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score; 
       
    performances(correct,total,correctpos,totalpospred,totalpos,correctneg,totalnegpred,totalneg,dataName)
 

#%%
# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0 
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word] 
 
        total+=1
        if sentiment=="positive":
            totalpos+=1 
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1 
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
 
    
# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
 

       
    performances(correct,total,correctpos,totalpospred,totalpos,correctneg,totalnegpred,totalneg,dataName)

#%%

# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get) 
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:] 
    
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    return head,tail

#%%
#---------- Main Script --------------------------
sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)


#run naive bayes classifier on datasets
print ("Naive Bayes")

# STEP 2 Q3
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)

# STEP 3 Q1
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)

# STEP 4
# Q1
head,tail = mostUseful(pWordPos, pWordNeg, pWord, 100)
# Q2
head,tail = mostUseful(pWordPos, pWordNeg, pWord, 50)


 
# How many are in the sentiment dictionary? 
head_num = 0
tail_num = 0
for word in head:
    if word in sentimentDictionary.keys():
        head_num +=1

for word in tail:
    if word in sentimentDictionary.keys():
        tail_num +=1

print('There are %d most useful negative words and %d most\
      useful positive words in the sentiment dictionary.'%(head_num,tail_num))

# STEP 5

# Q1 
#run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, -3)

#%%
# Q3 The improved new rule-based function
def improved_testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
       
        weight = 1
        for word in Words:
            # Negation rule
            if word == 'not':
                weight *= -1
            #Intensifier rule
            if word in ['definitely','very','extremely']:
                weight *= 2
            # diminisher rule
            if word in ['somewhat','barely','rarely']:
                weight *= 0.5
            if word == '!':
                weight *= 2
            # Exclamation rule    
            if word in sentimentDictionary:
                #Capitaization rule
                if word.isupper():
                    word = word.lower()
                    score+=2*weight*sentimentDictionary[word] 
                else:   
                    score+=weight*sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                
    performances(correct,total,correctpos,totalpospred,totalpos,correctneg,totalnegpred,totalneg,dataName)

#%%
# The performence of improved rule-based system
improved_testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
improved_testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)
improved_testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, -3)


