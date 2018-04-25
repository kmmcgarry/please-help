'''
Kristen McGarry
Last Updated: February 23, 2018
'''

import nltk
import string
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer



def splitData(fileName):
    categoriesIntoFiles = {}
    csvDict={'0':[],'1':[]}
    with open (fileName,'rb') as tsvfile:
        reader = csv.DictReader(tsvfile)

        for row in reader:
            comment_id = row['comment_post_id']
            label = row['label']
            content = row['comment_content']
            csvDict[label].append([comment_id,content])
    # print len(csvDict['0'])
    # print len(csvDict['1'])
    return csvDict


def wordFreq(trainDict,testDict):
    trainText1 = ""
    trainText0 = ""

    for i in trainDict['0']:
        trainText0 += (" " + i[1])
    for i in trainDict['1']:
        trainText1 += (" " + i[1])

    textTrain1 = nltk.word_tokenize(trainText1)
    textTrain0 = nltk.word_tokenize(trainText0)

    print len(textTrain0)
    print len(textTrain1)


    real_words_1 = []
    real_words_0 = []

    for word in textTrain1:
        if word.isalpha():
            real_words_1.append(word)
        else:
            continue

    for word in textTrain0:
        if word.isalpha():
            real_words_0.append(word)
        else:
            continue

    dict1 = nltk.FreqDist(real_words_1)
    dict0 = nltk.FreqDist(real_words_0)


    #print("WORD FREQUENCIES:")
    sorted1 = sorted(dict1.items(), key=lambda tup: tup[1])
    sorted0 = sorted(dict0.items(), key=lambda tup: tup[1])

    #print sorted1
    #print sorted0

# Feature extraction from text
# Method: bag of words (bow)
def model1(trainDict,testDict):
    train = {"data":[],"target":[],"commentID":[]}
    for i in trainDict['0']:
        train["target"].append(0)
        train["data"].append(i[1])
        train["commentID"].append(i[0])
    for i in trainDict['1']:
        train["target"].append(1)
        train["data"].append(i[1])
        train["commentID"].append(i[0])


    new_comments = []
    for i in testDict['1']:
        new_comments.append([i[0],i[1],1])
    for i in testDict['0']:
        new_comments.append([i[0],i[1],0])

    random.shuffle(new_comments)

    testLabels = []
    testCommentID = []
    testComment = []
    for i in new_comments:
        testCommentID.append(i[0])
        testComment.append(i[1])
        testLabels.append(i[2])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train["data"])
    #print X_train_counts.shape
    # uses each word as a feature
    print count_vect.get_feature_names()

    clf = MultinomialNB().fit(X_train_counts, train["target"])


    text_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])

    text_clf = text_clf.fit(train["data"], train["target"])

    predicted = text_clf.predict(testComment)
    print np.mean(predicted == testLabels)
    #count of 1/ len predicted
    predicted = predicted.tolist()
    print (predicted)
    print (testLabels)

    precision =  (predicted.count(1)*1.0) / (len(predicted))
    recall = (predicted.count(1)*1.0)/ (testLabels.count(1))



    print precision

    print recall
    print (2*precision*recall)/(precision+recall)

    print classification_report(testLabels, predicted)

# Model: BOW with TF/IDF
def model2(trainDict,testDict):
    train = {"data":[],"target":[],"commentID":[]}
    for i in trainDict['0']:
        train["target"].append(0)
        train["data"].append(i[1])
        train["commentID"].append(i[0])
    for i in trainDict['1']:
        train["target"].append(1)
        train["data"].append(i[1])
        train["commentID"].append(i[0])


    new_comments = []
    for i in testDict['1']:
        new_comments.append([i[0],i[1],1])
    for i in testDict['0']:
        new_comments.append([i[0],i[1],0])

    random.shuffle(new_comments)

    testLabels = []
    testCommentID = []
    testComment = []
    for i in new_comments:
        testCommentID.append(i[0])
        testComment.append(i[1])
        testLabels.append(i[2])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train["data"])
    #print X_train_counts.shape
    # uses each word as a feature
    print count_vect.get_feature_names()

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape


    clf = MultinomialNB().fit(X_train_counts, train["target"])


    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

    text_clf = text_clf.fit(train["data"], train["target"])

    predicted = text_clf.predict(testComment)

    print np.mean(predicted == testLabels)
    #count of 1/ len predicted
    predicted = predicted.tolist()
    print (predicted)
    print (testLabels)

    precision =  (predicted.count(1)*1.0) / (len(predicted))
    recall = (predicted.count(1)*1.0)/ (testLabels.count(1))



    print precision

    print recall
    print (2*precision*recall)/(precision+recall)

    print classification_report(testLabels, predicted)

testCsvDict = splitData('/Users/kristen/Desktop/rand_doc_test.csv')
trainCsvDict = splitData('/Users/kristen/Desktop/rand_doc_train.csv')
wordFreq(trainCsvDict,testCsvDict)
#model1(trainCsvDict,testCsvDict)
#model2(trainCsvDict,testCsvDict)
