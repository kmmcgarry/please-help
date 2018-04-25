'''
Kristen McGarry
Last Updated: March 15, 2018
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
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def splitData(fileName):
    csvDict={'0':[],'1':[]}

    with open(fileName,'rU',encoding='latin-1') as tsvfile:
        reader = csv.reader((x.replace('\0', '') for x in tsvfile), delimiter= "\t")
        for row in reader:
            if row != ['question_post_id', 'question', 'label']:
                comment_id = row[0]
                content = row[1]
                label = row[2]
                if label != "Label":
                    csvDict[label].append([comment_id,content])

    # print len(csvDict['0'])
    # print len(csvDict['1'])
    return (csvDict)


def wordFreq(trainDict,testDict):
    trainText1 = ""
    trainText0 = ""

    for i in trainDict['0']:
        trainText0 += (" " + i[1])
    for i in trainDict['1']:
        trainText1 += (" " + i[1])

    textTrain1 = nltk.word_tokenize(trainText1)
    textTrain0 = nltk.word_tokenize(trainText0)

    # print len(textTrain0)
    # print len(textTrain1)


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

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(train["data"])
    #print X_train_counts.shape
    # uses each word as a feature
    #print (count_vect.get_feature_names())

    #clf = MultinomialNB().fit(X_train_counts, train["target"])


    text_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])

    text_clf = text_clf.fit(train["data"], train["target"])

    predicted = text_clf.predict(testComment)
    # print (np.mean(predicted == testLabels))
    # #count of 1/ len predicted
    # predicted = predicted.tolist()
    # print (predicted)
    # print (testLabels)

    #precision =  (predicted.count(1)*1.0) / (len(predicted))
    #recall = (predicted.count(1)*1.0)/ (testLabels.count(1))


    print (classification_report(testLabels, predicted))

# Model: Mulitnomial BOW with TF/IDF
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
    #
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(train["data"])
    #print X_train_counts.shape
    # uses each word as a feature
    #print (count_vect.get_feature_names())

    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_train_tfidf.shape


    #clf = MultinomialNB().fit(X_train_counts, train["target"])


    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

    text_clf = text_clf.fit(train["data"], train["target"])

    predicted = text_clf.predict(testComment)

    # print (np.mean(predicted == testLabels))
    # #count of 1/ len predicted
    # predicted = predicted.tolist()
    # print (predicted)
    # print (testLabels)
    #
    # precision =  (predicted.count(1)*1.0) / (len(predicted))
    # recall = (predicted.count(1)*1.0)/ (testLabels.count(1))
    #
    # print ("PRECISION: ")
    # print (precision)
    # print ("RECALL: ")
    # print (recall)


    print (classification_report(testLabels, predicted))

#SVC with BOW
def model3(trainDict,testDict):
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

    text_clf = Pipeline([('vect', CountVectorizer()),('clf', svm.SVC())])

    text_clf = text_clf.fit(train["data"], train["target"])

    predicted = text_clf.predict(testComment)

    print (classification_report(testLabels, predicted))

# SVC, bow tf-idf
def model4(trainDict,testDict):
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

        # count_vect = CountVectorizer()
        # X_train_counts = count_vect.fit_transform(train["data"])
        #print X_train_counts.shape
        # uses each word as a feature
        #print count_vect.get_feature_names()

        # clf = svm.SVC()
        # clf.fit(X_train_counts, train["target"])


        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', svm.SVC())])

        text_clf = text_clf.fit(train["data"], train["target"])

        predicted = text_clf.predict(testComment)

        print (classification_report(testLabels, predicted))


# SVC, bow after removing stop words
def model5(trainDict,testDict):
        train = {"data":[],"target":[],"commentID":[]}
        for i in trainDict['0']:
            train["target"].append(0)
            train["data"].append(i[1])
            train["commentID"].append(i[0])
        for i in trainDict['1']:
            train["target"].append(1)
            train["data"].append(i[1])
            train["commentID"].append(i[0])


        new_question = []
        for i in testDict['1']:
            new_question.append([i[0],i[1],1])
        for i in testDict['0']:
            new_question.append([i[0],i[1],0])

        random.shuffle(new_question)

        testLabels = []
        testQuestionID = []
        testQuestion = []
        for i in new_question:
            testQuestionID.append(i[0])
            testQuestion.append(i[1])
            testLabels.append(i[2])

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train["data"])
        SVC = svm.SVC(kernel="linear")
        clf = SVC.fit(X_train_counts, train["target"])

        X_new_counts = count_vect.transform(testQuestion)

        predicted = clf.predict(X_new_counts)

        print ("Accuracy: "  + str(metrics.accuracy_score(testLabels, predicted)))
        print ("ROC AUC Score: " + str(metrics.roc_auc_score(testLabels, predicted)))
        print (classification_report(testLabels, predicted))


# logistic regression
def model6(trainDict,testDict):
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

        text_clf = Pipeline([('vect', CountVectorizer()),('clf', linear_model.LogisticRegression())])
        print ("fitting model")
        text_clf = text_clf.fit(train["data"], train["target"])
        print ("predicting..")
        predicted = text_clf.predict(testComment)

        print (metrics.accuracy_score(testLabels, predicted))
        print (metrics.roc_auc_score(testLabels, predicted))
        print (classification_report(testLabels, predicted))

#random forest
def model7(trainDict,testDict):
    train = {"data":[],"target":[],"commentID":[]}
    for i in trainDict['0']:
        train["target"].append(0)
        train["data"].append(i[1])
        train["commentID"].append(i[0])
    for i in trainDict['1']:
        train["target"].append(1)
        train["data"].append(i[1])
        train["commentID"].append(i[0])


    new_question = []
    for i in testDict['1']:
        new_question.append([i[0],i[1],1])
    for i in testDict['0']:
        new_question.append([i[0],i[1],0])

    random.shuffle(new_question)

    testLabels = []
    testQuestionID = []
    testQuestion = []
    for i in new_question:
        testQuestionID.append(i[0])
        testQuestion.append(i[1])
        testLabels.append(i[2])

    text_clf = Pipeline([('vect', CountVectorizer()),('clf', RandomForestClassifier())])
    print ("fitting model")
    text_clf = text_clf.fit(train["data"], train["target"])
    print ("predicting..")
    predicted = text_clf.predict(testQuestion)

    print ("Accuracy: "  + str(metrics.accuracy_score(testLabels, predicted)))
    print ("ROC AUC Score: " + str(metrics.roc_auc_score(testLabels, predicted)))
    print (classification_report(testLabels, predicted))

# logistic regression, TF/IDF
def model8(trainDict,testDict):
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

        text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', linear_model.LogisticRegression())])
        print ("fitting model")
        text_clf = text_clf.fit(train["data"], train["target"])
        print ("predicting..")
        predicted = text_clf.predict(testComment)

        print (metrics.accuracy_score(testLabels, predicted))
        print (metrics.roc_auc_score(testLabels, predicted))
        print (classification_report(testLabels, predicted))


# logistic regression, TF/IDF, different n-gram features
def model9(trainDict,testDict):
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

        text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 4))),('clf', linear_model.LogisticRegression())])
        print ("fitting model")
        text_clf = text_clf.fit(train["data"], train["target"])
        print ("predicting..")
        predicted = text_clf.predict(testComment)

        print (metrics.accuracy_score(testLabels, predicted))
        print (metrics.roc_auc_score(testLabels, predicted))
        print (classification_report(testLabels, predicted))


# testCsvDict = splitData('/Users/kristen/Documents/SI_699/question_label_200_test.tsv')
# print ("----------DONE--------")
# trainCsvDict = splitData('/Users/kristen/Documents/SI_699/question_label_200_train.tsv')
# print ("----------DONE--------")

testCsvDict = splitData('clean_test_data.tsv')
print ("----------DONE--------")
trainCsvDict = splitData('clean_train_data.tsv')
print ("----------DONE--------")

#wordFreq(trainCsvDict,testCsvDict)
# print ("Naive Bayes, BOW")
# model1(trainCsvDict,testCsvDict)
# print ("Naive Bayes, BOW, TFIDF")
# model2(trainCsvDict,testCsvDict)
# print ("SVC, BOW")
# model3(trainCsvDict,testCsvDict)
# print ("SVC, BOW, TFIDF")
# model4(trainCsvDict,testCsvDict)
#model5(trainCsvDict,testCsvDict)
#model6(trainCsvDict,testCsvDict)
print ("Logistic Regression TF/IDF:")
#model8(trainCsvDict,testCsvDict)
model9(trainCsvDict,testCsvDict)
