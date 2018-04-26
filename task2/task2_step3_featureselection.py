import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn import metrics
import random
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import os
# import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame



# def range_intervals(_ll):
#     total_len = len(_ll)
#     if total_len < 500:
#         return list(range(total_len))
#     else:
#         return list(range(100))+list(range(100,500,10))+list(range(500,total_len,100))

def splitData(fileName):
    csvDict={'0':[],'1':[]}

    with open(fileName,'rU',encoding='latin-1') as tsvfile:
        reader = csv.reader((x.replace('\0', '') for x in tsvfile), delimiter= "\t")
        for row in reader:
            if ((row != ['question_post_id', 'question', 'label']) and (row != [])):
                comment_id = row[0]
                content = row[1]
                label = row[2]
                if label != "Label":
                    csvDict[label].append([comment_id,content])

    return (csvDict)



def featureselection(trainDict,testDict):
    train = {"data":[],"target":[],"questionID":[]}
    for i in trainDict['0']:
        train["target"].append(0)
        train["data"].append(i[1])
        train["questionID"].append(i[0])
    for i in trainDict['1']:
        train["target"].append(1)
        train["data"].append(i[1])
        train["questionID"].append(i[0])

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
    print (X_train_counts.shape)

    # chi2_output = chi2(X_train_counts, train["target"])
    # chi2_score = chi2_output[0]
    # chi2_p = chi2_output[1]
    #
    # counter = 0
    # for pvalue in chi2_p:
    #     if pvalue < 0.05:
    #         counter += 1
    # print (counter)

    skb = SelectKBest(chi2,k=10)


    X_train_counts2 = skb.fit_transform(X_train_counts,train["target"])
    print (X_train_counts2.shape)

    # print (X_train_counts2.shape)
    #print (X_train_counts2.vocabulary_())
    #rint (X_train_counts2.stop_words_())

    #print(DataFrame(X_train_counts.A, columns=count_vect.get_feature_names()).to_string())


    tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts2)
    # feature_names = tfidf_transformer.get_feature_names()
    # print (feature_names)


    text_clf = RandomForestClassifier()
    print ("fitting model")
    text_clf = text_clf.fit(X_train_tfidf, train["target"])

    X_test_counts = count_vect.transform(testComment)
    X_test_counts2 = skb.transform(X_test_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts2)


    print ("predicting..")

    predicted = text_clf.predict(X_test_tfidf)
    print (classification_report(testLabels, predicted))
    print (metrics.accuracy_score(testLabels, predicted))


    predicted_prob = text_clf.predict_proba(X_test_tfidf)

    predicted_prob = predicted_prob[:,1]

    print (metrics.roc_auc_score(testLabels, predicted_prob))


    true_label = predicted
    probabilities = predicted_prob

    _sorted = sorted(list(zip(true_label,probabilities)),key=lambda x : -x[1])
    _Y = []

    for j in range_intervals(_sorted):
        _k = j+1
        _sub_sorted = _sorted[:_k]
        _count = 0.0
        for t in _sub_sorted:
            if t[0] == 1:
                _count += 1
        _precision = _count/_k
        _Y.append(_precision)
    #scatter plot
    _X=[range(1,len(range_intervals(_sorted))+1)]
    plt.scatter(_X, _Y, c='blue')

    #change axes ranges
    plt.xlim(1,1600)
    plt.ylim(0,1)

    #add title
    plt.title('Random Forest on UMLS Cleaned Data')

    #add x and y labels
    plt.xlabel('K')
    plt.ylabel('Precision')

    #show plot
    plt.show()


# Uncomment below this line to execute
# -------------------------------------
# trainDict = splitData('cleanumls_train_data.tsv')
# print ("split train...")
# testDict = splitData('cleanumls_test_total.tsv')
# print ("split test...")
#
# featureselection(trainDict,testDict)
