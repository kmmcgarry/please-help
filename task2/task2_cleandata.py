# iterate through all questions and remove stop words and numeric values
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean(tsvFile,outputTsv):
    stopWords = set(stopwords.words('english'))
    with open(tsvFile,'rU',encoding='latin-1') as tsvfile, open(outputTsv,'w') as outTsv:
        reader = csv.reader((x.replace('\0', '') for x in tsvfile), delimiter= "\t")
        writer = csv.writer(outTsv,delimiter="\t")
        writer.writerow(['question_post_id', 'question', 'label'])

        for row in reader:
            if row != ['question_post_id', 'question', 'label']:
                postid = row[0]
                question = row[1]
                label = row[2]

                wordtokens = word_tokenize(question)

                filteredsentence = []

                for w in wordtokens:
                    try:
                        test = (int(w))
                        filteredsentence.append("num")
                    except:
                        if w not in stopWords:
                            filteredsentence.append(w)

                finalquestion = ""
                for token in filteredsentence:
                    finalquestion = finalquestion + " " + token

                writer.writerow([postid,finalquestion,label])
    tsvfile.close()
    outTsv.close()
