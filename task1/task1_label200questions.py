# Purpose: Grabbing all labels from comments, and assigning them to the questions in the subsample of 200 comments.
import csv

# get labels
labels = {}
with open("200_labels.csv",'rU') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        commentpostId = row[0]
        label = row[1]
        labels[commentpostId] = label

in_file = open("rand_doc.txt","r")
text = in_file.readlines()

listOfId = []
questionLabel = {}
# get all post_id from comments
# order: comment_post_id	comment_time	user_id	to_user_id	to_user_name	comment_content	question_post_id
for comment in text:
    #print comment
    comment_post_id = comment.split('\t')[0]
    label = labels[comment_post_id]
    post_id = comment.split('\t')[-1]

    questionLabel[post_id.replace('\n','')] = label

    if post_id not in listOfId:
        listOfId.append(post_id.replace('\n',''))
in_file.close()

#get question for each post id
in_file_2 = open("/Users/kristen/Documents/SI_699/new_question_table.txt","r")
text_2 = in_file_2.readlines()
#key = question_post_id, value = question

questions = {}
for question in text_2:
    contents = question.split('\t')
    post_id = contents[0]
    content = contents[6]

    questions[post_id] = content



# get question, content, and label and write to tsv file
with open("question_label_200.tsv", "w") as csvfile:
    csvfile.write("Question_Post_Id    Content    Label\n")
    for id in listOfId:
        content = questions[id]
        label = questionLabel[id]
        csvfile.write(id + "\t" + content + "\t" + label + "\n")
