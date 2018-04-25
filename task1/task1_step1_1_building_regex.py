'''
step1:
creating a regular expression with random samples of 200 from the comment table. 
'''

import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import pandas as pd


def create_truelabel_dict():
	dd={}
	dd['True_Label']=[]
	for i in range(len(ask)):
		if i != 0: 
			label = ask[i].split('\t')[-1].replace('\n','')
			dd['True_Label'].append(int(label))
	return dd

def label_by_regex(rgx,jj):
	c_0 = 0
	c_1 = 0
	jj['predicted'] = []
	for i in range(len(ask)):
		#skip the header
		if i != 0: 
			comment = ask[i].split('\t')[-3]
			label = ask[i].split('\t')[-1].replace('\n','')
			
			if re.findall(rgx,comment):
				jj['predicted'].append(1)
				if label=='0':
					c_0+=1
				else:
					c_1+=1	
			else:
				jj['predicted'].append(0)
			# 	if label=='1':
			# 		print(comment)
			# 		print("===================================")
	# prec = c_1/(c_1+c_0)
	# reca = c_1/76
	# print("Total: "+str(c_1+c_0))
	# print("1's: "+str(c_1))
	# print("0's: "+str(c_0))
	# print()
	# print("Precision: "+str(prec))
	# print("Recall: "+str(reca))
	# print("F1 Score:" +str((2*prec*reca)/(prec+reca)))
	return classification_report(jj['True_Label'],jj['predicted'])

def classifaction_report_csv(report,which_regex):
	lines = report.split('\n')
	row = [which_regex]
	for line in lines[2:4]:
		row_data = line.split('      ')
		if line == 2:
			#precision 1
			row.append(float(row_data[2]))
			#recall 2
			row.append(float(row_data[3]))
			#f1_score 3
			row.append(float(row_data[4]))
			#support 4
			row.append(float(row_data[5]))
		else:
			#precision 5
			row.append(float(row_data[2]))
			#recall 6
			row.append(float(row_data[3]))
			#f1_score 7
			row.append(float(row_data[4]))
			#support 8
			row.append(float(row_data[5]))
	#average precision
	row.append((row[1]+row[5])/2)
	row.append((row[2]+row[6])/2)
	row.append((row[3]+row[7])/2)
	row.append((row[4]+row[8]))
	return row

regex1 = r"(ask|see)[ ]+(your|a|the|ur)?[ ]?(dr|doctor|doc|physician|docs)"
regex2 = r"(ask|see)[ ]+(your|a|the|ur)?[ ]?(dr|doctor|doc|physician|docs|specialist)"
regex3 = r"(ask|see)+"
regex4 = r"(ask|see|tell|visit|talk)+"
regex5 = r"(ask|see|tell|visit|talk)[ ]+(your|a|the|ur)+"
regex6 = r"(ask|see|tell|visit|talk)[ ]+(your|a|the|ur)?"
regex7 = r"(your|a|the|ur)+[ ]?(dr|doctor|doc|physician|docs)+"
regex8 = r"(your|a|the|ur)+[ ]?(dr|doctor|doc|physician|docs)?"
regex9 = r"(your|a|the|ur)?[ ]?(dr|doctor|doc|physician|docs)?"
regex10 = r"(your|a|the|ur)+[\w ]*(dr|doctor|doc|physician|docs)+"
regex11 = r"(ask|see|visit|consult|talk|discuss|call)?[\w ]*(your|a|the|ur)+[\w ]*(dr|doctor|doc|physician|docs)+"
regex12 = r"(ask|see|visit|consult|talk|discuss|call)?[\w ]*"
regex13 = r"(your|a|the|ur)+[ ]?(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary)+"
regex14 = r"(your|a|the|ur)+[ ]?(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary|care|gp|GP|general|practitioner)+"
regex15 = r"(your|a|the|ur)+[\w ]*(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary|care|gp|GP|general|practitioner)+"
regex16 = r"(your|a|the|ur)?[\w ]*(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary|care|gp|GP|general|practitioner)+"
regex17 = r"(ask|see|tell|visit|talk)+[\w ]{0,1}(your|a|the|ur)+[\w ]{0,3}(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary|care|gp|GP|general|practitioner)+"
regex18 = r"(your|a|the|ur)+[\w ]{0,3}(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary|care|gp|GP|general|practitioner)+"
regex19 = r"(ask|see|tell|visit|talk|consult|go|check|ASK|SEE|TELL|VISIT|TALK|CONSULT|GO|CHECK)+[ ]*[\w]{0,1}(your|a|the|ur)+[\w ]{0,5}(dr|doctor|doc|physician|docs|md|MD|m\.d\.|primary|care|gp|GP|general|practitioner)+"

ll = [regex1,regex2,regex3,regex4,regex5, 
regex6,regex7,regex8,regex9,regex10,
regex11,regex12,regex13,regex14,regex15,
regex16,regex17,regex18,regex19]

if __name__ == '__main__':
	ask = open('rand_doc_master.tsv','r').readlines()
	print("True 1's:  76")
	print("True 0's: 124")
	print('\n')
	initial_dict = create_truelabel_dict()

	#LABEL: just regex
	for i in range(len(ll)):
			print("RegEx"+str(i+1)+": "+ll[i])
			jj = label_by_regex(ll[i],initial_dict)
			if i ==0:
				report_data = [classifaction_report_csv(jj,"RegEx"+str(i+1))]
			else:
				report_data.append(classifaction_report_csv(jj,"RegEx"+str(i+1)))
	dataframe = pd.DataFrame(report_data,columns=['RegEx','Precision_0','Recall_0','F1_score_0','Support_0','Precision_1','Recall_1','F1_score_1','Support_1','Precision_Average','Recall_Average','F1_score_Average','Support_Average'])
	dataframe.to_csv('classification_report.csv', index = False)
	
