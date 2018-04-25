'''
step1:
the FP/FN analysis after building "initial" regex
'''
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import pandas as pd


def create_truelabel_dict(raw_data,fullSet=False):
	dd={}
	dd['True_Label']=[]
	if fullSet:
		#git rid of the header
		raw_data = raw_data[1:]
	for i in range(len(raw_data)):
		label = raw_data[i].split('\t')[-1].replace('\n','')
		dd['True_Label'].append(int(label))
	return dd

def label_by_regex(rgx,jj,raw_data,fullSet=False):
	c_0 = 0
	c_1 = 0
	jj['predicted'] = []
	list_of_comments = []
	if fullSet:
		#git rid of the header
		raw_data = raw_data[1:]
	for i in range(len(raw_data)):
		comment = raw_data[i].split('\t')[-3]
		label = raw_data[i].split('\t')[-1].replace('\n','')
		if re.findall(rgx[0],comment.lower()):
			jj['predicted'].append(1)
			if label=='0':
				c_0+=1
			else:
				c_1+=1
			#checking false positive
			if label=='0':
				list_of_comments.append(comment)
		else:
			#additional check
			if re.findall(rgx[1],comment.lower()):
				print(1)
				jj['predicted'].append(1)
				#checking false positive
				if label=='0':
					list_of_comments.append(comment)
			else:
				jj['predicted'].append(0)
				# #checking false negative
				# if label=='1':
				# 	list_of_comments.append(comment)

	# prec = c_1/(c_1+c_0)
	# reca = c_1/76
	# print("Total: "+str(c_1+c_0))
	# print("1's: "+str(c_1))
	# print("0's: "+str(c_0))
	# print()
	# print("Precision: "+str(prec))
	# print("Recall: "+str(reca))
	# print("F1 Score:" +str((2*prec*reca)/(prec+reca)))
	print(len(list_of_comments))
	return list_of_comments, classification_report(jj['True_Label'],jj['predicted'])

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

regex20 = r"(ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting)+[ ][\w ]{0,20}(your|a|the|ur|to)+[\w ]{0,20}(dr|doctor|doc|physician|docs|md|m\.d\.|primary|care|gp|general|practitioner|psychologist)+"
regex21 = r"(ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting)+[ ][\w ]{0,20}(your|a|the|ur|to)+[\w ]{0,20}(dr |doctor|doc |physician|docs |md|m\.d\.|primary|care|gp |general|practitioner|psychologist)+"
regex22 = r"(call|calling|ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting|speaking|speak|follow|seeing)+[ ]?[\w ]{0,40}( dr |doctor| doc |physician| docs | md | m\.d\.|primary care| gp |general|practitioner|psychologist|pediatrician|cardiologist)+"

#regex23 = r"(call|calling|ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting|speaking|speak|follow|seeing|make an appointment)+[ ][\w ]{0,40}(dr |doctor|doc |physician|docs |md |m\.d\.|primary care|gp |general|practitioner|psychologist|pediatrician|cardiologist|dentist|obg |obg\.)+"
regex23 = r"(call|calling|ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting|contact|speaking|speak|follow|seeing|make an appointment|make an appt\.|have an appointment|have an appt\.|schedule|seek|taken|run|refer)+[ ][\w ]{0,40}(dr |doctor|doc |physician|docs |md |m\.d\.|primary care|gp |gp\.|practitioner|psychologist|pediatrician|cardiologist|dentist|obg |obg\.|check up|ob |ob\.|gastroenterologist|endocrinologist|specialist|pcp |pcp\.)+"

additional_regex=r"(have)[ ][\w ]{0,10}(dr |doctor|doc |physician|docs |md |m\.d\.|primary care|gp |gp\.|practitioner|psychologist|pediatrician|cardiologist|dentist|obg |obg\.|check up|ob |ob\.|gastroenterologist)[ ](run |check | write )"

regex24 = r"(call|calling|ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting|contact|speaking|speak|follow|seeing|make an appointment|make an appt\.|have an appointment|have an appt\.|schedule|seek|taken|run|refer)+[ ][\w ]{0,40}( dr |doctor| doc |physician| docs | md | m\.d\.|primary care| gp | gp\.|practitioner|psychologist|pediatrician|cardiologist|dentist| obg | obg\.| check up| ob | ob\.|gastroenterologist|endocrinologist|specialist| pcp | pcp\.| llmd | llmd\.)+"
#regex23 = r"(call|calling|ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting|speaking|speak|from|follow|seeing|have|getting)+[ ][\w ]{0,40}(dr|doctor|doc|physician|docs|md|m\.d\.|primary|care|gp|general|practitioner|psychologist|pediatrician|cardiologist|dentist)+"
ll = [regex1,regex2,regex3,regex4,regex5, 
regex6,regex7,regex8,regex9,regex10,
regex11,regex12,regex13,regex14,regex15,
regex16,regex17,regex18,regex19]



if __name__ == '__main__':
	ask = open('data/rand_doc_master.tsv', 'r').readlines()
	#170
	train_set = ask[1:171]
	#30
	test_set = ask[171:]

	initial_dict_train = create_truelabel_dict(train_set)
	#LABEL: just regex
	comments_list, results = label_by_regex([regex22,additional_regex],initial_dict_train,train_set)
	for i in comments_list:
		print(i)
		print('\n')
		print('\n')
		print('\n')
	initial_dict_test = create_truelabel_dict(test_set)
	print("regex22")
	comments_list, results = label_by_regex([regex22,additional_regex],initial_dict_test,test_set)
	print(results)
	print("---------------")
	print("regex24-test set")
	comments_list, results = label_by_regex([regex24,additional_regex],initial_dict_test,test_set)
	print(results)
	print("---------------")

	# comments_list, results = label_by_regex(regex23,initial_dict_test,test_set)
	# print(results)
	print("regex24-train set")
	comments_list, results = label_by_regex([regex24,additional_regex],initial_dict_train,train_set)
	print(results)
	print("---------------")
	

	print("regex24-entire set")
	initial_dict = create_truelabel_dict(ask,True)
	comments_list, results = label_by_regex([regex24,additional_regex],initial_dict,ask,True)
	print(results)
	print("---------------")



	# for i in comments_list:
	# 	print(i)
	# 	print('\n')
	# 	print('\n')
	# 	print('\n')
	# initial_dict = create_truelabel_dict(ask,True)
	# #LABEL: just regex
	# comments_list, results = label_by_regex([regex23,additional_regex],initial_dict,ask,True)
	# print(results)










