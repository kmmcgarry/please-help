'''
step3:
creating a table: question_id, question, label on the flux
'''

#actual code starts below 
post_label = open('/scratch/si699w18_fluxm/deahanyu/postId_label_table.txt','r').readlines()
dict_id_label = {}
for i in post_label[1:]:
	a = i.replace('\n','').split(',')
	dict_id_label[a[0]] = a[1]
post_label = None

question_table = open('/scratch/si699w18_fluxm/jiaqima/MedHelp/new_question_table.txt','r').readlines()

save_here = open('/scratch/si699w18_fluxm/deahanyu/question_label_table.txt','w')
save_here.write('question_post_id\tquestion\tlabel\n')

diff = []
for i in question_table:
	each = i.replace('\n','').split('\t')
	if len(each) == 9:
		diff.append(each)
	#there must be a post id 
	if each[0][:5]=='post_':
		post_id = each[0]
		question = each[-4].lower()
		if post_id in dict_id_label:
			dict_id_label[post_id] =[question,dict_id_label[post_id]] 

save_here.write('\n'.join('{},{},{}'.format(x[0],x[1][0],x[1][1]) for x in dict_id_label.items()))
save_here.close()
print(diff)


table = open('/scratch/si699w18_fluxm/deahanyu/question_label_table.txt','r').readlines()

