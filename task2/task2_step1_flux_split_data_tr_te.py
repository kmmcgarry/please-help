import random

data = open('/scratch/si699w18_fluxm/deahanyu/question_label_table.txt','r')
#skip the header
data = data.read().split('\n')[1:]

save_here_tr = open('/scratch/si699w18_fluxm/deahanyu/train_data.txt','w')
save_here_tr.write('question_post_id\tquestion\tlabel\n')
save_here_te = open('/scratch/si699w18_fluxm/deahanyu/test_data.txt','w')
save_here_te.write('question_post_id\tquestion\tlabel\n')

random.shuffle(data)
total_len = len(data)

tr = data[:int(total_len*0.8)]
te = data[int(total_len*0.8):]

print(len(tr))
print(len(te))

for i in tr:
	save_here_tr.write(i+'\n')
save_here_tr.close()

for i in te:
	save_here_te.write(i+'\n')
save_here_te.close()
