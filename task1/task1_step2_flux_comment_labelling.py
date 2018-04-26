'''
step2:
labelling on the comment table on flux
'''
# #practice table variable 
# comment_table = ['comment_post_id\tcomment_time\tuser_id\tto_user_id\tto_user_name\tcomment_content\tquestion_post_id\n', "post_2028347\tOct 26, 2007\t309783\t248305\tjonezzey13\tAcne is caused by oil, bacteria and hormones. If you're getting pimples ask your doctor for a course of oral antibiotics but do mention any other medications you are taking in case they are contra-indicated. Also invest in a good facial wash for acne prone skin. I'm sure people are not just saying things to make you feel better, you're in discomfort every day so no wonder you feel down about yourself.\tpost_1702782\n", "post_5818112\tJun 20, 2010\t1351611\t248305\tjonezzey13\tHi jonezzey,I have a similar problem. A couple years ago, I had a pretty bad tooth infection. I was told I needed a root canal but couldn't afford it. I was on antibiotics to try and kill the infection. After a couple months of dealing with it, the tooth pain went away. But then I broke out really bad with acne just right above the tooth infection. I swear that the two are linked but my dermatologist and general doctor don't think so. Now I have really bad cystic acne and I've tried everything to get rid of it. I was pretty convinced it didn't have to do with my tooth but now I am definitely reconsidering that possibility.\tpost_1702782\n", "post_5954720\tJul 25, 2010\t1288072\tNone\tNone\tDon't hijack the post, start your own post, and will gladly try to answer it.\tpost_1702782\n", "post_6337506\tNov 09, 2010\t776067\tNone\tjonnezzy13\tSame as me.    My tooth chipped and lost its molar about a month or more then i started experiencing terrible acne problems. The worst acne is on the same side as my chipped, molar missing tooth.    I never went to see a dentist.    I'm finally going to see a dentist in a few weeks.    It's been over a year now since this has all started. I'm 32, I'd already gone through acne problems as a teenager, I do not need to go through it again.\tpost_1702782\n", 'post_7278419\tSep 28, 2011\t1821377\tNone\thttp\thttp://www.21stcenturydental.com/smith/education/acne.htmI had a tooth infection and was taking the antibiotic for two days when my skin started to break out. I thought it was an allergic reaction to the antibiotic, but now am thinking it might have not been the case.\tpost_1702782\n', 'post_7927475\tMay 05, 2012\t2144453\t1351611\temo_lee20\tI have had many teeth problems in my life, I still need alot more work done but have not had the money.    My face has been breaking out for the last 4 months really bad! I have never had an issue with acne in my life! I swear it is related to my teeth! I think the infection in my mouth is causing the break out on my face.    Logically if we think about it it makes sense.\tpost_1702782\n', "post_8591026\tDec 06, 2012\t4452381\t248305\tjonezzey13\tI had acne for 3 years (I am in my 40s) and tried everything - from dermatologists to regular doctors, changed my diet for example tried giving up dairy - no luck. gave up meat - no luck. gave up nuts, and on and on. Yet my acne was relentless. I would get pimples on my forehead, under my eyes, on my jawline and even on my eyelids. I had a root canal done a few years prior that I always had a low level of pain with. I never really thought much about it but finally went to the dentist and was told that I had an infection/abscess and that the root canal had to be redone. But in all the reading I had done on the internet there are a lot of holistic types that swear root canals are not the way to go and that in many cases they lead to problems with dental cavitation, infections, etc. So I opted to have the tooth pulled. within 2 weeks of pulling my tooth my acne was completely gone! I have no regrets about pulling the tooth and I can't believe after 3 years of acne is 100% clear! For some reason doctors, dentists, and dermatologists have no clue about dental infections and acne but I'm living proof that there is correlation.\tpost_1702782\n", "post_2027495\tOct 26, 2007\t309783\t266783\tReuben2222\tIt could be that you're sensitive to the sun in which case cover up or use SPF. It is possible to develop an allergy to practically anything including spicy foods or laundry detergent. There is no evidence to prove that oily foods cause bad skin but a healthy diet can only help you. I would see your doctor if they don't heal it could be psoriasis. x\tpost_1848808\n", 'post_2509343\tFeb 14, 2008\t419707\tNone\tloiloi and Reuben2222\tBut if the sores are on your buttocks and scalp (like mine are) they are not exposed to the sun.    I have thick hair on my head and get the sores on my scalp.    And I wear pants that always cover my rear end....so, again, these areas are not exposed to the sun.I have to admit....I\'m a "picker" out of nervous habit and pick these sores.    Is this causing them to NOT heal?\tpost_1848808\n']

#actual code starts below 
import re

comment_table = open('new_comment_table.txt','r').readlines()

regex1=r"(call|calling|ask|see|tell|visit|talk|consult|go|check|find|checked|show|contacting|contact|speaking|speak|follow|seeing|make an appointment|make an appt\.|have an appointment|have an appt\.|schedule|seek|taken|run|refer)+[ ][\w ]{0,40}( dr |doctor| doc |physician| docs | md | m\.d\.|primary care| gp | gp\.|practitioner|psychologist|pediatrician|cardiologist|dentist| obg | obg\.| check up| ob | ob\.|gastroenterologist|endocrinologist|specialist| pcp | pcp\.| llmd | llmd\.)+"
regex2=r"(have)[ ][\w ]{0,10}(dr |doctor|doc |physician|docs |md |m\.d\.|primary care|gp |gp\.|practitioner|psychologist|pediatrician|cardiologist|dentist|obg |obg\.|check up|ob |ob\.|gastroenterologist)[ ](run |check | write )"

def is_negation(regexx,commentt):
	initial_postion = re.search(regexx,commentt).start()
	#getting some parts of the sentence
	prior_characters = commentt[initial_postion-15:initial_postion]
	#look for if there is a period, and get the one after the period
	prior_characters = prior_characters.split('.')[-1]
	#check if there is no negation
	return [i for i in ["n't","never","not", "no"] if i in prior_characters]

def get_label(regexx,commentt):
	if not is_negation(regexx,commentt):
		# list_of_label.append(1)
		return 1
	else:
		return 0

def do_labelling(regexx1,regexx2,commentt):
	if re.findall(regexx1,commentt):
		return get_label(regexx1,commentt)
	else:
		#additional check
		if re.findall(regexx2,commentt):
			return get_label(regexx2,commentt)
		else:
			return 0

save_here = open('/scratch/si699w18_fluxm/deahanyu/postId_label_table.txt','w')
save_here.write('question_post_id,label\n')

dict_id_label = {}

for i in comment_table:
	each = i.replace('\n','').split('\t')
	#there must be a post id 
	if each[-1][:5]=='post_':
		post_id = each[-1]
		comment = each[-2].lower()

		if post_id not in dict_id_label:
			dict_id_label[post_id] = do_labelling(regex1,regex2,comment)
		else:
			#we want to check again only when it was labeled as 0
			#1's are fine because we know that that question is 1.
			if dict_id_label[post_id] == 0:
				dict_id_label[post_id] = do_labelling(regex1,regex2,comment)

save_here.write('\n'.join('{},{}'.format(x[0],x[1]) for x in dict_id_label.items()))
save_here.close()




