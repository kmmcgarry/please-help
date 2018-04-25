'''
step0:
running an inter-rater reliability test between K and D
'''
from sklearn.metrics import cohen_kappa_score
import pandas as pd

p = pd.read_csv('k_d.csv')
print("Cohen's Kappa before the concrete codebook: "+str(cohen_kappa_score(p['Kristen'].tolist(), p['Deahan'].tolist())))
