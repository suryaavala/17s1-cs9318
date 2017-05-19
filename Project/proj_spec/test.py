import time
localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))


import pandas as pd
from helper import *
import nltk
#print(nltk.pos_tag(['Surya']))

phe = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
vows = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
cons = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

lines = read_data('./asset/training_data.txt')
for i in range(len(lines)):
    lines[i] = lines[i].split(":")

df = pd.DataFrame()
labels = ['word' , 'syl']
df = pd.DataFrame.from_records(lines, columns=labels)
df['syla'] = df['syl'].str.split(" ")
#df['type'] = df['word'].nltk.pos_tag()
#df['type'] = df['word'].apply(nltk.pos_tag)
df['list'] = df['word'].apply(lambda x: [x])
df['type'] = df['list'].apply(lambda x : nltk.pos_tag(x)[0][1])
#empty values in a dataframe np.where(pd.isnull(df))
#unique values p = df.type.unique()

localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
