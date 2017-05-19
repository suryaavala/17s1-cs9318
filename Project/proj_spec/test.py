import time
localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))


import pandas as pd
from helper import *
import nltk
#print(nltk.pos_tag(['Surya']))



lines = read_data('./asset/training_data.txt')
for i in range(len(lines)):
    lines[i] = lines[i].split(":")

df = pd.DataFrame()
labels = ['word' , 'syl']
df = pd.DataFrame.from_records(lines, columns=labels)
df['syla'] = df['syl'].str.split(" ")
#df['type'] = df['word'].nltk.pos_tag()
df['type'] = df['word'].apply(nltk.pos_tag)
#empty values in a dataframe np.where(pd.isnull(df))

localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
