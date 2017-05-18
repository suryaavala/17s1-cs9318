import time
localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))


import pandas as pd
from helper import *





lines = read_data('./asset/training_data.txt')
for i in range(len(lines)):
    lines[i] = lines[i].split(":")

df = pd.DataFrame()
labels = ['word' , 'syl']
df = pd.DataFrame.from_records(lines, columns=labels)
df['syla'] = df['syl'].str.split(" ")
#empty values in a dataframe np.where(pd.isnull(df))

localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
