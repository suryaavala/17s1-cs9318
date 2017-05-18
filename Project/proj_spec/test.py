import pandas as pd
from helper import *
import time



localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
lines = read_data('./asset/training_data.txt')
for i in range(len(lines)):
    lines[i] = lines[i].split(":")

df = pd.DataFrame()
labels = ['word' , 'syl']
df = pd.DataFrame.from_records(lines, columns=labels)
df['syla'] = df['syl'].str.split(" ")
# for l in lines:
#     word, syl = l.split(":")
#     syl = syl.split(" ")
#     df = df.append({'word' : word, 'syl' : syl}, ignore_index = True)

localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
