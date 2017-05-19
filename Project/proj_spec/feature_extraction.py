import time
localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))

import pandas as pd
from helper import *
import nltk
import numpy as np
from sklearn.preprocessing import OneHotEncoder

'''
Phenome Declaration
'''
# phe = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
#vows = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
# cons = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

phe_dict = {'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z':38, 'ZH': 39}

vows = [1, 2, 3, 4, 5, 6, 11, 12, 13, 17, 18, 25, 26, 33, 34]
cons = [7, 8, 9, 10, 14, 15, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39]

lines = read_data('./asset/training_data.txt')
data = []
for i in range(len(lines)):
#for i in range(5):
    word, nome = lines[i].split(":")
    nomes = nome.split(" ")
    #NOTE word = 'COED', nome = 'K OW1 EH2 D', tag = 'NN', ['K', 'OW1', 'EH2', 'D']
    tag = nltk.pos_tag([word])[0][1]
    nb_syllab = len(nomes)
    stress = []
    syl = []
    vow_encoding = []
    encoded_word = []
    for n in nomes:
        if n[-1].isdigit():
            stress.append(int(n[-1]))
            n = n[:-1]
        else:
            stress.append(-1)
        syl.append(n)
        encoded_letter = phe_dict[n]
        if encoded_letter in vows:
            vow_encoding.append(1)
        else:
            vow_encoding.append(0)

    encoded_word = list(map(phe_dict.get, syl))

    #print(word, syl, tag, nb_syllab, vow_encoding, encoded_word, stress)
    data.append([word, syl, tag, nb_syllab, vow_encoding, encoded_word, stress])

df = pd.DataFrame()
labels = ['word' , 'syl', 'pos', 'nb_syllab', 'vow_encoding', 'encoded_word', 'stress']
df = pd.DataFrame.from_records(data, columns=labels)

localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
