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

# phe_dict = {'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z':38, 'ZH': 39}

phe_dict = {'AA': 0, 'AE': 0, 'AH': 0, 'AO': 0, 'AW': 0, 'AY': 0, 'B': 0, 'CH': 0, 'D': 0, 'DH': 0, 'EH': 0, 'ER': 0, 'EY': 0, 'F': 0, 'G': 0, 'HH': 0, 'IH': 0, 'IY': 0, 'JH': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'NG': 0, 'OW': 0, 'OY': 0, 'P': 0, 'R': 0, 'S': 0, 'SH': 0, 'T': 0, 'TH': 0, 'UH': 0, 'UW': 0, 'V': 0, 'W': 0, 'Y': 0, 'Z': 0, 'ZH': 0}

# vows = [1, 2, 3, 4, 5, 6, 11, 12, 13, 17, 18, 25, 26, 33, 34]
# cons = [7, 8, 9, 10, 14, 15, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39]

lines = read_data('./asset/training_data.txt')
data = []
for i in range(len(lines)):
#for i in range(5):
    in_dict = {'AA': 0, 'AE': 0, 'AH': 0, 'AO': 0, 'AW': 0, 'AY': 0, 'B': 0, 'CH': 0, 'D': 0, 'DH': 0, 'EH': 0, 'ER': 0, 'EY': 0, 'F': 0, 'G': 0, 'HH': 0, 'IH': 0, 'IY': 0, 'JH': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'NG': 0, 'OW': 0, 'OY': 0, 'P': 0, 'R': 0, 'S': 0, 'SH': 0, 'T': 0, 'TH': 0, 'UH': 0, 'UW': 0, 'V': 0, 'W': 0, 'Y': 0, 'Z': 0, 'ZH': 0}

    out_dict = {'AA': -2, 'AE': -2, 'AH': -2, 'AO': -2, 'AW': -2, 'AY': -2, 'B': -2, 'CH': -2, 'D': -2, 'DH': -2, 'EH': -2, 'ER': -2, 'EY': -2, 'F': -2, 'G': -2, 'HH': -2, 'IH': -2, 'IY': -2, 'JH': -2, 'K': -2, 'L': -2, 'M': -2, 'N': -2, 'NG': -2, 'OW': -2, 'OY': -2, 'P': -2, 'R': -2, 'S': -2, 'SH': -2, 'T': -2, 'TH': -2, 'UH': -2, 'UW': -2, 'V': -2, 'W': -2, 'Y': -2, 'Z': -2, 'ZH': -2}


    word, nome = lines[i].split(":")
    nomes = nome.split(" ")
    #NOTE word = 'COED', nome = 'K OW1 EH2 D', tag = 'NN', ['K', 'OW1', 'EH2', 'D']
    tag = nltk.pos_tag([word])[0][1]
    nb_syllab = len(nomes)
    stress = []
    syl = []
    #vow_encoding = []
    #encoded_word = []
    j = 0
    for n in nomes:
        if n[-1].isdigit():
            stress.append(int(n[-1]))
            out_dict[n] = int(n[-1])
            n = n[:-1]
        else:
            stress.append(-1)
        syl.append(n)
        in_dict[n] = j
        # encoded_letter = phe_dict[n]
        # if encoded_letter in vows:
        #     vow_encoding.append(1)
        # else:
        #     vow_encoding.append(0)

        j += 1
        if i == 1:
            print(out_dict)

    #encoded_word = list(map(phe_dict.get, syl))

    #print(word, syl, tag, nb_syllab, vow_encoding, encoded_word, stress)
    #data.append([word, syl, tag, nb_syllab, vow_encoding, encoded_word, stress])
    in_l = []
    out_l = []

    for k in list(in_dict.keys()):
        in_l.append(in_dict[k])
        out_l.append(out_dict[k])
    if i == 1:
        print(out_dict)

    # if j == 1:
    #     print(tag, nb_syllab, in_dict)
    d = [tag, nb_syllab]
    d.extend(in_l)
    d.append(out_l)
    data.append(d)

df = pd.DataFrame()
#labels = ['word' , 'syl', 'pos', 'nb_syllab', 'vow_encoding', 'encoded_word', 'stress']
labels = ['pos', 'nb_syllab']
labels.extend(list(phe_dict.keys()))
labels.append('stress')


df = pd.DataFrame.from_records(data, columns=labels)

localtime = time.localtime()
mi = localtime.tm_min
sec = localtime.tm_sec
print('{}:{}'.format(mi, sec))
