## import modules here


################# training #################

def train(data, classifier_file):# do not change the heading of the function
    import pandas as pd
    import nltk
    import numpy as np
    '''
    Phenme Declaration
    '''
    phe_dict = {'AA': 0, 'AE': 0, 'AH': 0, 'AO': 0, 'AW': 0, 'AY': 0, 'B': 0, 'CH': 0, 'D': 0, 'DH': 0, 'EH': 0, 'ER': 0, 'EY': 0, 'F': 0, 'G': 0, 'HH': 0, 'IH': 0, 'IY': 0, 'JH': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'NG': 0, 'OW': 0, 'OY': 0, 'P': 0, 'R': 0, 'S': 0, 'SH': 0, 'T': 0, 'TH': 0, 'UH': 0, 'UW': 0, 'V': 0, 'W': 0, 'Y': 0, 'Z': 0, 'ZH': 0}

    ##############NOTE DATA CLEANING##################
    #lines = data[:]
    train_data = []
    for i in range(len(data)):
    #for i in range(5):
        in_dict = {'AA': 0, 'AE': 0, 'AH': 0, 'AO': 0, 'AW': 0, 'AY': 0, 'B': 0, 'CH': 0, 'D': 0, 'DH': 0, 'EH': 0, 'ER': 0, 'EY': 0, 'F': 0, 'G': 0, 'HH': 0, 'IH': 0, 'IY': 0, 'JH': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'NG': 0, 'OW': 0, 'OY': 0, 'P': 0, 'R': 0, 'S': 0, 'SH': 0, 'T': 0, 'TH': 0, 'UH': 0, 'UW': 0, 'V': 0, 'W': 0, 'Y': 0, 'Z': 0, 'ZH': 0}

        #out_dict = {'AA': -2, 'AE': -2, 'AH': -2, 'AO': -2, 'AW': -2, 'AY': -2, 'B': -2, 'CH': -2, 'D': -2, 'DH': -2, 'EH': -2, 'ER': -2, 'EY': -2, 'F': -2, 'G': -2, 'HH': -2, 'IH': -2, 'IY': -2, 'JH': -2, 'K': -2, 'L': -2, 'M': -2, 'N': -2, 'NG': -2, 'OW': -2, 'OY': -2, 'P': -2, 'R': -2, 'S': -2, 'SH': -2, 'T': -2, 'TH': -2, 'UH': -2, 'UW': -2, 'V': -2, 'W': -2, 'Y': -2, 'Z': -2, 'ZH': -2}
        try:
            word, nome = data[i].split(":")
        except ValueError:
            continue
        nomes = nome.split(" ")
        tag = nltk.pos_tag([word])[0][1]
        nb_syllab = len(nomes)
        out = 0
        syl = []


        for j in range(len(nomes)):
            if nomes[j][-1].isdigit():
                stress = nomes[j][-1]
                nome = nomes[j][:-1]
                in_dict[nome] = j+1
                if int(stress) == 1:
                    out = j
                #out_dict[nome] = int(stress)
                syl.append(nome)

            else:
                in_dict[nomes[j]] = j+1
                syl.append(nomes[j])


        in_l = [in_dict[k] for k in in_dict]
        #out_l = [out_dict[l] for l in out_dict]

        d = [word,syl,tag, nb_syllab]
        d.extend(in_l[:])
        #d.append(out_l[:])
        d.append(out)

        train_data.append(d)


    df = pd.DataFrame()
    labels = ['word', 'sy','pos', 'nb_syllab']
    labels.extend(list(phe_dict.keys()))
    labels.append('stress')

    df = pd.DataFrame.from_records(train_data, columns=labels)

    #NOTE ML
    X = df[df.columns[3:-1]].as_matrix()
    Y = df[df.columns[-1]].as_matrix()

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)

    ##########NOTE SAVING THE CLASSIFIER#########
    import pickle
    save_classifier = open (classifier_file, 'wb')
    pickle.dump(clf, save_classifier)
    save_classifier.close()
    return


################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    import pandas as pd
    import nltk
    import numpy as np
    from sklearn import tree
    import pickle

    classifier_f = open(classifier_file, 'rb')
    clf = pickle.load(classifier_f)
    classifier_f.close()

    '''
    Phenme Declaration
    '''
    phe_dict = {'AA': 0, 'AE': 0, 'AH': 0, 'AO': 0, 'AW': 0, 'AY': 0, 'B': 0, 'CH': 0, 'D': 0, 'DH': 0, 'EH': 0, 'ER': 0, 'EY': 0, 'F': 0, 'G': 0, 'HH': 0, 'IH': 0, 'IY': 0, 'JH': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'NG': 0, 'OW': 0, 'OY': 0, 'P': 0, 'R': 0, 'S': 0, 'SH': 0, 'T': 0, 'TH': 0, 'UH': 0, 'UW': 0, 'V': 0, 'W': 0, 'Y': 0, 'Z': 0, 'ZH': 0}

    ##############NOTE DATA CLEANING##################
    #lines = data[:]
    train_data = []
    for i in range(len(data)):
    #for i in range(5):
        in_dict = {'AA': 0, 'AE': 0, 'AH': 0, 'AO': 0, 'AW': 0, 'AY': 0, 'B': 0, 'CH': 0, 'D': 0, 'DH': 0, 'EH': 0, 'ER': 0, 'EY': 0, 'F': 0, 'G': 0, 'HH': 0, 'IH': 0, 'IY': 0, 'JH': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'NG': 0, 'OW': 0, 'OY': 0, 'P': 0, 'R': 0, 'S': 0, 'SH': 0, 'T': 0, 'TH': 0, 'UH': 0, 'UW': 0, 'V': 0, 'W': 0, 'Y': 0, 'Z': 0, 'ZH': 0}

        #out_dict = {'AA': -2, 'AE': -2, 'AH': -2, 'AO': -2, 'AW': -2, 'AY': -2, 'B': -2, 'CH': -2, 'D': -2, 'DH': -2, 'EH': -2, 'ER': -2, 'EY': -2, 'F': -2, 'G': -2, 'HH': -2, 'IH': -2, 'IY': -2, 'JH': -2, 'K': -2, 'L': -2, 'M': -2, 'N': -2, 'NG': -2, 'OW': -2, 'OY': -2, 'P': -2, 'R': -2, 'S': -2, 'SH': -2, 'T': -2, 'TH': -2, 'UH': -2, 'UW': -2, 'V': -2, 'W': -2, 'Y': -2, 'Z': -2, 'ZH': -2}
        try:
            word, nome = data[i].split(":")
        except ValueError:
            continue
        nomes = nome.split(" ")
        tag = nltk.pos_tag([word])[0][1]
        nb_syllab = len(nomes)
        out = 0
        syl = []


        for j in range(len(nomes)):
            if nomes[j][-1].isdigit():
                stress = nomes[j][-1]
                nome = nomes[j][:-1]
                in_dict[nome] = j+1
                if int(stress) == 1:
                    out = j
                #out_dict[nome] = int(stress)
                syl.append(nome)

            else:
                in_dict[nomes[j]] = j+1
                syl.append(nomes[j])


        in_l = [in_dict[k] for k in in_dict]
        #out_l = [out_dict[l] for l in out_dict]

        d = [word,syl,tag, nb_syllab]
        d.extend(in_l[:])
        #d.append(out_l[:])
        d.append(out)

        train_data.append(d)


    df = pd.DataFrame()
    labels = ['word', 'sy','pos', 'nb_syllab']
    labels.extend(list(phe_dict.keys()))
    labels.append('stress')

    df = pd.DataFrame.from_records(train_data, columns=labels)

    #NOTE ML
    X = df[df.columns[3:-1]].as_matrix()
    Y = df[df.columns[-1]].as_matrix()
    y_predict = clf.predict(X)

    nb_syl = list(y_predict)

    vows = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']

    syls = list(df[df.columns[1]].as_matrix())
    nb_vows = []
    for i in range(len(syls)):
        nb_vow = 0
        #nb_pos = nb_syl[i]
        for s in syls[i][:nb_syl[i]]:
            if s in vows:
                nb_vow += 1
        nb_vows.append(nb_vow+1)


    return nb_vows
