
"""
    File contenente la totalita' delle funzionalita' del progetto.
    Nello specifico vengono gestite le richieste che il client manda al server,
    richieste sia POST che GET.
    Viene creato il modello classificatore, etichettato il nuovo messaggio e
    restituiti i possibili suggerimenti per migliorarlo.
"""

import os
import pickle
import seaborn
import random

import numpy
from flask import request, jsonify

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import matplotlib.pyplot as plt

from dashboard import app
import run

# Variabili globali
allLikes = []
allRetweets = []
numBad = 0
knnModel = KNeighborsClassifier()

labels = []
scaler = preprocessing.RobustScaler()

COLUMNS = []
NUM_RUN = 10
CODA = 0.2        # grandezza code degli estremi
OTHERS = 0.0
numGruppoFile = 1

MIN_LIKES = 500
MAX_LIKES = 0

good_df = pd.DataFrame()
df = pd.DataFrame()

# Gruppi di musei
group_one = []
group_two = []
group_three = []
group_four = []
group_five = []
group_six = []

# Numero di tweet da generare
n_tweet = 50


# GET METHODS

"""
 Ritorna l'elenco totale dei musei, in fase di apertura dell'app
"""
@app.route('/museums', methods=['GET'])
def museums():
    global group_one, group_two, group_three, \
            group_four, group_five, group_six

    group_one = []
    group_two = []
    group_three = []
    group_four = []
    group_five = []
    group_six = []

    fname = "data/museums.txt"
    readFile(fname)

    if (run.DEBUG):
        print (group_one)
        print (group_two)
        print (group_three)
        print (group_four)
        print (group_five)
        print (group_six)

    musei = group_one + group_two + group_three + \
        group_four + group_five + group_six

    return jsonify (musei = musei)


"""
 Legge i file di testo contenenti i nomi dei musei,
 divisi per gruppi
"""
def readFile (fname):

    i = 1
    with open(fname) as f:
        for line in f:
            parola = line.split('-')

            for j in range (0, len(parola)):
                if (j == len(parola)-1):
                    parola[j] = parola[j].rstrip()

                if (i == 1):
                    group_one.append(parola[j])
                elif (i == 2):
                    group_two.append(parola[j])
                elif (i == 3):
                    group_three.append(parola[j])
                elif (i == 4):
                    group_four.append(parola[j])
                elif (i == 5):
                    group_five.append(parola[j])
                else:
                    group_six.append(parola[j])

            i = i + 1




# POST METHODS

"""
 In base al museo scelto, imposta il classificatore sui
 dati del gruppo apposito
"""
@app.route('/groups', methods=['POST'])
def groups():
    global numGruppoFile, knnModel, scaler

    nome_museo = request.get_json()["name"]
    numGruppoFile = searchGroup(nome_museo)

    # Crea il classificatore sui dati del gruppo
    fname = "media/model" + str(numGruppoFile) + ".pkl"
    scalername = "media/scaler" + str(numGruppoFile) + ".pkl"

    if (os.path.isfile(fname)):
        knnModel = pickle.load(open(fname, "rb"))
        scaler = pickle.load(open(scalername, "rb"))
    else:
        train()

    return jsonify(gruppo = numGruppoFile)


"""
  Ritorna il gruppo a cui appartiene il museo,
  passato come argomento
"""
def searchGroup(nome_museo):
    if (run.DEBUG):
        print (nome_museo)

    if nome_museo in group_one:
        return 1
    elif nome_museo in group_two:
        return 2
    elif nome_museo in group_three:
        return 3
    elif nome_museo in group_four:
        return 4
    elif nome_museo in group_five:
        return 5
    else:
        return 6


"""
 Ritorna le caratteristiche per l'analisi del tweet
"""
def feature ():

    global COLUMNS

    COLUMNS = [
        #'NURLS',
        'NIMG',
        'NHASH',
        'NMENTION',
        #'LENGTH',
        #'DENSE',
        #'SENT',
        #'LONG'
    ]


"""
  Associa l'etichetta riguardante l'andamento del tweet
"""
def classes (riga):
    global numBad, allLikes
    c1 = 'no'

    likes = allLikes[riga]

    if likes > MAX_LIKES:
        c1 = 'good'
    else:
        c1 = 'bad'
        numBad = numBad + 1

    return c1


"""
  Apre il file contenente i tweet di un relativo
  gruppo di musei e crea il modello
"""
def train():
    global numGruppoFile, knnModel

    # Sceglie il file
    file = "threads-g" + str(numGruppoFile) + ".txt"
    path = os.path.abspath("data/" + file)

    if (run.DEBUG):
        print("FILE: ", numGruppoFile)

    # Apre il file e lo divide per colonne
    data = pd.read_csv(path, sep='\t')

    # Viene creata la matrice di correlazione tra le caratteristiche
    name = 'media/img/correlationMatrix' + str(numGruppoFile) + '.png'
    if not (os.path.isfile(name)):
        corr = data.corr()

        mask = numpy.zeros_like(corr, dtype=numpy.bool)
        mask[numpy.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))

        cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
        plt.figure(figsize=(16, 9))
        seaborn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.savefig(name)

    global allLikes, MAX_LIKES, MIN_LIKES, \
        CODA, COLUMNS, numBad
    riga = 0
    for index, row in data.iterrows():
        allLikes.append(data.iloc[riga]['NLIKE'])
        riga = riga + 1

    lenAll = len(allLikes)
    numGood = len(allLikes)
    while numGood > lenAll * CODA:
        MAX_LIKES = MAX_LIKES + 1
        numGood = sum(i > MAX_LIKES for i in allLikes)

    numBad = len(allLikes)
    while numBad > lenAll * CODA:
        MIN_LIKES = MIN_LIKES - 1
        numBad = sum(i < MIN_LIKES for i in allLikes)

    if (run.DEBUG):
        print("THREADS - num righe -: ", str(lenAll))
        print("MAX_LIKES: " + str(MAX_LIKES))
        print("MIN_LIKES: " + str(MIN_LIKES))

    global labels, scaler

    feature()
    for i in range(0, riga):
        label = classes(i)
        labels.append(label)

    with open("media/labels" + str(numGruppoFile)+".txt", 'wb') as f:
        pickle.dump(labels, f)

    # X = Caratteristiche
    # y = Etichette
    X = data[COLUMNS].copy()
    y = numpy.array(labels)

    acc = []
    for indx in range(0, NUM_RUN):
        if (run.DEBUG):
            print("NUMRUN: ", str(indx))

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.20,
                                                            random_state=indx,
                                                            shuffle=True)
        # Scala i dati
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Calcola il migliore k, per la previsione, a fine di avere il minor errore
        # su scelta arbitraria tra 1 e 30
        accuracy = []
        for i in range(1, 30):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            accuracy.append(metrics.accuracy_score(y_test, pred_i))

        optimal_k = 5
        max_acc = max(accuracy)
        for i in range(0, len(accuracy)):
            if (accuracy[i] == max_acc):
                optimal_k = i
                break

        if (run.DEBUG):
            print("k: ", optimal_k)

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 30), accuracy)
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Accuracy')
        plt.title('Run ' + str(indx) + ' - Group '+ str(numGruppoFile) + '\n')
        file = "media/img/k_run/gruppo"+str(numGruppoFile)+"/"+ str(indx) + ".png"
        plt.savefig(file)

        knnModel = KNeighborsClassifier(n_neighbors=optimal_k, weights='uniform')

        # Riempie il modello usando X come training e y come valori target
        knnModel.fit(X_train, y_train)

        # Associa l'etichetta della classe ai dati inseriti
        y_pred = knnModel.predict(X_test)

        acc.append(metrics.accuracy_score(y_test, y_pred))

    if (run.DEBUG):
        print("KNN prediction accuracy (medium):", sum(acc) / len(acc))

    # Esporta il classificatore
    pickle.dump(knnModel, open("media/model" + str(numGruppoFile) + ".pkl", "wb"))

    # Esporta lo scaler
    pickle.dump(scaler, open("media/scaler" + str(numGruppoFile) + ".pkl", "wb"))


"""
  Metodo principale di confronto che riceve il nuovo 
  messaggio e lo analizza in base al modello creato in precedenza.
"""
@app.route('/api', methods=['POST'])
def api():
    global good_df, df
    raw_data = request.get_json()

    result = analyzeText(raw_data, False, None)
    suggestion = getSuggestion(good_df, df, False)

    if (run.DEBUG):
        print (suggestion)

    return jsonify(result = result,
                   suggestion = suggestion)


"""
    Prende in ingresso il JSON Object e ne crea un dataframe in PANDAS,
    che poi viene confrontato con il modello per ottenere la prediction.
    param raw_data  : presente solo se la funzione non e' usata dal generatore,
                        rappesenta l'oggetto JSON da utlizzare.
    param generator : parametro booleano per indicare o meno che la funzione sia
                       usata dal generatore o no.
    param generator_row : riga del DataFrame che indica uno specifico tweet generato.
"""
def analyzeText(raw_data, generator, generator_row):

    global COLUMNS, good_df, df, numGruppoFile, \
            knnModel, scaler

    file = "threads-g" + str(numGruppoFile) + ".txt"
    path = os.path.abspath("data/" + file)
    if (run.DEBUG):
        print("FILE: ", numGruppoFile)

    feature()
    DATA = pd.read_csv(path, sep='\t')
    DATA = DATA[COLUMNS].copy()

    df = df.iloc[0:0]
    good_df = good_df.iloc[0:0]

    # Dataframe che viene creato dai dati in ingresso
    if (generator == False):
        df = pd.DataFrame(raw_data, columns=COLUMNS, index=[0])

    if (run.DEBUG):
        if (generator == False):
            print ("dataframe: ", df)

    # Scala i dati e calcola la label del messaggio
    if (generator == False):
        DF_scaled = scaler.transform(df)
        label = knnModel.predict(DF_scaled)

        # Ritorno vicini
        dist, ind = knnModel.kneighbors(X=DF_scaled, n_neighbors=15, return_distance=True)
    else:
        dist, ind = knnModel.kneighbors(X=generator_row, n_neighbors=15, return_distance=True)

    if (run.DEBUG):
        for i in range (0, len(dist[0])):
            print ("Distanze: ", dist[0][i])

    # good_df e' un dataframe che contiene i dati dei soli
    # tweet positivi
    good_df = pd.DataFrame(columns=COLUMNS)

    with open("media/labels"+ str(numGruppoFile)+".txt", 'rb') as f:
        labels = pickle.load(f)

    # Per tutta la lunghezza del df dei vicini,
    # vengono filtrati in base ai giudizi positivi e alla distanza
    for i in range(0, len(ind[0])):
        if (labels[ind[0][i]] == 'good' and dist[0][i] <= 1.0):
            good_df.loc[len(good_df)] = DATA.iloc[ind[0][i]]

    if (run.DEBUG):
        if (generator == False):
            print ("LABEL NUOVO: ", label)

    if (generator == False):
        if (label == 'bad'):
            return "Not good"
        else:
            return "Good"


"""
  Metodo che permette di analizzare i messaggi positivi vicini e, 
  da questi, creare dei suggerimenti.
  param pos_df : DataFrame con i dati dei vicini positivi
  param df_messaggio : DataFrame con i dati del messaggio da confrontare
  param generator: parametro booleano per indicare se questa funzione viene
               usata dal generatore o meno
"""
def getSuggestion(pos_df, df_messaggio, generator):

    if (run.DEBUG):
        print (pos_df)

    sugg = []
    if pos_df.empty:
        sugg.append("No suggestions available")

    else:
        if (generator == False):
            nimg = df_messaggio.iloc[0]['NIMG']
            nhash = df_messaggio.iloc[0]['NHASH']
            nmention = df_messaggio.iloc[0]['NMENTION']
        else:
            nimg = df_messaggio['NIMG']
            nhash = df_messaggio['NHASH']
            nmention = df_messaggio['NMENTION']

        med_img = sum(pos_df['NIMG']) / len(pos_df)
        med_img = int(med_img+0.5)

        med_hash = sum(pos_df['NHASH']) / len(pos_df)
        med_hash = int(med_hash+0.5)

        med_mentions = sum(pos_df['NMENTION']) / len(pos_df)
        med_mentions = int(med_mentions+0.5)

        if (med_img >= nimg):
            sugg.append("Add an image")
        elif (med_img < nimg):
            sugg.append("Remove an image")

        if (med_hash >= nhash):
            sugg.append("Add an hashtag")
        elif (med_hash < nhash):
            sugg.append("Remove an hashtag")

        if (med_mentions >= nmention):
            sugg.append("Add a mention")
        elif (med_mentions < nmention):
            sugg.append("Remove a mention")

    return sugg


"""
  Funzione che crea e analizza una porzione randomica di
  50 tweet, seguendo i suggerimenti ricevuti.
  Di questa analisi ne crea un grafico che permette di 
  sottolineare un eventuale andamento crescente.
"""
@app.route('/generator', methods=['GET'])
def generator ():
    global COLUMNS, knnModel, scaler, good_df, \
        numGruppoFile, n_tweet

    file_grafico = "media/img/GOOD_" + str(numGruppoFile) + ".png"
    if (not os.path.isfile(file_grafico)):

        file = "threads-g" + str(numGruppoFile) + ".txt"
        path = os.path.abspath("data/" + file)
        DATA = pd.read_csv(path, sep='\t')

        max_hash = DATA['NHASH'].max()
        max_mention = DATA['NMENTION'].max()

        tweet = pd.DataFrame(columns=COLUMNS, dtype='int32')
        sugg = []

        # Probabilita' che esca 'good', da graficare per ogni tweet generato.
        # Matrice:  righe = tweet
        #           colonne = suggerimenti
        prob_good = []

        for i in range (0, n_tweet):
            if (run.DEBUG):
                print ("--- TWEET NUM --- ", i)

            sugg.clear()
            good = []  # Array per i singoli suggerimenti

            # Crea i tweet con scelte randomiche
            NIMG = random.randint(0, 4)
            NHASH = random.randint(0, max_hash)
            NMENTION = random.randint(0, max_mention)

            tweet.loc[i, 'NIMG'] = NIMG
            tweet.loc[i, 'NHASH'] = NHASH
            tweet.loc[i, 'NMENTION'] = NMENTION

            tweet['NIMG'] = tweet['NIMG'].astype(int)
            tweet['NHASH'] = tweet['NHASH'].astype(int)
            tweet['NMENTION'] = tweet['NMENTION'].astype(int)

            for j in range (0, 6):
                sugg.clear()

                # Analizza di nuovo il tweet
                tweetArray = scaler.transform(tweet.loc[i].values.reshape(1, -1))
                label = knnModel.predict(tweetArray)
                prob = knnModel.predict_proba(tweetArray)

                good.append(prob[0][1])

                # Trova gli elementi vicini
                analyzeText(None, True, tweetArray)

                # Ritorna suggerimenti
                sugg.append(getSuggestion(good_df, tweet.loc[i], True))
                if (run.DEBUG):
                    print (sugg)

                # Compie un suggerimento
                ind_sugg_scelto = random.randint(0, 2)
                doAction (sugg[0], tweet.loc[i], ind_sugg_scelto,
                              max_hash, max_mention)

            prob_good.append(good)

        median_good = []
        somma = []
        for j in range (0, 6):
            s = 0
            for i in range (0, n_tweet):
                s = s + prob_good[i][j]
            somma.append(s)

        for i in range (0, len(somma)):
            median_good.append(somma[i]/n_tweet)

        if (run.DEBUG):
            print ("somma: ", somma)
            print ("media: ", median_good)

        # Crea grafico
        plt.figure(figsize=(12, 6))
        plt.plot(range (0, 6), median_good)
        plt.xlabel('# Suggestion')
        plt.ylabel('Probability for GOOD (%)\n')
        plt.title('Trend of the generator on ' + str(numGruppoFile) + '° group\n'
                  + " - 50 tweet -\n")
        plt.savefig(file_grafico)

    # Questo tipo di metodo deve avere un valore di ritorno
    return jsonify(result="Generatore creato")



"""
  Compie le azioni suggerite, scelte in maniera randomica,
  anche in caso di 'No suggestions available'
  @param: suggerimento: Rappresenta l'array di suggerimenti per il tweet specifico
  @param: tweet: Messaggio da cambiare
  @param: ind_sugg_scelto: Indice del suggerimento da seguire
"""
def doAction (suggerimento, tweet, ind_sugg_scelto, max_hash, max_mention):

    if (tweet['NHASH'] >= max_hash):
        tweet['NHASH'] = tweet['NHASH'] - 1
        return
    elif (tweet['NMENTION'] >= max_mention):
        tweet['NMENTION'] = tweet['NMENTION'] - 1
        return

    # Caso: ci sono suggerimenti
    if (len(suggerimento) > 1):
        if ind_sugg_scelto == 0:
            if "Remove" in suggerimento[ind_sugg_scelto]:
                tweet['NIMG'] = tweet['NIMG'] - 1
            else:
                if (tweet['NIMG'] < 4):
                    tweet['NIMG'] = tweet['NIMG'] + 1
        elif ind_sugg_scelto == 1:
            if "Remove" in suggerimento[ind_sugg_scelto]:
                tweet['NHASH'] = tweet['NHASH'] - 1
            else:
                tweet['NHASH'] = tweet['NHASH'] + 1
        else:
            if "Remove" in suggerimento[ind_sugg_scelto]:
                tweet['NMENTION'] = tweet['NMENTION'] - 1
            else:
                tweet['NMENTION'] = tweet['NMENTION'] + 1

    # Caso: non ci sono suggerimenti
    else:
        if ind_sugg_scelto == 0:
            if tweet['NIMG'] == 0:
                tweet['NIMG'] = tweet['NIMG'] + 1
            else:
                scelta = random.randint(0, 2)
                if (scelta == 0 or tweet['NIMG'] == 4):
                    tweet['NIMG'] = tweet['NIMG'] - 1
                else:
                    tweet['NIMG'] = tweet['NIMG'] + 1
        elif ind_sugg_scelto == 1:
            if tweet['NHASH'] == 0:
                tweet['NHASH'] = tweet['NHASH'] + 1
            else:
                scelta = random.randint(0, 2)
                if (scelta == 0):
                    tweet['NHASH'] = tweet['NHASH'] - 1
                else:
                    tweet['NHASH'] = tweet['NHASH'] + 1
        else:
            if tweet['NMENTION'] == 0:
                tweet['NMENTION'] = tweet['NMENTION'] + 1
            else:
                scelta = random.randint(0, 2)
                if (scelta == 0):
                    tweet['NMENTION'] = tweet['NMENTION'] - 1
                else:
                    tweet['NMENTION'] = tweet['NMENTION'] + 1
