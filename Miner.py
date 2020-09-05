import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def convert_keys(array):
    data = []
    for i in range(0, len(array["key"])):
        if array["key"][i] == 0.0:
            data.append("C")
        elif array["key"][i] == 1.0:
            data.append("C#")
        elif array["key"][i] == 2.0:
            data.append("D")
        elif array["key"][i] == 3.0:
            data.append("D#")
        elif array["key"][i] == 4.0:
            data.append("E")
        elif array["key"][i] == 5.0:
            data.append("F")
        elif array["key"][i] == 6.0:
            data.append("F#")
        elif array["key"][i] == 7.0:
            data.append("G")
        elif array["key"][i] == 8.0:
            data.append("G#")
        elif array["key"][i] == 9.0:
            data.append("A")
        elif array["key"][i] == 10.0:
            data.append("A#")
        elif array["key"][i] == 11.0:
            data.append("B")
        else:
            data.append("U")
    array["converted_key"] = data


def convert_modes(array):
    data = []
    for i in range(0, len(array["key"])):
        if array["mode"][i] == 1:
            data.append("Major")
        elif array["mode"][i] == 0:
            data.append("Minor")
        else:
            data.append("Unknown")
    array["converted_mode"] = data


def find_tune(array):
    data = []
    for i in range(0, len(array["key"])):
        temp = ""
        if array["converted_mode"][i] == "Major":
            temp = ""
        elif array["converted_mode"][i] == "Minor":
            temp = "m"
        data.append(array["converted_key"][i] + temp)
    array["tune"] = data


def count_tunes(array):
    temp = array["tune"].value_counts()
    temp.plot(kind="bar")
    return temp


def count_artists(array):
    temp_array = array.groupby(["artist"])
    temp = temp_array["artist"].count()
    values = temp.sort_values(ascending=False).head(10)
    values = values[values.values > 2]
    values.plot(kind="barh")
    return values


def classify_tempos(array):
    data = []
    for i in range(0, len(array)):
        if array["tempo"][i] >= 200:
            data.append("Prestissimo")
        elif array["tempo"][i] >= 168:
            data.append("Presto")
        elif array["tempo"][i] >= 120:
            data.append("Allegro")
        elif array["tempo"][i] >= 108:
            data.append("Moderato")
        elif array["tempo"][i] >= 76:
            data.append("Andante")
        elif array["tempo"][i] >= 66:
            data.append("Adagio")
        elif array["tempo"][i] >= 60:
            data.append("Larghetto")
        elif array["tempo"][i] >= 45:
            data.append("Lento")
        elif array["tempo"][i] <= 45:
            data.append("Largo")
        else:
            data.append("Unknown")
    array["tempo_name"] = data


def normalize_tempo_names(array):
    data = []
    for i in range(0, len(array)):
        if array["tempo_name"][i] == "Prestissimo":
            data.append(0)
        elif array["tempo_name"][i] == "Presto":
            data.append(1)
        elif array["tempo_name"][i] == "Allegro":
            data.append(2)
        elif array["tempo_name"][i] == "Moderato":
            data.append(3)
        elif array["tempo_name"][i] == "Andante":
            data.append(4)
        elif array["tempo_name"][i] == "Adagio":
            data.append(5)
        elif array["tempo_name"][i] == "Larghetto":
            data.append(6)
        elif array["tempo_name"][i] == "Lento":
            data.append(7)
        elif array["tempo_name"][i] == "Largo":
            data.append(8)
    array["ntempo_name"] = data


def normalize_tunes(array):
    data = []
    for i in range(0, len(array)):
        if array["tune"][i] == "C":
            data.append(0)
        elif array["tune"][i] == "C#":
            data.append(1)
        elif array["tune"][i] == "D":
            data.append(2)
        elif array["tune"][i] == "D#":
            data.append(3)
        elif array["tune"][i] == "E":
            data.append(4)
        elif array["tune"][i] == "F":
            data.append(5)
        elif array["tune"][i] == "F#":
            data.append(6)
        elif array["tune"][i] == "G":
            data.append(7)
        elif array["tune"][i] == "G#":
            data.append(8)
        elif array["tune"][i] == "A":
            data.append(9)
        elif array["tune"][i] == "A#":
            data.append(10)
        elif array["tune"][i] == "B":
            data.append(11)
        elif array["tune"][i] == "Cm":
            data.append(0)
        elif array["tune"][i] == "C#m":
            data.append(1)
        elif array["tune"][i] == "Dm":
            data.append(2)
        elif array["tune"][i] == "D#m":
            data.append(3)
        elif array["tune"][i] == "Em":
            data.append(4)
        elif array["tune"][i] == "Fm":
            data.append(5)
        elif array["tune"][i] == "F#m":
            data.append(6)
        elif array["tune"][i] == "Gm":
            data.append(7)
        elif array["tune"][i] == "G#m":
            data.append(8)
        elif array["tune"][i] == "Am":
            data.append(9)
        elif array["tune"][i] == "A#m":
            data.append(10)
        elif array["tune"][i] == "Bm":
            data.append(11)
    array["ntune"] = data


def find_new_songs(array):
    predictors = array[["valence",'acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'loudness',
       'speechiness', 'ntempo_name', "ntune"]]
    target = array["like"]

    x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.40, random_state=0)
    models = [('Logistic Regression', LogisticRegression()), ('Naive Bayes', GaussianNB()),
              ('Decision Tree (CART)', DecisionTreeClassifier()), ('K-NN', KNeighborsClassifier()),
              ('Gradient Boosting Classifier', GradientBoostingClassifier()),
              ('AdaBoostClassifier', AdaBoostClassifier()), ('BaggingClassifier', BaggingClassifier()),
              ('RandomForestClassifier', RandomForestClassifier())]

    for name, model in models:
        model = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        from sklearn import metrics

        print("%s -> Hassasiyet: %%%.2f" % (name, metrics.accuracy_score(y_test, y_pred) * 100))

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    top2018 = pd.read_csv("top2018.csv", encoding="ISO-8859-1")
    top2017 = pd.read_csv("top2017.csv", encoding="ISO-8859-1")
    s = pd.concat([top2018, top2017], axis=0, ignore_index=True)
    convert_modes(s)
    convert_keys(s)
    find_tune(s)
    classify_tempos(s)
    normalize_tempo_names(s)
    normalize_tunes(s)
    counter = 0
    for i in range(0,len(s)):
        row = np.array([s[["valence",'acousticness', 'danceability', 'energy',
                                 'instrumentalness', 'liveness', 'loudness',
                                 'speechiness', 'ntempo_name', "ntune"]].iloc[i]])
        row = row.reshape(1, -1)
        y_pred = rfc.predict(row)
        if y_pred[0] == 1:
            counter += 1
            print(counter,")",s["name"].iloc[i],"-",s["artists"].iloc[i])
    print("Toplam Eşleşen Şarkı Sayısı:",counter)


def finder(array):
    convert_modes(array)
    convert_keys(array)
    array = array[array["converted_key"] != "U"]
    array = array[array["converted_mode"] != "Unknown"]
    find_tune(array)
    classify_tempos(array)
    array.drop(['duration_ms', 'time_signature', 'key', 'mode','tempo','Unnamed: 0'], axis=1, inplace=True)
    array = array[array["tempo_name"] != "Unknown"]
    normalize_tempo_names(array)
    normalize_tunes(array)
    maxTune = count_tunes(array).idxmax()
    plt.show()
    count_artists(array)
    plt.show()
    plt.figure(figsize=(20, 15))
    mask = np.zeros_like(array.corr())
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(array.corr(),
                xticklabels=array.corr().columns.values,
                yticklabels=array.corr().columns.values,
                linewidths=.9,
                vmin=-1,
                vmax=1,
                mask=mask,
                cmap='coolwarm',
                annot=True)
    plt.show()
    find_new_songs(array)


data = pd.read_csv("data.csv", encoding="ISO-8859-1")
finder(data)
exit()
