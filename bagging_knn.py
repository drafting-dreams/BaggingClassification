from random import randint
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((-1, 1))


def multi_label(raw_df):
    df = raw_df.copy(deep=True)
    df.columns = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'type']
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df


def get_n_sample_df(n, df):
    size = len(df)
    resArr = []
    for ii in range(n):
        temp_df = pd.DataFrame(columns=df.columns)
        for i in range(size):
            randomIndex = randint(0, size - 1)
            temp_df = temp_df.append(df.iloc[randomIndex])
        temp_df.reset_index(drop=True, inplace=True)
        resArr.append(temp_df)
    return resArr


def train_classifiers(num, df):
    arr = get_n_sample_df(num, df)
    result = []

    for bag in arr:
        # bag[bag.columns[:-1]] = scaler.fit_transform(bag[bag.columns[:-1]])
        X_train = bag[bag.columns[:-1]]
        Y_train = (bag['type'])
        knn = KNeighborsClassifier()
        knn = knn.fit(X_train, Y_train)
        result.append(knn)

    return result


def getResultFromClassfiers(cs, X):
    predictions = []
    votedPredictions = []
    for c in cs:
        res = c.predict(X)
        predictions.append(res)

    predictionsNum = len(predictions)
    listLen = len(predictions[0])

    for i in range(listLen):
        d = {}
        for ii in range(predictionsNum):
            result = predictions[ii][i]
            if result in d.keys():
                d[result] += 1
            else:
                d[result] = 1

        print(d)
        votedPredictions.append(max(d.items(), key=lambda x: x[1])[0])

    return votedPredictions


def bagging_knn(classierNum, df_train, df_test):
    df_train = multi_label(df_train)
    df_test = multi_label(df_test)

    cs = train_classifiers(classierNum, df_train)

    X_test = df_test[df_test.columns[:-1]]
    Y_test = df_test['type']

    getResultFromClassfiers(cs, X_test)


df_train = pd.read_csv('glass.data')
df_test = pd.read_csv('glass.test')

bagging_knn(3, df_train, df_test)
