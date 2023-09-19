# https://www.kaggle.com/code/lena0104/heart-disease-dt-rf-svm-knn-lr-nb-dnn
# dataset ==> https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import sys


def load_data():
    data = pd.read_csv("./input/heart.csv")
    # print(data.head())
    # plt.figure(figsize=(18,10))
    # sns.heatmap(data.corr(),annot=True,cmap='viridis')
    # plt.show()

    return data

def preprocess_data(data):
    # data['cp']有0,1,2,3，刪掉第一個為了避免同質性，所以會多出三個cp，做1-hot
    df_cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)
    df_thal = pd.get_dummies(data['thal'], prefix = "thal", drop_first=True)
    df_slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
    df_sex = pd.get_dummies(data['sex'], prefix = "sex", drop_first=True)
    df_fbs = pd.get_dummies(data['fbs'], prefix = "fbs", drop_first=True)
    df_restecg = pd.get_dummies(data['restecg'], prefix = "restecg", drop_first=True)
    df_exang = pd.get_dummies(data['exang'], prefix = "exang", drop_first=True)

    # 剩下沒有做同質性處理的資料
    df_num = data.drop(['cp','thal','slope','sex','fbs','restecg','exang'], axis=1)


    frames = [df_num,df_cp, df_thal, df_slope, df_sex,df_fbs,df_restecg,df_exang]
    df = pd.concat(frames, axis = 1)
    df.info()
    
    return df


def scale_df(df):
    # 初始化 MinMaxScaler，并指定自定义的归一化区间
    # scaler = MinMaxScaler(feature_range=(custom_min, custom_max))
    
    scale = MinMaxScaler()

    df[['age','trestbps','chol','thalach','oldpeak','ca']] = pd.DataFrame(scale.fit_transform(df[['age','trestbps','chol','thalach','oldpeak','ca']].values), 
                                                                            columns=['age','trestbps','chol','thalach','oldpeak','ca'], 
                                                                            index=df.index)

    # print('第二個')
    # data = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
    # print(scale.fit_transform(data))

    return df


def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target', axis=1), df['target'], test_size = 0.33, random_state=10) #split the data

    return X_train, X_test, y_train, y_test

def rate(pred):
    cfm = confusion_matrix(y_test, pred)
    cfm = cfm.astype(np.float64)
    total=sum(sum(cfm))
    sensitivity = cfm[0,0]/(cfm[0,0]+cfm[1,0])
    specificity = cfm[1,1]/(cfm[1,1]+cfm[0,1])
    accuracy = (cfm[0,0]+cfm[1,1])/total
    dt={
        "Accuracy": accuracy,
        "Sensivity": sensitivity,
        "Specificity": specificity
    }    
    return dt

# create decision_tree model and use it
class decision_tree():
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = DecisionTreeClassifier()
    
    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train,y_train)

    def predict(self, target):
        dtree_predictions = self.model.predict(target)
        
        return dtree_predictions

# create Random Forest
# n_estimators = 林中樹木的數量
class randon_forest():
    def __init__(self, n_estimators):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = RandomForestClassifier(n_estimators = n_estimators)

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, target):
        rforest_predictions = self.model.predict(target)

        return rforest_predictions

class SVM():
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = SVC()

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, target):
        rforest_predictions = self.model.predict(target)

        return rforest_predictions

class KNN():
    def __init__(self, n_neighbors):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = KNeighborsClassifier(n_neighbors = n_neighbors)

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, target):
        knn_predictions = self.model.predict(target)

        return knn_predictions

class LR():
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, target):
        lr_predictions = self.model.predict(target)

        return lr_predictions

class NB():
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, target):
        nb_predictions = self.model.predict(target)

        return nb_predictions

if __name__ == '__main__':
    data = load_data()
    df = preprocess_data(data)

    df = scale_df(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print(X_train)
    sys.exit(0)

    # create decision_tree
    dtree = decision_tree()
    dtree.train(X_train, y_train)
    dtree_predictions = dtree.predict(X_test)
    dtree_report = pd.DataFrame(rate(dtree_predictions), index=['Decision Tree'])
    # print(dtree_report)

    # create randon_forest
    rforest = randon_forest(n_estimators = 800)
    rforest.train(X_train, y_train)
    rforest_predictions = rforest.predict(X_test)
    score = cross_val_score(rforest.model, X_train, y_train, cv=10)
    # print(score)
    # print(rforest_predictions)

    # create Support Vector Machine
    svm = SVM()
    svm.train(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    svm_report = pd.DataFrame(rate(svm_predictions), index=['Support Vector Machine'])
    # print(svm_predictions)
    # print(classification_report(y_test, svm_predictions))
    # print(svm_report)

    # create K-Nearest neighbor (KNN)
    knn = KNN(n_neighbors=34)
    knn.train(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_report = pd.DataFrame(rate(knn_predictions), index=['K_Nearest Neighbor'])
    # print(knn_predictions)

    # create Logistic Regression
    lr = LR()
    lr.train(X_train, y_train)
    lr_predictions = lr.predict(X_test)
    # print(classification_report(y_test, lr_predictions))

    # create Naive Bayes
    nb = NB()
    nb.train(X_train, y_train)
    nb_predictions = nb.predict(X_test)
    print(classification_report(y_test, nb_predictions))

