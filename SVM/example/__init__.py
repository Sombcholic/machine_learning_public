# https://cloud.tencent.com/developer/article/2163822?from=15425&areaSource=102001.2&traceId=BH9g_EuBmjHD3qITWiP9V
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import colors
# import matplotlib

from sklearn import svm
from sklearn import model_selection
from sklearn.datasets import load_iris

# 將字符串轉化為整數
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

# 準備data
def prepare_dataset(filepath):
    root = filepath
    data = np.loadtxt(root,
                dtype=float,   # 數據類型
                delimiter=',', # 數據分割符
                converters={4: iris_type}  # 將第五列使用函數進行轉換
            )
    return data

# 分割資料
def split_dataset(data):
    x, y = np.split(data, (4, ), axis = 1)
    x = x[:, :2]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.2)

    return x_train, x_test, y_train, y_test, x, y


# create SVM classifier
def SVM_classifier():
    clf = svm.SVC( C=0.8,  # 誤差懲罰係數
                    kernel='linear',    # 線性高斯核
                    decision_function_shape='ovr' # 決策函數
        )
    return clf

# 訓練模型
def train(clf, x_train, y_train):
    # 訓練集特徵向量 和 訓練集目標值
    clf.fit(x_train, y_train.ravel())

# 呈現準確度
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))

def print_accuracy(clf, x_train, y_train, x_test, y_test):
    print('training prediction:%.3f' % (clf.score(x_train, y_train)))
    print('test data prediction:%.3f' % (clf.score(x_test, y_test)))
    # 原始結果和預測結果進行對比 predict() 表示對x_train樣本進行預測，返回樣本類別
    show_accuracy(clf.predict(x_train), y_train, 'train data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')
    # 計算決策函數的值 表示x到各個分割平面的距離
    print('decision_function:\n', clf.decision_function(x_train))

def draw(clf, x, y):
    iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
    # 開始畫圖
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()

    # 生成網格採樣點
    x1, x2 = np.mgrid[x1_min: x1_max:200j, x2_min:x2_max:200j]
    
    # 測試點
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)
    print('grid_test:\n', grid_test)

    # 輸出樣本到決策面的距離
    z = clf.decision_function(grid_test)
    print('the distance to decision plane:\n', z)
    grid_hat = clf.predict(grid_test)

    # 預測分類值 得到[0, 0, ..., 2, 2]
    print('grid_hat:\n', grid_hat)
    # 使得grid_hat 和 x1形狀一致
    grid_hat = grid_hat.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y),
                edgecolor='k', s=50, cmap=cm_dark)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Iris data classification via SVM', fontsize=30)
    plt.grid()
    plt.show()

    
if __name__ == "__main__":
    # 製作資料
    data = prepare_dataset('D:\python\machine_learning\SVM\example\dataset\iris.data')
    
    # 分割資料集
    x_train, x_test, y_train, y_test, x, y = split_dataset(data)

    # 定義模型
    clf = SVM_classifier()

    # 訓練模型
    train(clf, x_train, y_train)

    print('------------------------eval---------------------')
    print_accuracy(clf, x_train, y_train, x_test, y_test)

    print('---------------------show---------------------')
    draw(clf, x, y)


