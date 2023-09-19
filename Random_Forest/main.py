from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



if __name__ == "__main__":
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    rfc.fit(X_train, y_train)

    y_predict=rfc.predict(X_test)
    print(y_predict)

    print(rfc.score(X_test, y_test))

    imp=rfc.feature_importances_
    print(imp)

    names = iris.feature_names

    zip(imp, names)
    imp, names = zip(*sorted(zip(imp, names)))


    # 調整畫面大小
    plt.figure(figsize=(8, 8.5))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance of Features')
    plt.ylabel('Features')
    plt.title('Importance of Each Feature')
    plt.show()
