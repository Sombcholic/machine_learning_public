from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import datasets
import pydotplus


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3).fit(X_train, y_train)

    print(clf.score(X_train, y_train))

    a = clf.predict(X_test)
    print(a)

    print(clf.score(X_test, y_test))

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris_gini_max3.pdf')

    