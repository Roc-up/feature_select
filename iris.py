import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris = datasets.load_iris()
    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 30))

    # Add the noisy data to the informative features
    X = np.hstack((iris.data, E))
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_indices = np.arange(X.shape[-1])

    plt.figure(1)
    # plt.clf()
    plt.subplot(2,1,1)
    # 训练SVC模型
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    print("SVM before selected:{}".format(clf.score(x_test, y_test)))

    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()
    plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')

    # 使用REECV递归消除特征
    my_selector = RFECV(clf, cv=5)
    my_selector.fit(x_train, y_train)
    rfe_weight = my_selector.estimator_.coef_
    print("SVM after selected:{}".format(my_selector.estimator_.score(my_selector.transform(x_test), y_test)))

    svm_weights_selected = (rfe_weight ** 2).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.max()
    plt.bar(X_indices[my_selector.get_support()] - .05, svm_weights_selected, width=.2,
            label='SVM weights after selection', color='b')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    # plt.show()

    old_svm_score = []
    new_svm_score = []
    for i in range(30):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        old_svm_score.append(clf.score(x_test, y_test))
        new_svm_score.append(my_selector.estimator_.score(my_selector.transform(x_test), y_test))

    print(old_svm_score)
    print(new_svm_score)
    plt.subplot(2,1,2)
    plt.plot(range(1, 30 + 1), old_svm_score, color='g', label='old_svm_score')
    plt.plot(range(1, 30 + 1), new_svm_score, color='b', label='new_svm_score')
    plt.ylabel('score')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.show()
