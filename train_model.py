from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False


class Model:

    def __init__(self, data, target, class_names=[]):
        self.data = data
        self.target = target
        self.feature_names = data.columns
        self.class_names = ['不畅销', '畅销']

        # 将data和target转换为数组
        self.data = self.data.to_numpy()
        print("data数组如下:{}".format(self.data))
        self.target = self.target.array
        print("target数组如下:{}".format(self.target))

        # 划分训练集和测试集
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.target, test_size=0.3, random_state=30)

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.target, test_size=0.3)

    # 交叉验证
    def cross_validation(self, model, cv=3):
        score = cross_val_score(model, self.x_train, self.y_train, cv=cv)
        return score

    def grid_search_cv(self, param_grid, model, cv):
        gs = GridSearchCV(model, param_grid=param_grid, cv=cv)
        gs.fit(self.x_train, self.y_train)
        return gs.best_params_, gs.best_score_

    def draw_validation_curve(self, clf, param_name, param_range):
        # self.split_data()
        train_scores, test_scores = validation_curve(
            clf, self.x_train, self.y_train, param_name=param_name, param_range=param_range,
            scoring="accuracy", n_jobs=1, cv=5)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve")
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)

        lw = 2
        plt.axhline(1, color='r')
        # plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.semilogx(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        # plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.rc('font', family='SimHei', size=12)
        plt.show()

    def draw_compare_curve(self, n_range=10, curve1=[], name1='',
                           curve2=[], name2='', x_label='', y_label='', y_range=[0,1]):
        fig = plt.figure()
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(range(1, n_range + 1), curve1, color='g', label=name1)
        ax1.plot(range(1, n_range + 1), curve2, color='b', label=name2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        ax1.set_ylim(y_range)
        plt.show()

    def draw_feature_importance(self, importances, feature_name, title='', show_n_features=10):
        plt.figure()
        plt.title(title)
        indices = np.argsort(importances)[::-1]
        indices = indices[:show_n_features]
        names = [feature_name[i] for i in indices]
        plt.bar(range(len(names)), importances[indices], color='lightblue', align='center')
        plt.xticks(range(len(names)), names, rotation=90)
        plt.xlim([-1, len(names)])
        plt.tight_layout()
        plt.rc('font', family='SimHei')
        plt.show()

    def draw_compare_importance(self, importances1, importances2, feature_name, label1='1', label2='2'):
        plt.figure()
        plt.title('Feature Importance')
        indices1 = np.argsort(importances1)[::-1]
        indices2 = np.argsort(importances2)[::-1]

        plt.bar(np.arange(self.x_train.shape[1]) - .25, importances1[indices1], color='blue', align='center')
        plt.bar(np.arange(self.x_train.shape[1]) - .05, importances2[indices2], color='red', align='center')

