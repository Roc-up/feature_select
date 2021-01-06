from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import joblib
from load_data import DataHandle
from train_model import Model
import numpy as np
import os


# 用于字符编码的字典
types_dict = {'手机型号': 0, '屏幕类型': 1, '分辨率': 0, '芯片型号': 0, 'GPU': 0,
              '操作系统': 0, '充电接口': 1, '网络类型': 1, '5G': 1}

# 用于连续值离散化的字典
series_dict = {'上市时间': [0, 3, 'ordinal', 'quantile'],
               '最低价格': [0, 5, 'ordinal', 'quantile'],
               '最高价格': [0, 5, 'ordinal', 'quantile'],
               '重量': [0, 3, 'ordinal', 'quantile'],
               '屏幕尺寸': [0, 3, 'ordinal', 'quantile'],
               'CPU主频率': [0, 3, 'ordinal', 'quantile'],
               '电池容量': [0, 3, 'ordinal', 'quantile']}

# 用于数据无量纲化的字典
dimension_dict = {'上市时间': 0, '最低价格': 0, '最高价格': 0,  '重量': 0, '屏幕尺寸': 0,
                  'CPU主频率': 0, '电池容量': 0, '最小RAM': 0, '最大RAM': 0, '最小ROM': 0,
                  '最大ROM': 0, 'CPU得分': 0, '摄像头数': 0, '后置主像素': 0, '前置主像素': 0,
                  '充电功率': 0, '无线充电': 0}


class FeatureSelect:

    def __init__(self):
        self.selector_rf = None
        self.selector_svm = None
        self.x_selected_rf = None
        self.x_selected_svm = None
        self.rfc = RandomForestClassifier(criterion='gini'
                                          , n_estimators=191
                                          , max_depth=13
                                          # , max_features=6
                                          , min_samples_leaf=1
                                          , random_state=30
                                          )
        self.svc = SVC(kernel='linear',
                       C=0.007564633275546291,
                       tol=0.0001,
                       cache_size=1000,
                       random_state=30
                       )
        self.dh = DataHandle('data/phone_data.csv')

        self.dh.read_data('data/smote_rf_data.csv')
        self.model_rf = Model(self.dh.get_data(), self.dh.get_target())

        self.dh.read_data('data/smote_svm_data.csv')
        self.model_svm = Model(self.dh.get_data(), self.dh.get_target())

    def set_svm_dataset(self, path):
        self.dh.read_data(path)
        self.model_svm = Model(self.dh.get_data(), self.dh.get_target())

    def set_rf_dataset(self, path):
        self.dh.read_data(path)
        self.model_rf = Model(self.dh.get_data(), self.dh.get_target())

    def get_rf_score(self):
        self.rfc.fit(self.model_rf.x_train, self.model_rf.y_train)
        return self.rfc.score(self.model_rf.x_test, self.model_rf.y_test)

    def get_svm_score(self):
        self.svc.fit(self.model_svm.x_train, self.model_svm.y_train)
        return self.svc.score(self.model_svm.x_test, self.model_svm.y_test)

    def get_svm_cv_score(self, cv=5):
        return self.model_svm.cross_validation(self.svc, cv=cv)

    def get_rf_cv_score(self, cv=5):
        return self.model_rf.cross_validation(self.rfc, cv=cv)

    def feature_select_rf(self, min_features=5):
        self.selector_rf = (RFECV(self.rfc, step=1, cv=5, min_features_to_select=min_features))
        self.selector_rf.fit(self.model_rf.x_train, self.model_rf.y_train)
        return self.selector_rf.n_features_, self.selector_rf.grid_scores_

    def feature_select_svm(self, min_features=5):
        # self.model_svm.split_data()
        self.selector_svm = RFECV(self.svc, step=1, cv=5, min_features_to_select=min_features)
        self.selector_svm.fit(self.model_svm.x_train, self.model_svm.y_train)
        return self.selector_svm.n_features_, self.selector_svm.grid_scores_

    def set_x_selected_rf(self, data):
        self.x_selected_rf = self.selector_rf.transform(data)

    def set_x_selected_svm(self, data):
        self.x_selected_svm = self.selector_svm.transform(data)

    def save_model(self, model, path=''):
        if path == '':
            path = './unnamed'
        joblib.dump(model, path+'.pkl')

    def load_model(self, path=''):
        if path == '':
            return 0
        return joblib.load(path)


if __name__ == '__main__':
    fs = FeatureSelect()
    # print(fs.get_rf_cv_score(10))
    # print(fs.get_svm_cv_score(5))
    # print(fs.get_svm_score())

    '''# 生成训练svm的数据集
    fs.dh.read_data('data/phone_data.csv')
    fs.dh.handle_missing_data()
    fs.dh.generate_target_sample()
    fs.dh.data_encoding(types_dict)
    fs.dh.data_dimensionless(dimension_dict)
    fs.dh.save_data('new_svm_data', 'data/')'''

    '''# 保存svm模型
    fs.set_svm_dataset('data/new_svm_data.csv')
    fs.set_rf_dataset('data/rf_data.csv')
    fs.feature_select_svm()
    new_clf = fs.selector_svm.estimator_
    fs.save_model(new_clf, 'data/new_svm')

    #保存rf模型
    fs.feature_select_rf()
    new_rf = fs.selector_rf.estimator_
    fs.save_model(new_rf, 'data/new_rf')'''

    '''# 加载模型，并比较
    fs.svc = fs.load_model('data/new_svm.pkl')
    fs.rfc = fs.load_model('data/new_rf.pkl')
    # fs.set_svm_dataset('data/new_svm_data.csv')
    # fs.set_rf_dataset('data/rf_data.csv')
    fs.feature_select_rf()
    fs.feature_select_svm()
    # fs.feature_select_svm()
    # fs.feature_select_rf()
    # fs.set_x_selected_rf()
    # fs.set_x_selected_rf()
    score_svm = []
    score_rf = []
    for i in range(0, 30):
        fs.model_rf.split_data()
        fs.model_svm.split_data()
        fs.set_x_selected_rf()
        fs.set_x_selected_svm()
        score_svm.append(fs.svc.score(fs.x_selected_svm, fs.model_svm.y_test))
        score_rf.append(fs.rfc.score(fs.x_selected_rf, fs.model_rf.y_test))
    fs.model_rf.draw_compare_curve(len(score_rf), score_rf, 'RandomForest',
                                   score_svm, 'SVM', '', 'score', y_range=[0, 1])'''

    '''# 测试特征选择后svm模型得分
    fs.feature_select_rf()
    fs.set_x_selected_rf()
    new_rf = fs.selector_rf.estimator_
    print(new_rf.score(fs.x_selected_rf, fs.model_rf.y_test))'''

    # 递归特征选择结果
    num_svm, score_svm = fs.feature_select_svm()
    num_rf, score_rf = fs.feature_select_rf()
    fs.model_svm.draw_compare_curve(len(score_svm), score_rf, 'SMOTE_RF', score_svm,
                                    'SMOTE_SVM', 'Features number', 'score', y_range=[0, 1])
    print("随机森林的选择特征数量为:{}".format(num_rf))
    print("支持向量机选择特征数量为:{}".format(num_svm))

    '''# 用于比较SVM新旧模型的测试曲线
    score_svm = []
    score_rf = []
    fs.rfc.fit(fs.model_rf.x_train, fs.model_rf.y_train)
    fs.svc.fit(fs.model_svm.x_train, fs.model_svm.y_train)
    for i in range(30):
        fs.model_svm.split_data()
        fs.model_rf.split_data()
        # old_svc = SVC().fit(fs.model_svm.x_train, fs.model_svm.y_train)
        score_svm.append(fs.svc.score(fs.model_svm.x_test, fs.model_svm.y_test))
        score_rf.append(fs.rfc.score(fs.model_rf.x_test, fs.model_rf.y_test))
    fs.model_svm.draw_compare_curve(len(score_svm), score_rf, 'SMOTE_RF', score_svm,
                                    'SMOTE_SVM', '', 'score', y_range=[0, 1])'''

    '''# 用于比较RF新旧模型的测试曲线
    score_old = []
    score_new = []
    old_rf = RandomForestClassifier(random_state=30)    
    for i in range(30):
        fs.model_rf.split_data()
        old_rf.fit(fs.model_rf.x_train, fs.model_rf.y_train)
        score_new.append(fs.get_rf_score())
        score_old.append(old_rf.score(fs.model_rf.x_test, fs.model_rf.y_test))
    fs.model_svm.draw_compare_curve(len(score_new), score_old, 'Old_RF', score_new,
                                    'New_RF', '', 'score', y_range=[0, 1])'''


    # RF模型调参
    # fs.model_rf.draw_validation_curve(fs.rfc, 'n_estimators', range(1, 500, 20))
    '''param_grid = {
                  'n_estimators': [15]
                   ,'max_depth': [11]
                    ,'min_samples_leaf': [1]
                    , 'min_samples_split': range(1, 20)
                   # ,'max_features': [6]
                    ,'criterion': ['gini', 'entropy']
                  }
    result = fs.model_rf.grid_search_cv(param_grid, fs.rfc, 5)
    print("网格搜索最优参数如下")
    for param in result[0]:
        print(param+':{}'.format(result[0][param]))
    print("其得分为:{}".format(result[1]))'''

    #  svm模型调参
    # fs.model_svm.split_data()
    '''param_grid = {'intercept_scaling': np.logspace(-4, 2, 100)
                  , 'penalty': ['l2', 'l1']
                  , 'loss': ['squared_hinge']}'''
    '''param = {'C': [0.007564633275546291],
             'tol': [0.0001]}
    # fs.model_svm.draw_validation_curve(fs.svc, 'tol', np.logspace(-4, 1, 100))
    result = fs.model_svm.grid_search_cv(param, fs.svc, cv=5)
    print("网格搜索最优参数如下")
    for pa in result[0]:
        print(pa+':{}'.format(result[0][pa]))
    print("其得分为:{}".format(result[1]))'''



