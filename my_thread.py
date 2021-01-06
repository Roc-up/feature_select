from PyQt5 import QtCore
from feature_select import FeatureSelect
import feature_select as feat
import numpy as np
import pandas as pd
import time


class PredictThread(QtCore.QThread):
    predict_result = QtCore.pyqtSignal(str)

    def __init__(self):
        super(PredictThread, self).__init__()
        self.fs = FeatureSelect()
        self.sign = 0
        self.data = None

    def set_sign(self, i, data):
        self.sign = i
        self.data = data

    def run(self):
        if self.sign == 0:
            self.fs.svc = self.fs.load_model('data/smote_new_svm.pkl')
            temp = self.fs.dh.set_datetime(self.data)
            temp = self.fs.dh.transform_svm_data(temp, feat.types_dict, feat.dimension_dict)
            self.fs.feature_select_svm()
            self.fs.set_x_selected_svm(temp)
            result = self.fs.svc.predict(self.fs.x_selected_svm)
            test = pd.DataFrame(result, columns=['预测结果'])
            test = pd.concat([self.data, test], axis=1)
            test.to_csv('predict/result.csv', sep=',', header=True, index=False, encoding='utf-8')
            if len(result) == 1:
                if result[0] == 0:
                    self.predict_result.emit('你选择的分类器为SVM，其预测结果为不畅销。')
                else:
                    self.predict_result.emit('你选择的分类器为SVM，其预测结果为畅销。')
            elif len(result) > 1:
                self.predict_result.emit('你选择同时预测多个商品，预测结果请查看predict/result.csv文件。')
            else:
                self.predict_result.emit('输入测试数据为空。')

        elif self.sign == 1:
            self.fs.rfc = self.fs.load_model('data/smote_new_rf.pkl')
            temp = self.fs.dh.set_datetime(self.data)
            temp = self.fs.dh.transform_rf_data(temp, feat.types_dict, feat.series_dict)
            self.fs.feature_select_rf()
            self.fs.set_x_selected_rf(temp)
            result = self.fs.rfc.predict(self.fs.x_selected_rf)
            test = pd.DataFrame(result, columns=['预测结果'])
            test = pd.concat([temp, test], axis=1)
            test.to_csv('predict/result.csv', sep=',', header=True, index=False, encoding='utf-8')
            if len(result) == 1:
                if result[0] == 0:
                    self.predict_result.emit('你选择的分类器为RandomForest，其预测结果为不畅销。')
                else:
                    self.predict_result.emit('你选择的分类器为RandomForest，其预测结果为畅销。')
            elif len(result) > 1:
                self.predict_result.emit('你选择同时预测多个商品，预测结果请查看predict/result.csv文件。')
            else:
                self.predict_result.emit('输入测试数据为空。')


class FeatureImportanceThread(QtCore.QThread):
    plot_importance = QtCore.pyqtSignal(str, np.ndarray, pd.Index)

    def __init__(self):
        super(FeatureImportanceThread, self).__init__()
        self.fs = FeatureSelect()
        self.sign = 0

    def set_sign(self, i):
        self.sign = i

    def run(self):
        if self.sign == 0:
            self.fs.svc = self.fs.load_model('data/smote_new_svm.pkl')
            self.fs.feature_select_svm()
            importance_svm = np.array(list(map(abs, self.fs.svc.coef_[0])))
            feature_name_svm = self.fs.dh.get_feature_names()
            feature_name_svm = feature_name_svm[self.fs.selector_svm.get_support()]
            self.plot_importance.emit('Feature Importance of SVM', importance_svm, feature_name_svm)
            # print(feature_name_svm)
            # self.fs.model_svm.draw_feature_importance(importance_svm, feature_name_svm,
                                                      # 'Feature Importance of SVM')
        elif self.sign == 1:
            self.fs.rfc = self.fs.load_model('data/smote_new_rf.pkl')
            self.fs.feature_select_rf()
            importance_rf = self.fs.rfc.feature_importances_
            feature_name_rf = self.fs.dh.get_feature_names()
            feature_name_rf = feature_name_rf[self.fs.selector_rf.get_support()]
            self.plot_importance.emit('Feature Importance of RF', importance_rf, feature_name_rf)
            # self.fs.model_svm.draw_feature_importance(importance_rf, feature_name_rf,
                                                      # 'Feature Importance of RF')
        else:
            print('error')


class BackendThread(QtCore.QThread):
    # 通过类成员对象定义信号
    update_date = QtCore.pyqtSignal(str, int)
    path = "./res/PetAction/HOOD-action_"
    # 处理业务逻辑

    def __init__(self):
        super(BackendThread, self).__init__()

    def run(self):
        while True:
            for i in range(0, 10):
                self.update_date.emit(self.path, int(i))
                time.sleep(0.5)
            self.path = self.swith_path(0)

    def swith_path(self, choice):
        return {
            0: "./res/PetAction/HOOD-action_",
        }.get(choice, "./res/PetAction/HOOD-action_")