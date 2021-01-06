import feature_select
from feature_select import FeatureSelect
from time import time
import numpy as np

if __name__ == '__main__':
    fs = FeatureSelect()
    # 导入模型
    fs.rfc = fs.load_model('data/smote_new_rf.pkl')
    fs.svc = fs.load_model('data/smote_new_svm.pkl')

    # 生成选择器
    fs.feature_select_svm()
    fs.feature_select_rf()

    '''# 绘制两个模型准确度曲线和运行时间曲线
    score_svm = []
    score_rf = []
    time_svm = []
    time_rf = []
    for i in range(0, 30):
        fs.model_rf.split_data()
        fs.model_svm.split_data()
        fs.set_x_selected_rf()
        fs.set_x_selected_svm()
        time0 = time()
        score_svm.append(fs.svc.score(fs.x_selected_svm, fs.model_svm.y_test))
        time_gap = time()-time0
        time_svm.append(time_gap)
        time0 = time()
        score_rf.append(fs.rfc.score(fs.x_selected_rf, fs.model_rf.y_test))
        time_gap = time() - time0
        time_rf.append(time_gap)
    fs.model_rf.draw_compare_curve(len(score_rf), score_rf, 'RandomForest',
                                   score_svm, 'SVM', '', 'score', y_range=[0, 1])
    fs.model_rf.draw_compare_curve(len(score_rf), time_rf, 'RandomForest',
                                   time_svm, 'SVM', '', 'RunTime', y_range=[0, 0.1])'''
    # fs.set_x_selected_svm()

    importance_svm = np.array(list(map(abs, fs.svc.coef_[0])))
    feature_name_svm = fs.dh.get_feature_names()
    feature_name_svm = feature_name_svm[fs.selector_svm.get_support()]
    fs.model_svm.draw_feature_importance(importance_svm, feature_name_svm, 'Feature Importance of SVM')

    importance_rf = fs.rfc.feature_importances_
    feature_name_rf = fs.dh.get_feature_names()
    feature_name_rf = feature_name_rf[fs.selector_rf.get_support()]
    fs.model_svm.draw_feature_importance(importance_rf, feature_name_rf, 'Feature Importance of RF')




