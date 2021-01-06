from feature_select import FeatureSelect
import joblib
from load_data import DataHandle
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from feature_select import FeatureSelect

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


if __name__ == '__main__':
    '''
     model = joblib.load('unnamed.pkl')
     dh = DataHandle('data/svm_data.csv')
     fs = FeatureSelect()
     fs.feature_select_svm()
     data = fs.selector_svm.transform(dh.get_data())
     print(model.score(data, dh.get_target()))'''

    '''# 生成预处理文件
    dh1 = DataHandle('data/phone_data.csv')
    dh1.handle_missing_data()
    dh1.generate_target_sample()
    dh1.data_encoding(types_dict)
    dh1.handle_timetype(['上市时间'])
    print("before:{}".format(dh1.data['上市时间'].dtypes))
    print("过采样前的数据大小：{}".format(dh1.data.shape))
    dh1.data_oversample()
    print("过采样后的数据大小：{}".format(dh1.data.shape))
    print("after:{}".format(dh1.data['上市时间'].dtypes))
    dh1.after_oversample()
    # dh1.series_discretization(series_dict)
    dh1.data_dimensionless(dimension_dict)
    dh1.save_data('smote_svm_data', 'data/')'''

    fs1 = FeatureSelect()
    fs2 = FeatureSelect()
    score_svm = []
    score_rf = []
    fs1.set_svm_dataset('data/new_svm_data.csv')
    old_svc = LinearSVC().fit(fs1.model_svm.x_train, fs1.model_svm.y_train)
    # old_svc = SVC(kernel='linear').fit(fs.model_svm.x_train, fs.model_svm.y_train)
    print("normal:{}".format(fs1.model_svm.x_train.shape))
    for i in range(30):
        fs1.model_svm.split_data()
        score_rf.append(old_svc.score(fs1.model_svm.x_test, fs1.model_svm.y_test))

    fs2.set_svm_dataset('data/smote_svm_data.csv')
    new_svc = LinearSVC().fit(fs2.model_svm.x_train, fs2.model_svm.y_train)
    # new_svc = SVC(kernel='linear').fit(fs.model_svm.x_train, fs.model_svm.y_train)
    print("smote:{}".format(fs2.model_svm.x_train.shape))
    for i in range(30):
        fs2.model_svm.split_data()
        score_svm.append(new_svc.score(fs2.model_svm.x_test, fs2.model_svm.y_test))
    fs2.model_svm.draw_compare_curve(len(score_svm), score_rf, 'Old_SVM', score_svm,
                                    'SMOTE_SVM', '', 'score', y_range=[0, 1])


