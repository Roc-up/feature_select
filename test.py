import numpy as np
from feature_select import FeatureSelect
import pandas as pd
import joblib
from load_data import DataHandle

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
    dh = DataHandle('data/phone_data.csv')
    dh.handle_missing_data()
    dh.generate_target_sample()
    dh.handle_timetype(['上市时间'])
    data = dh.get_data().loc[0, :]
    data = data.to_frame()
    data = data.stack()
    data = data.unstack(0)
    # data = dh.transform_rf_data(data, types_dict, series_dict)
    # data = dh.transform_svm_data(data, types_dict, dimension_dict)'''
    print(data.columns)
    '''key_list = types_dict.keys()
    ordinal_list = []
    onehot_list = []
    for key in key_list:
        if types_dict[key] == 0:
            ordinal_list.append(key)
        elif types_dict[key] == 1:
            onehot_list.append(key)
    ordinal = joblib.load('datahandle/ordinal.pkl')
    dh.data.loc[:, ordinal_list] = ordinal.transform(dh.data.loc[:, ordinal_list])
    onehot = joblib.load('datahandle/onehot.pkl')
    result = onehot.transform(dh.data.loc[:, onehot_list]).toarray()
    result = pd.DataFrame(result)
    columns = []
    for l in onehot.categories_:
        columns = columns + list(l)
    result.columns = columns
    for i in range(len(onehot_list)):
        key = list(onehot.categories_[i])
        temp = result.loc[:, key]
        pos = dh.data.columns.get_loc(onehot_list[i])
        data1 = dh.data.iloc[:, 0:pos]
        data2 = dh.data.iloc[:, pos + 1:]
        data1 = pd.concat([data1, temp], axis=1)
        dh.data = pd.concat([data1, data2], axis=1)
    dh.data = dh.x_same_encoding(dh.data, types_dict)
    print(dh.data)
    key_series = series_dict.keys()
    i=0
    for key in key_series:
        series = joblib.load('datahandle/series'+str(i)+'.pkl')
        dh.data.loc[:, key] = series.transform(dh.data.loc[:, key].values.reshape(-1, 1))
    print(dh.data)'''

