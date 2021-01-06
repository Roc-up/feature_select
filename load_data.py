import pandas as pd
import numpy as np
import datetime
import calendar
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib

pd.set_option('display.max_columns', None)


class DataHandle:
    def __init__(self, filepath):
        self.filepath = filepath.split('.')[0]
        self.data = pd.read_csv(filepath, header=0)

    def get_feature_names(self):
        return self.data.columns[0:-1]

    def save_data(self, filename, path):
        self.data.to_csv(path+filename+'.csv', sep=',', header=1, index=0, encoding='utf-8')

    def read_data(self, path):
        self.data = pd.read_csv(path, header=0)

    # 默认数据最后一列为标签
    def get_target(self):
        return self.data.iloc[:, -1]

    def get_data(self):
        return self.data.drop(self.data.columns[-1], axis=1)

    def x_same_encoding(self, data, types_dict={}):
        key_list = list(types_dict.keys())
        ordinal_list = []
        onehot_list = []
        for key in key_list:
            if types_dict[key] == 0:
                ordinal_list.append(key)
            elif types_dict[key] == 1:
                onehot_list.append(key)
        ordinal = joblib.load('datahandle/ordinal.pkl')
        data.loc[:, ordinal_list] = ordinal.transform(data.loc[:, ordinal_list])
        onehot = joblib.load('datahandle/onehot.pkl')
        result = onehot.transform(data.loc[:, onehot_list]).toarray()
        result = pd.DataFrame(result)
        columns = []
        for l in onehot.categories_:
            columns = columns + list(l)
        result.columns = columns
        for i in range(len(onehot_list)):
            key = list(onehot.categories_[i])
            temp = result.loc[:, key]
            pos = data.columns.get_loc(onehot_list[i])
            data1 = data.iloc[:, 0:pos]
            data2 = data.iloc[:, pos + 1:]
            data1 = pd.concat([data1, temp], axis=1)
            data = pd.concat([data1, data2], axis=1)
        return data

    def set_datetime(self, data, columns_name=['上市时间']):
        # 设置日期格式
        data['上市时间'] = data['上市时间'].astype(np.str)
        for i in range(data['上市时间'].shape[0]):
            temp = data.loc[i, '上市时间'][0:4] + '-' + data.loc[i, '上市时间'][4:]
            data.loc[i, '上市时间'] = temp
        data['上市时间'] = pd.to_datetime(data['上市时间'])
        if len(columns_name):
            for col in columns_name:
                for i in range(data.shape[0]):
                    data.loc[i, col] = calendar.timegm(data.loc[i, col].timetuple())
                data[col] = data[col].astype(np.int64)
        else:
            return 0
        return data

    def transform_rf_data(self, data, types_dict={}, series_dict={}):
        data = self.x_same_encoding(data, types_dict)
        key_series = series_dict.keys()
        i = 0
        for key in key_series:
            series = joblib.load('datahandle/series' + str(i) + '.pkl')
            data.loc[:, key] = series.transform(data.loc[:, key].values.reshape(-1, 1))
        return data

    def transform_svm_data(self, data, types_dict={}, dimension_dict={}):
        data = self.x_same_encoding(data, types_dict)
        key_dimension = dimension_dict.keys()
        dimension = joblib.load('datahandle/standard.pkl')
        data.loc[:, key_dimension] = dimension.transform(data.loc[:, key_dimension])
        return data

    def data_oversample(self):
        str_col = ['分辨率', '芯片型号', '操作系统']
        samples_before = self.data.shape[0]
        first_col = self.data.loc[:, '手机型号']
        smoted_data = self.data.iloc[:, 1:-1]
        smoted_target = self.data.iloc[:, -1]
        smoted_data, smoted_target = SMOTE().fit_sample(smoted_data, smoted_target)
        samples_after = smoted_data.shape[0]
        temp = list(range(samples_before, samples_after))
        first_col2 = pd.Series(temp, index=range(samples_before, samples_after))
        first_col = pd.concat([first_col, first_col2], axis=0)
        first_col = pd.DataFrame({'手机型号': first_col})
        # first_col.reindex(columns=['手机型号'])
        step1 = pd.concat([smoted_data, smoted_target], axis=1)
        self.data = pd.concat([first_col, step1], axis=1)
        self.after_oversample()
        self.data.loc[:, str_col] = np.around(self.data.loc[:, str_col],)

    def after_oversample(self):
        screen = self.data.iloc[:, 10:18]
        gpuos = self.data.iloc[:, 23:25]
        usb = self.data.iloc[:, 29:32]
        fiveG = self.data.iloc[:, 34:36]
        self.data.iloc[:, 23:25] = np.around(gpuos)
        for i in range(150, self.data.shape[0]):
            scr = screen.iloc[i, :]
            scr_max = scr[scr == scr.max()].index
            scr_max = np.random.choice(scr_max)
            screen.iloc[i, :] = 0.0
            screen.loc[i, scr_max] = 1.0

            u = usb.iloc[i, :]
            u_max = u[u == u.max()].index
            u_max = np.random.choice(u_max)
            usb.iloc[i, :] = 0.0
            usb.loc[i, u_max] = 1.0

            fg = fiveG.iloc[i, :]
            fg_max = fg[fg == fg.max()].index
            fg_max = np.random.choice(fg_max)
            fiveG.iloc[i, :] = 0.0
            fiveG.loc[i, fg_max] = 1.0
        self.data.iloc[:, 10:18] = screen
        self.data.iloc[:, 29:32] = usb
        self.data.iloc[:, 34:36] = fiveG

    def handle_timetype(self, columns_name=[]):
        if len(columns_name):
            for col in columns_name:
                for i in range(self.data.shape[0]):
                    self.data.loc[i, col] = calendar.timegm(self.data.loc[i, col].timetuple())
                self.data[col] = self.data[col].astype(np.int64)
        else:
            return 0
        return 1

    def handle_missing_data(self):
        check_result = self.data.isnull().any()
        miss_num = self.data.isnull().sum()
        print("特征缺失情况如下：")
        print(miss_num[miss_num > 0])
        for i in range(0, self.data.shape[1]):
            if check_result[i]:
                median = self.data.iloc[:, i].mode()[0]
                self.data.iloc[:, i] = self.data.iloc[:, i].fillna(median)
                print("特征{}".format(self.data.columns[i])+"存在缺失值，已用众数{}填充".format(median))

    def new_data_encoding(self, types_dict={}):
        key_list = list(types_dict.keys())
        ordinal_list = []
        onehot_list = []
        ordinal = OrdinalEncoder()
        onehot = OneHotEncoder()
        result = []
        for key in key_list:
            if types_dict[key] == 0:
                ordinal_list.append(key)
            elif types_dict[key] == 1:
                onehot_list.append(key)
        print(ordinal_list)
        print(onehot_list)
        temp_o = self.data.loc[:, ordinal_list]
        if len(ordinal_list) == 1:
            ordinal.fit(temp_o.values.reshape(-1, 1))
            self.data.loc[:, ordinal_list] = ordinal.transform(temp_o.values.reshape(-1, 1))
        elif len(ordinal_list):
            ordinal.fit(temp_o)
            self.data.loc[:, ordinal_list] = ordinal.transform(temp_o)
        joblib.dump(ordinal, 'datahandle/ordinal.pkl')

        temp_hot = self.data.loc[:, onehot_list]
        if len(onehot_list) == 1:
            onehot.fit(temp_hot.values.reshape(-1, 1))
            self.data.loc[:, onehot_list] = onehot.transform(temp_hot.values.reshape(-1, 1))
        elif len(onehot_list):
            onehot.fit(temp_hot)
            result = onehot.transform(temp_hot).toarray()
            result = pd.DataFrame(result)
        columns = []
        joblib.dump(onehot, 'datahandle/onehot.pkl')
        for l in onehot.categories_:
            columns = columns + list(l)
        result.columns = columns
        for i in range(len(onehot_list)):
            key = list(onehot.categories_[i])
            temp = result.loc[:, key]
            pos = self.data.columns.get_loc(onehot_list[i])
            data1 = self.data.iloc[:, 0:pos]
            data2 = self.data.iloc[:, pos+1:]
            data1 = pd.concat([data1, temp], axis=1)
            self.data = pd.concat([data1, data2], axis=1)
        print(self.data)

            # types_dict为所有特征类型编号，
    def data_encoding(self, types_dict={}):
        key_list = list(types_dict.keys())

        for key in key_list:
            temp = self.data.loc[:, key]
            if types_dict[key] == 0:  # 0代表常规编码
                self.data.loc[:, key] = OrdinalEncoder().fit_transform(temp.values.reshape(-1, 1))
            elif types_dict[key] == 1:  # 1代表独热编码
                enc = OneHotEncoder(categories='auto').fit(temp.values.reshape(-1, 1))
                result = enc.transform(temp.values.reshape(-1, 1)).toarray()
                pos = self.data.columns.get_loc(key)
                data1 = self.data.iloc[:, 0:pos]
                data2 = self.data.iloc[:, pos:]
                data1 = pd.concat([data1, pd.DataFrame(result)], axis=1)
                self.data = pd.concat([data1, data2], axis=1)
                name = enc.get_feature_names()
                ret = list(self.data.columns)
                for i in range(0, len(name)):
                    ret[pos + i] = name[i]
                self.data.columns = ret
                self.data = self.data.drop(columns=key, axis=1)

            elif types_dict[key] == 2:  # 2代表标签编码
                self.data.loc[:, key] = LabelEncoder().fit_transform(temp)
            else:
                return 0
            # self.data.to_csv(self.filepath+'_encoded.csv', sep=',', header=True, encoding='utf-8')
        return 1

    # 连续值离散化 series_dict为需要离散化的特征及参数
    def series_discretization(self, series_dict={}):
        key_list = list(series_dict.keys())
        i = 0
        for key in key_list:
            types_list = series_dict[key]
            if types_list[0] == 0:
                discretizer = KBinsDiscretizer(
                    n_bins=types_list[1],
                    encode=types_list[2],
                    strategy=types_list[3]).fit(self.data.loc[:, key].values.reshape(-1, 1))
                self.data.loc[:, key] = discretizer.transform(self.data.loc[:, key].values.reshape(-1, 1))
                filename = "datahandle/series"+str(i)+".pkl"
                joblib.dump(discretizer, filename)
                i = i + 1
            else:
                return 0
        # self.data.to_csv(self.filepath+'_discretized.csv', sep=',', header=True, encoding='utf-8')
        return 1

    def data_dimensionless(self, dimension_dict={}):
        key_list = list(dimension_dict.keys())
        col_list_std = []
        col_list_minmax = []
        for key in key_list:
            temp = self.data.loc[:, key]
            if dimension_dict[key] == 0:  # 数据标准化
                if key != self.data.columns[-1] and (temp.dtypes == 'int64' or temp.dtypes == 'float64'):
                    col_list_std.append(key)
            elif dimension_dict[key] == 1:  # 数据归一化
                if key != self.data.columns[-1] and (temp.dtypes == 'int64' or temp.dtypes == 'float64'):
                    col_list_minmax.append(key)
                return 0

        if len(col_list_minmax) == 1:
            minmax_scaler = MinMaxScaler().fit(self.data.loc[:, col_list_std].values.reshape(-1, 1))
            self.data.loc[:, col_list_std] = minmax_scaler.transform(
                self.data.loc[:, col_list_std].values.reshape(-1, 1))
        elif len(col_list_minmax):
            minmax_scaler = MinMaxScaler().fit(self.data.loc[:, col_list_std])
            self.data.loc[:, col_list_std] = minmax_scaler.transform(self.data.loc[:, col_list_std])

        if len(col_list_std) == 1:
            std_scaler = StandardScaler().fit(self.data.loc[:, col_list_std].values.reshape(-1, 1))
            self.data.loc[:, col_list_std] = std_scaler.transform(
                self.data.loc[:, col_list_std].values.reshape(-1, 1))
        elif len(col_list_std):
            std_scaler = StandardScaler().fit(self.data.loc[:, col_list_std])
            self.data.loc[:, col_list_std] = std_scaler.transform(self.data.loc[:, col_list_std])
        joblib.dump(std_scaler, 'datahandle/standard.pkl')

        # self.data.to_csv(self.filepath+'_dimensionless.csv', sep=',', header=True, encoding='utf-8')
        return 1

    def data_dimensionless_all(self, sign=True):
        types_list = self.data.dtypes
        columns_list = []
        for i in range(0, self.data.shape[1]):
            temp = self.data.iloc[:, i]
            if self.data.columns[i] != self.data.columns[-1] and \
                    (temp.dtypes == 'int64' or temp.dtypes == 'float64'):
                columns_list.append(self.data.columns[i])
        if len(columns_list) == 1:
            if sign:
                self.data.loc[:, columns_list] = StandardScaler().fit_transform(
                    self.data.loc[:, columns_list].values.reshape(-1, 1))
            else:
                self.data.loc[:, columns_list] = MinMaxScaler().fit_transform(
                    self.data.loc[:, columns_list].values.reshape(-1, 1))

        elif len(columns_list):
            if sign:
                self.data.loc[:, columns_list] = StandardScaler().fit_transform(
                    self.data.loc[:, columns_list])
            else:
                self.data.loc[:, columns_list] = MinMaxScaler().fit_transform(
                    self.data.loc[:, columns_list])
        # self.data.to_csv(self.filepath+'_dimensionless_all.csv', sep=',', header=True, encoding='utf-8')
        return 1

    # 为手机数据生成标签
    def generate_target_sample(self):
        # 设置日期格式
        self.data['上市时间'] = self.data['上市时间'].astype(np.str)
        for i in range(self.data['上市时间'].shape[0]):
            temp = self.data.loc[i, '上市时间'][0:4] + '-' + self.data.loc[i, '上市时间'][4:]
            self.data.loc[i, '上市时间'] = temp
        self.data['上市时间'] = pd.to_datetime(self.data['上市时间'])

        # 提取上市时间和评价数两个标签的数据
        part_data = self.data.loc[:, ['上市时间', '评价数']]
        # print(part_data)

        # 添加标签为当前时间的数据
        part_data = part_data.reindex(columns=list(part_data.columns) + ['当前时间'])
        part_data.loc[:, '当前时间'] = '{0:%Y-%m-%d}'.format(datetime.datetime.now())
        part_data.loc[:, '当前时间'] = pd.to_datetime(part_data['当前时间'])
        # print(part_data)

        # 计算上市时间与当前时间的月份差
        for i in range(0, part_data.shape[0]):
            x = part_data.loc[i, '当前时间']
            y = part_data.loc[i, '上市时间']
            part_data.loc[i, '月数'] = (x.year - y.year) * 12 + (x.month - y.month)
        # print(part_data)

        # 计算出每月的评价数
        for i in range(0, part_data.shape[0]):
            evaluation_num = part_data.loc[i, '评价数']
            month = part_data.loc[i, '月数']
            if month == 0:
                month = 0.03
            part_data.loc[i, '评价数/月'] = evaluation_num / month
        # print(part_data)

        # 规定超出平均每月评价数视为畅销, target记录为1
        mean_num = part_data['评价数/月'].mean()
        standard = mean_num * 0
        # print("平均值:{}".format(mean_num))
        # test = part_data.loc[:, ['评价数', '月数', '评价数/月']]
        # test.to_csv('data/test.csv', sep=',', header=True, encoding='utf-8')
        for i in range(0, part_data.shape[0]):
            x_per_month = part_data.loc[i, '评价数/月']
            if x_per_month > mean_num - standard:
                part_data.loc[i, 'target'] = 1
            else:
                part_data.loc[i, 'target'] = 0
        # print(part_data)
        print('标签个数:{}'.format(part_data.loc[:, 'target'].value_counts()))

        # 将target标签加入data
        part_data['target'] = part_data['target'].astype(np.int64)
        self.data = pd.concat([self.data, part_data['target']], axis=1)
        self.data = self.data.drop('评价数', axis=1)


if __name__ == '__main__':
    dh = DataHandle("data/phone_data.csv")
    # dh.handle_missing_data()
    # dh.data_encoding()
    # dh.series_discretization({'上市时间': [0, 5, 'ordinal', 'quantile']})





