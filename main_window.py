from gui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QTableWidgetItem, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QDate, QTime
from matlibplot_widget import MyMplCanvas
from my_thread import FeatureImportanceThread
from my_thread import BackendThread
from my_thread import PredictThread
import pandas as pd
import numpy as np
from PyQt5 import sip
from my_son import LoadingGifWin

col_name = ['手机型号', '上市时间', '最低价格', '最高价格', '最小RAM', '最大RAM', '最小ROM', '最大ROM', '重量',
            '屏幕尺寸', '屏幕类型', '分辨率', '芯片型号', 'CPU得分', 'CPU核数', 'CPU主频率', 'GPU', '操作系统',
            '摄像头数', '后置主像素', '前置主像素', '电池容量', '充电接口', '充电功率', '网络类型', '5G', '无线充电']


class MainWindow(QMainWindow, Ui_MainWindow):
    open_path = pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.waiting = LoadingGifWin()

        self.logo.setPixmap(QPixmap('res/logo.png').scaled(self.logo.width(),
                                                           self.logo.height()))
        self.setWindowIcon(QIcon('res/icon.png'))  # 设置图标

        self.view_rank_btn.clicked.connect(self.view_ranking)  # 为查看排行按钮增加事件监听
        self.view_rank_btn.clicked.connect(self.show_waiting)
        self.submit_predict_btn.clicked.connect(self.submit_predicting)  # 为预测按钮增加事件监听
        self.submit_predict_btn.clicked.connect(self.show_waiting)
        self.load_data_btn.clicked.connect(self.load_data)
        self.delete_row_btn.clicked.connect(self.delete_row)
        self.add_row_btn.clicked.connect(self.add_row)
        self.clear_btn.clicked.connect(self.clear_data)
        self.open_path.connect(self.update_table)
        self.select_model_cmb2.currentIndexChanged.connect(self.clear_widget)

        '''self.backend = BackendThread()
        self.backend.update_date.connect(self.show_mypet)  # 连接信号
        self.backend.start()'''
        pixmap = QPixmap("res/data_analysis.png")  # 按指定路径找到图片
        self.pet.setPixmap(pixmap)  # 在label上显示图片
        self.pet.setScaledContents(True)  # 让图片自适应label大小

        self.importance = FeatureImportanceThread()
        self.importance.plot_importance.connect(self.show_importance)
        self.importance.plot_importance.connect(self.close_waiting)
        self.canvas = None
        # self.canvas = MyMplCanvas()
        self.mylayout = QGridLayout(self.groupBox)
        # self.mylayout.addWidget(self.canvas, 0, 1)

        self.predictThread = PredictThread()
        self.predictThread.predict_result.connect(self.show_predict_result)
        self.predictThread.predict_result.connect(self.close_waiting)
        # self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)

    def clear_widget(self):
        if type(self.canvas) != type(None):
            self.mylayout.removeWidget(self.canvas)
            sip.delete(self.canvas)
            self.canvas = None

    def close_waiting(self):
        self.waiting.close()

    def show_waiting(self):
        self.waiting.show()

    def clear_data(self):
        self.tableWidget.clearContents()
        self.result_edit.setText('')

    def delete_row(self):
        row = self.tableWidget.currentRow()
        if row:
            self.tableWidget.removeRow(row)
        else:
            print('请选择需要删除的行')

    def add_row(self):
        row_num = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(row_num+1)

    def load_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.csv )')[0]
        print(openfile_name)
        self.open_path.emit(openfile_name)

    def update_table(self, path):
        print(path)
        if len(path):
            input_table = pd.read_csv(path)
            input_table_rows = input_table.shape[0]
            input_table_columns = input_table.shape[1]
            input_table_header = input_table.columns.values.tolist()

            self.tableWidget.setColumnCount(input_table_columns)
            self.tableWidget.setRowCount(input_table_rows)
            self.tableWidget.setHorizontalHeaderLabels(input_table_header)

            for i in range(input_table_rows):
                rows_values = input_table.iloc[[i]]
                rows_values_array = np.array(rows_values)
                rows_values_list = rows_values_array.tolist()[0]
                for j in range(input_table_columns):
                    item_list = rows_values_list[j]
                    input_table_item = str(item_list)
                    newItem = QTableWidgetItem(input_table_item)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.tableWidget.setItem(i, j, newItem)
        else:
            self.centralwidget.show()

    def get_table_data(self):
        col = self.tableWidget.columnCount()
        row_list = set()
        index = self.tableWidget.selectionModel().selection().indexes()
        for i in index:
            row_list.add(i.row())
        select_data = []
        for r in row_list:
            row_data = [self.tableWidget.item(r, p).text() for p in range(col)]
            select_data.append(row_data)
        select_data = pd.DataFrame(select_data, index=range(len(row_list)), columns=col_name)
        return select_data

    def show_predict_result(self, result):
        self.result_edit.setText(result)

    def submit_predicting(self):
        self.result_edit.setText('')
        data = self.get_table_data()
        print(data)
        model = self.select_model_cmb1.currentText()
        if model == 'SVM':
            print('SVM')
            self.predictThread.set_sign(0, data)
            self.predictThread.start()
        elif model == 'RandomForest':
            self.predictThread.set_sign(1, data)
            self.predictThread.start()

    def view_ranking(self):
        model = self.select_model_cmb2.currentText()
        if model == 'SVM':
            self.importance.set_sign(0)
            self.importance.start()
        elif model == 'RandomForest':
            self.importance.set_sign(1)
            self.importance.start()

    def show_mypet(self, path, i):
        pixmap = QPixmap("%s%s.png" % (path, str(i)))  # 按指定路径找到图片
        self.pet.setPixmap(pixmap)  # 在label上显示图片
        self.pet.setScaledContents(True)  # 让图片自适应label大小

    def show_importance(self, title, importance, featurename):
        if type(self.canvas) != type(None):
            self.mylayout.removeWidget(self.canvas)
            sip.delete(self.canvas)
        self.canvas = MyMplCanvas()
        feature_num = self.feature_num_spinBox.value()
        self.canvas.update_figure(title, importance, featurename, feature_num)
        self.mylayout.addWidget(self.canvas, 0, 1)

