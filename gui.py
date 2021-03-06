# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        MainWindow.setMinimumSize(QtCore.QSize(1080, 720))
        MainWindow.setMaximumSize(QtCore.QSize(1080, 720))
        MainWindow.setMouseTracking(False)
        MainWindow.setWindowFilePath("")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 6, 1061, 701))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.logo = QtWidgets.QLabel(self.layoutWidget)
        self.logo.setMinimumSize(QtCore.QSize(170, 60))
        self.logo.setText("")
        self.logo.setObjectName("logo")
        self.horizontalLayout_5.addWidget(self.logo)
        self.name = QtWidgets.QLabel(self.layoutWidget)
        self.name.setMinimumSize(QtCore.QSize(250, 40))
        self.name.setMaximumSize(QtCore.QSize(250, 40))
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setWeight(75)
        self.name.setFont(font)
        self.name.setTextFormat(QtCore.Qt.AutoText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.name.setObjectName("name")
        self.horizontalLayout_5.addWidget(self.name)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.tabWidget = QtWidgets.QTabWidget(self.layoutWidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(1000, 600))
        self.tabWidget.setMaximumSize(QtCore.QSize(1050, 600))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.predict_tab = QtWidgets.QWidget()
        self.predict_tab.setObjectName("predict_tab")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.predict_tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1011, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.select_model_cmb1 = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_model_cmb1.setFont(font)
        self.select_model_cmb1.setObjectName("select_model_cmb1")
        self.select_model_cmb1.addItem("")
        self.select_model_cmb1.addItem("")
        self.horizontalLayout.addWidget(self.select_model_cmb1)
        spacerItem2 = QtWidgets.QSpacerItem(500, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.load_data_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.load_data_btn.setObjectName("load_data_btn")
        self.horizontalLayout.addWidget(self.load_data_btn)
        self.clear_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout.addWidget(self.clear_btn)
        self.delete_row_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.delete_row_btn.setObjectName("delete_row_btn")
        self.horizontalLayout.addWidget(self.delete_row_btn)
        self.add_row_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.add_row_btn.setObjectName("add_row_btn")
        self.horizontalLayout.addWidget(self.add_row_btn)
        self.submit_predict_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.submit_predict_btn.setObjectName("submit_predict_btn")
        self.horizontalLayout.addWidget(self.submit_predict_btn)
        self.tableWidget = QtWidgets.QTableWidget(self.predict_tab)
        self.tableWidget.setGeometry(QtCore.QRect(10, 80, 1011, 381))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(27)
        self.tableWidget.setRowCount(8)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(16, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(17, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(18, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(19, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(20, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(21, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(22, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(23, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(24, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(25, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(26, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.predict_tab)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 480, 1011, 52))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_4.setMinimumSize(QtCore.QSize(100, 40))
        self.label_4.setMaximumSize(QtCore.QSize(100, 40))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.result_edit = QtWidgets.QTextEdit(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.result_edit.sizePolicy().hasHeightForWidth())
        self.result_edit.setSizePolicy(sizePolicy)
        self.result_edit.setMinimumSize(QtCore.QSize(800, 35))
        self.result_edit.setMaximumSize(QtCore.QSize(900, 35))
        self.result_edit.setObjectName("result_edit")
        self.horizontalLayout_2.addWidget(self.result_edit)
        self.tabWidget.addTab(self.predict_tab, "")
        self.ranking_tab = QtWidgets.QWidget()
        self.ranking_tab.setObjectName("ranking_tab")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.ranking_tab)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(20, 10, 1011, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.select_model_cmb2 = QtWidgets.QComboBox(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_model_cmb2.setFont(font)
        self.select_model_cmb2.setObjectName("select_model_cmb2")
        self.select_model_cmb2.addItem("")
        self.select_model_cmb2.addItem("")
        self.horizontalLayout_3.addWidget(self.select_model_cmb2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.feature_num_spinBox = QtWidgets.QSpinBox(self.horizontalLayoutWidget_3)
        self.feature_num_spinBox.setMinimum(1)
        self.feature_num_spinBox.setMaximum(37)
        self.feature_num_spinBox.setObjectName("feature_num_spinBox")
        self.horizontalLayout_3.addWidget(self.feature_num_spinBox)
        spacerItem4 = QtWidgets.QSpacerItem(500, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.view_rank_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.view_rank_btn.setObjectName("view_rank_btn")
        self.horizontalLayout_3.addWidget(self.view_rank_btn)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 2)
        self.horizontalLayout_3.setStretch(5, 10)
        self.horizontalLayout_3.setStretch(6, 2)
        self.layoutWidget1 = QtWidgets.QWidget(self.ranking_tab)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 60, 1011, 501))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_4.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(self.layoutWidget1)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_4.addWidget(self.groupBox)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(10, 5, 10, 5)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        self.pet = QtWidgets.QLabel(self.layoutWidget1)
        self.pet.setEnabled(True)
        self.pet.setText("")
        self.pet.setObjectName("pet")
        self.verticalLayout.addWidget(self.pet)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem7)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 2)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.horizontalLayout_4.setStretch(0, 8)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 3)
        self.horizontalLayout_4.setStretch(3, 1)
        self.tabWidget.addTab(self.ranking_tab, "")
        self.verticalLayout_2.addWidget(self.tabWidget)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 8)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RocSystem"))
        self.name.setText(_translate("MainWindow", "商品分析系统"))
        self.label_2.setText(_translate("MainWindow", "选择预测模型："))
        self.select_model_cmb1.setItemText(0, _translate("MainWindow", "RandomForest"))
        self.select_model_cmb1.setItemText(1, _translate("MainWindow", "SVM"))
        self.load_data_btn.setText(_translate("MainWindow", "导入数据"))
        self.clear_btn.setText(_translate("MainWindow", "清空数据"))
        self.delete_row_btn.setText(_translate("MainWindow", "删除一行"))
        self.add_row_btn.setText(_translate("MainWindow", "新增一行"))
        self.submit_predict_btn.setText(_translate("MainWindow", "提交预测"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "5"))
        item = self.tableWidget.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "6"))
        item = self.tableWidget.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "7"))
        item = self.tableWidget.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "8"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "手机型号"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "上市时间"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "最低价格"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "最高价格"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "最小RAM"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "最大RAM"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "最小ROM"))
        item = self.tableWidget.horizontalHeaderItem(7)
        item.setText(_translate("MainWindow", "最大ROM"))
        item = self.tableWidget.horizontalHeaderItem(8)
        item.setText(_translate("MainWindow", "重量"))
        item = self.tableWidget.horizontalHeaderItem(9)
        item.setText(_translate("MainWindow", "屏幕尺寸"))
        item = self.tableWidget.horizontalHeaderItem(10)
        item.setText(_translate("MainWindow", "屏幕类型"))
        item = self.tableWidget.horizontalHeaderItem(11)
        item.setText(_translate("MainWindow", "分辨率"))
        item = self.tableWidget.horizontalHeaderItem(12)
        item.setText(_translate("MainWindow", "芯片型号"))
        item = self.tableWidget.horizontalHeaderItem(13)
        item.setText(_translate("MainWindow", "CPU得分"))
        item = self.tableWidget.horizontalHeaderItem(14)
        item.setText(_translate("MainWindow", "CPU核数"))
        item = self.tableWidget.horizontalHeaderItem(15)
        item.setText(_translate("MainWindow", "CPU主频率"))
        item = self.tableWidget.horizontalHeaderItem(16)
        item.setText(_translate("MainWindow", "GPU"))
        item = self.tableWidget.horizontalHeaderItem(17)
        item.setText(_translate("MainWindow", "操作系统"))
        item = self.tableWidget.horizontalHeaderItem(18)
        item.setText(_translate("MainWindow", "摄像头数"))
        item = self.tableWidget.horizontalHeaderItem(19)
        item.setText(_translate("MainWindow", "后置主像素"))
        item = self.tableWidget.horizontalHeaderItem(20)
        item.setText(_translate("MainWindow", "前置主像素"))
        item = self.tableWidget.horizontalHeaderItem(21)
        item.setText(_translate("MainWindow", "电池容量"))
        item = self.tableWidget.horizontalHeaderItem(22)
        item.setText(_translate("MainWindow", "充电接口"))
        item = self.tableWidget.horizontalHeaderItem(23)
        item.setText(_translate("MainWindow", "充电功率"))
        item = self.tableWidget.horizontalHeaderItem(24)
        item.setText(_translate("MainWindow", "网络类型"))
        item = self.tableWidget.horizontalHeaderItem(25)
        item.setText(_translate("MainWindow", "5G"))
        item = self.tableWidget.horizontalHeaderItem(26)
        item.setText(_translate("MainWindow", "无线充电"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.label_4.setText(_translate("MainWindow", "预测结果："))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.predict_tab), _translate("MainWindow", "商品畅销预测"))
        self.label_3.setText(_translate("MainWindow", "选择评估模型："))
        self.select_model_cmb2.setItemText(0, _translate("MainWindow", "RandomForest"))
        self.select_model_cmb2.setItemText(1, _translate("MainWindow", "SVM"))
        self.label.setText(_translate("MainWindow", "选择特征数量："))
        self.view_rank_btn.setText(_translate("MainWindow", "查看排行"))
        self.groupBox.setTitle(_translate("MainWindow", "查询结果"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ranking_tab), _translate("MainWindow", "查看属性排名"))
