import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class LoadingGifWin(QWidget):
    def __init__(self,parent=None):
        super(LoadingGifWin, self).__init__(parent)
        # 实例化标签到窗口中
        self.label=QLabel('', self)
        # self.label.setWindowOpacity(1)
        # self.label.setAttribute(Qt.WA_TranslucentBackground)
        self.mylayout = QVBoxLayout()
        self.mylayout.addWidget(self.label)
        # 设置标签的宽度与高度
        self.setFixedSize(500,500)
        # 设置无边框
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.movie=QMovie('./res/loading.gif')
        self.label.setMovie(self.movie)
        self.movie.start()


if __name__ == '__main__':
    app=QApplication(sys.argv)
    load=LoadingGifWin()
    load.show()
    sys.exit(app.exec_())