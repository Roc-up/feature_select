from feature_select import FeatureSelect
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QSizePolicy
import numpy as np
import math
from time import time


class MyMplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        plt.rcParams['font.family'] = ['SimHei']
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyMplCanvas, self).__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, title, importances, feature_name, show_n_features=10):
        self.axes.set_title(title)
        show_n_features = math.floor(show_n_features)
        indices = np.argsort(importances)[::-1]
        indices = indices[:show_n_features]
        names = [feature_name[i] for i in indices]
        self.axes.bar(range(len(names)), importances[indices], color='lightblue', align='center')
        self.axes.set_xticks(range(len(names)))
        self.axes.set_xticklabels(names, rotation=90)
        self.axes.set_xlim([-1, len(names)])
        self.fig.tight_layout()
        self.draw()

