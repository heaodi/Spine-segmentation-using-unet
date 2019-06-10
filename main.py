from PyQt5 import QtWidgets
from mainwindow import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from process.data import *
import sys
import os



# # class mywindow(Ui_MainWindow):
# class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
#     def __init__(self):
#         super(mywindow, self).__init__()
#         self.setupUi(Ui_MainWindow)
#         # self.pushButton.clicked.connect(self.start_clink())
#
#     def start_clink(self):
#         os._exit(0)

# class mywindow(QtWidgets.QWidget, Ui_MainWindow):
#     def __init__(self):
#         super(mywindow, self).__init__()
#         self.setupUi(self)
#
#     def start_clink(self):
#         os._exit(0)

# class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
#     def __init__(self):
#         super(mywindow, self).__init__()
#         self.setupUi(self)
#
#     def startClink(self):
#         os._exit(0)

class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=Ui_MainWindow):
        super(mywindow, self).__init__(parent)
        self.setupUi(parent)

        self.pushButton.clicked.connect(self.startClink)


    def startClink(self, remark):
        valid_label = "./data/spine/valid/image/image_175_5.png"
        img_qt = QPixmap(valid_label).scaled(self.label.width(), self.label.height())

        self.label.setPixmap(img_qt)
        # os._exit(0)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = mywindow(MainWindow)  # 注意把类名修改为myDialog
    # ui.setupUi(MainWindow)  myDialog类的构造函数已经调用了这个函数，这行代码可以删去
    MainWindow.show()
    sys.exit(app.exec_())
