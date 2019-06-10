from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtGui
from mainwindow import Ui_MainWindow
from PyQt5.QtGui import QPixmap, QImage
from process.data import *
from models.model import *
from scipy import misc
import nibabel as nib
import shutil
import sys
import os

model_save_path = './data/save_models/'
predict_path = "./data/spine/valid/image/"
test_path = "./data/spine/test_image/"
image_save_path = "./data/spine/result/test_save/"
predict_save_path = "./data/spine/result/predict_save/"
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=Ui_MainWindow):
        super(mywindow, self).__init__(parent)
        self.setupUi(parent)
        self.pushButton.clicked.connect(self.startClink)
        self.indexnum = 0
        self.queue = 0
        self.startclink = False
    def startClink(self):
        valid_label = predict_path + "image_175_5.png"
        # img_qt = QPixmap(valid_label).scaled(self.label.width(), self.label.height())
        self.filename = self.msg()
        print(self.filename)
        filetype = self.filename.split(".")[-1]
        print(filetype)
        if filetype == "gz":
            self.nii_gz2png()

        self.show_image()
        # self.nii_gz2png(filename)

        # self.label.setPixmap(img_qt)

        # os._exit(0)

    def msg(self):
        filename1, filetype = QFileDialog.getOpenFileName(self,
                                                         "选取文件",
                                                          test_path,
                                                          "All Files (*);;Gz Files (*.gz);;Image Files(*.png)")  # 设置文件扩展名过滤,注意用双分号间隔
        print(filename1, filetype)
        return filename1

    def nii_gz2png(self):
        self.img_src = nib.load(self.filename)
        self.indexnum = 0
        self.width, self.height, self.queue = self.img_src.dataobj.shape
        print(self.width, self.height, self.queue)
        img = self.img_src.get_data()
        if os.path.exists(image_save_path):
            shutil.rmtree(image_save_path)
            shutil.rmtree(predict_save_path)
            os.mkdir(image_save_path)
            os.mkdir(predict_save_path)
        else:
            os.mkdir(image_save_path)
            os.mkdir(predict_save_path)
        z = self.filename.split(".")[-3].split("/")[-1]
        print("z:", z)
        for j in range(0, self.queue):
             misc.imsave(image_save_path + z + '_' + str(j) + '.png', img[:, :, j])

    def show_image(self):
        train_data = glob.glob(image_save_path + "*.png")
        if train_data is not None:
            # self.img = self.img_src.get_data()
            if self.indexnum > self.queue:
                self.indexnum = self.queue
            elif self.indexnum < 0:
                self.indexnum = 0
            print(train_data[self.indexnum])
            self.img_show = QPixmap(train_data[self.indexnum]).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(self.img_show)


    def loadrpedictmodel(self):
        class_weight = [0.4, 0.6]
        def weighted_binary_crossentropy(y_true, y_pred):
            class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
            return K.sum(class_loglosses * K.constant(class_weight))
        self.model = load_model(model_save_path + "2019-05-27_03-37_98.33.h5", custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

    def predict_picture(self):
        images = os.listdir(image_save_path)
        predict_num = len(images)
        if predict_num > 0:
            testGene = testGenerator(image_save_path, predict_num, True)
            results = self.model.predict_generator(testGene, predict_num, verbose=1)
            saveResult(predict_path, predict_save_path, results)



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = mywindow(MainWindow)  # 注意把类名修改为myDialog
    # ui.setupUi(MainWindow)  myDialog类的构造函数已经调用了这个函数，这行代码可以删去
    MainWindow.show()
    sys.exit(app.exec_())
