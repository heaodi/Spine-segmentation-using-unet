from PyQt5.QtWidgets import QFileDialog, QApplication
from mainwindow import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from models.load_data import *
from models.model import *
from scipy import misc
import nibabel as nib
import shutil
import glob
import sys
import os
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True      # 限制gpu初始资源分配


model_save_path = './data/save_models/'
predict_path = "./data/spine/valid/image/"
test_path = "./data/spine/test_image/"
image_save_path = "./data/spine/result/test_save/"
predict_save_path = "./data/spine/result/predict_save/"

class WorkThread(QThread):
    trigger = pyqtSignal()

    def __int__(self):
        super(WorkThread, self).__init__()

    def run(self):
        for i in range(2000000000):
            pass

        # 循环完毕后发出信号
        self.trigger.emit()


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=Ui_MainWindow):
        super(mywindow, self).__init__(parent)
        self.setupUi(parent)
        self.pushButton.clicked.connect(self.startClink)
        self.pushButton_2.clicked.connect(self.previousClink)
        self.pushButton_3.clicked.connect(self.nextClink)
        # self.horizontalScrollBar.isRightToLeft.connect(self.pushButton_2)
        # self.horizontalScrollBar.mousePressEvent()
        self.indexnum = 0
        self.queue = 0
        self.startclink = False
        self.loadrpedictmodel()


    def startClink(self):
        self.label_5.setText("正在预测,请稍后...")
        self.label_5.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "  color:rgb(255, 0, 0);\n"
                                   "}")
        self.startclink = True
        # valid_label = predict_path + "image_175_5.png"
        # img_qt = QPixmap(valid_label).scaled(self.label.width(), self.label.height())
        # self.label.setPixmap(img_qt)
        self.filename = self.msg()
        # print(self.filename)
        filetype = self.filename.split(".")[-1]
        # print(filetype)
        if filetype == "gz":
            self.nii_gz2png()
            self.horizontalScrollBar.setValue(0)
            self.predict_picture()
        self.show_image()
        # self.nii_gz2png(filename)

    def previousClink(self):
        if self.startclink:
            if self.indexnum == 0:
                self.indexnum = 0
            else:
                self.indexnum -= 1
                self.horizontalScrollBar.setValue(self.indexnum)
                self.show_image()

    def nextClink(self):
        if self.startclink:
            if self.indexnum >= self.queue -1:
                self.indexnum = self.queue -1
                self.horizontalScrollBar.setValue(self.indexnum)
            else:
                self.indexnum += 1
                self.horizontalScrollBar.setValue(self.indexnum)
                self.show_image()


    def msg(self):
        filename1, filetype = QFileDialog.getOpenFileName(self,
                                                         "选取文件",
                                                          test_path,
                                                          "All Files (*);;Gz Files (*.gz);;Image Files(*.png)")  # 设置文件扩展名过滤,注意用双分号间隔
        # print(filename1, filetype)
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
        # print("z:", z)
        for j in range(0, self.queue):
            if (j+1) < 10:
               misc.imsave(image_save_path + z + '_0' + str(j) + '.png', img[:, :, j])
            else:
               misc.imsave(image_save_path + z + '_' + str(j) + '.png', img[:, :, j])

    def show_image(self):
        train_data = glob.glob(image_save_path + "*.png")
        predict_data = glob.glob(predict_save_path + "*.png")
        if train_data is not None:
            # self.img = self.img_src.get_data()
            if self.indexnum > self.queue:
                self.indexnum = self.queue
            elif self.indexnum < 0:
                self.indexnum = 0
            # print(train_data[self.indexnum])
            self.img_show = QPixmap(train_data[self.indexnum]).scaled(self.label.width(), self.label.height())
            self.pre_show = QPixmap(predict_data[self.indexnum]).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(self.img_show)
            self.label_3.setPixmap(self.pre_show)


    def loadrpedictmodel(self):
        class_weight = [0.4, 0.6]
        def weighted_binary_crossentropy(y_true, y_pred):
            class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
            return K.sum(class_loglosses * K.constant(class_weight))
        self.model = load_model(model_save_path + "2019-05-29_16-51_98.32.h5", custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

    def predict_picture(self):
        images = os.listdir(image_save_path)
        predict_num = len(images)
        if predict_num > 0:
            testGene = testGenerator(image_save_path, predict_num, True)
            QApplication.processEvents()
            self.results = self.model.predict_generator(testGene, predict_num, verbose=0)
            QApplication.processEvents()
            self.saveResult()
            self.label_5.setText("已完成")
            self.label_5.setStyleSheet("QLabel\n"
                                       "{\n"
                                       "  color:rgb(0, 255, 51);\n"
                                       "}")

    def saveResult(self):
        images = os.listdir(image_save_path)
        print("image:", len(images))
        # if num_image > len(images):
        #    num_image = len(images)
        for i, item in enumerate(self.results):
            img = labelVisualize(2, COLOR_DICT, item) if False else item[:, :, 0]
            # print(np.max(img))
            img[img > 0.4] = 255
            img[img <= 0.4] = 0
            img = img.astype(np.uint8)
            QApplication.processEvents()
            io.imsave(os.path.join(predict_save_path, "pre_"+str(images[i])), img)
            QApplication.processEvents()


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setStyleSheet("#MainWindow{background-image:url(D:/pythonCode/segment/img/frame.png);}")
    ui = mywindow(MainWindow)  # 注意把类名修改为myDialog

    MainWindow.show()
    sys.exit(app.exec_())
