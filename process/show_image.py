import matplotlib
matplotlib.use('TkAgg')
import cv2
import scipy.misc
import skimage.io as io
from skimage import img_as_float64
from matplotlib import pylab as plt
import nibabel as nib
from PIL import Image
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
from models.load_data import *
# img = nib.load('E:/data/spine/train/image/Case1.nii')
example_filename = 'E:/data/spine/train/image/Case1.nii'
test = "D:/pythoncode/segment/data/final_result/save_nii/Case196.nii.gz"
# example_filename = 'E:/data/spine/train/label/mask_case1.nii'

"""
    本段程序只是初期调试时实验的，没必要看

"""

img = nib.load(test)
print(img)
print(img.header['db_name'])  # 输出头信息

width, height, queue = img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()


# img1 = img.get_data()
# dst = img_as_float64(img1[:, :, 5]/np.max(img1[:, :, 5]))
# io.imsave("./data/spine/1_predict.png", dst)
#
#
# # cv2.imshow('image', img1[:, :, 5])
# # cv2.waitKey(0)
#
# scipy.misc.imsave("./data/spine/2_predict.png", img1[:, :, 5])
#
# # int16
#
# # num = 1
# # for i in range(0, queue, 10):
# #     img_arr = img.dataobj[:, :, i]
# #     plt.subplot(5, 4, num)
# #     plt.imshow(img_arr, cmap='gray')
# #     num += 1
# #
# # plt.show()
#
# train_data_path = "./data/spine/train/image/"
# train_label_path = "./data/spine/train/label/"
# test_data_path = "./data/spine/test/image/"
#
# train_data = "./data/spine/train/image1/"
# train_label = "./data/spine/train/label1/"
#
# data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
#                          shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
# save_temp = '../data/spine/train/save'
# myGene = trainGenerator(2, '../data/spine/train', 'image', 'label', data_gen_args, save_to_dir=False)
# train_image1 = io.imread("../data/spine/test/predict1/3_predict.png")
# print(train_image1[330])
# # train_image1 = io.imread(train_data + "image_0_0.png")
#     # print(train_image1[500][50:100])
#     # for image_show in myGene:
#     #     # io.imshow(image_show)
#     #     # io.show()
#     #     ttt = image_show[0][0][:, :, 0]
#     #     print(ttt[300][0:30])
#     #     print(np.max(ttt))
#     #     print(image_show[1][0].shape)
#     #     io.imshow(ttt)
#     #     io.show()
