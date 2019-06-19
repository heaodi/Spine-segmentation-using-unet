# import cv2
# import os
# from skimage import img_as_float64
# import numpy as np
# import skimage.io as io
import glob
from tqdm import tqdm
import nibabel as nib
from scipy import misc
import matplotlib
matplotlib.use('TkAgg')


def nii2png(data_path, save_path, start_num=0):
    """
    把data_path下的nii转化为png保存在save_path，原始nii为16bit，保存为png时转换成了8bit，也可以保存为16bit，但
    keras的扩充工具只能读取8bit的图片

    """
    train_data = glob.glob(data_path + "*.nii")
    y = data_path.split("/")[-2]+"_"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # train_label = glob.glob(data_path + "label/*.nii")
    # for i in range(0, len(train_data)):
    #     print(train_data[i])
    #     print(train_label[i])
    # img = nib.load(train_data[0]).get_data()
    # for i in range(0, len(train_data)):
    for data_index in tqdm(range(start_num, len(train_data))):
        # print('data name:', train_data[data_index])
    # for i in range(0, 1):
        img_src = nib.load(train_data[data_index])
        width, height, queue = img_src.dataobj.shape
        # print(width, height, queue)
        img = img_src.get_data()
        # img = (img / 256).astype('uint8')
        # img = img_as_float64(img / 65535)
        # img = img_as_float64(img / np.max(img))
        for j in range(0, queue):
            # io.imsave(save_path + y + str(data_index) + '_' + str(j) + '.png', img[:, :, j])
            if (j+1) < 10:
               misc.imsave(save_path + y + str(data_index) + '_0' + str(j) + '.png', img[:, :, j])
            else:
               misc.imsave(save_path + y + str(data_index) + '_' + str(j) + '.png', img[:, :, j])


            # plt.figure(0, figsize=(float(width / 100), float(height / 100)))
            # plt.imshow(showimage, cmap='gray')
            # plt.savefig(save_path + str(data_index) + '_' + str(j) + '.png')
            # plt.close(0)


if __name__ == '__main__':
    train_data_src = "E:/data/spine/train/image/"
    train_label_src = "E:/data/spine/train/label/"
    test_data_src = "E:/data/spine/test/"
    train_data_path = "../data/spine/train/image/"
    train_label_path = "../data/spine/train/label/"
    test_data_path = "../data/spine/test/image/"
    # nii2png(train_data_src, train_data_path)
    # nii2png(train_label_src, train_label_path)
    # nii2png(test_data_src, test_data_path)
