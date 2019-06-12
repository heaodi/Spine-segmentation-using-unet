from models.model import *
from models.load_data import *
from models.model1 import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import misc
import shutil
import keras
import datetime

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model_save_path = '../data/save_models/'


def weighted_binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(class_weight))


def loadnii(filename):
    img_src = nib.load(filename)
    width, height, queue = img_src.dataobj.shape
    # print(width, height, queue)
    return img_src

def saveResult(image_save_path, predict_save_path, results):
    images = os.listdir(image_save_path)
    print("image:", len(images))
    # if num_image > len(images):
    #    num_image = len(images)
    for i, item in enumerate(results):
        img = labelVisualize(2, COLOR_DICT, item) if False else item[:, :, 0]
        # print(np.max(img))
        img[img > 0.4] = 255
        img[img <= 0.4] = 0
        img = img.astype(np.uint8)
        io.imsave(os.path.join(predict_save_path, "pre_"+str(images[i])), img)



if __name__ == '__main__':
    test_data = "../data/final_result/test_image/"
    predict_path = "../data/final_result/nii_png/"
    predict_result = "../data/final_result/predict/"
    save_nii_path = "../data/final_result/save_nii/"
    class_weight = [0.4, 0.6]
    model = load_model(model_save_path + "2019-05-27_03-37_98.33.h5", custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

    images = os.listdir(test_data)
    for i in range(0, 1):
    # for i in range(0, len(images)):
        print(images[i])
        # print(test_data+images[i])
        files = test_data+images[i]
        testdata = loadnii(files)
        width, height, queue = testdata.dataobj.shape
        img = testdata.get_data()
        z = files.split(".")[-3].split("/")[-1]
        print("z:", z)
        if os.path.exists(predict_path):
            shutil.rmtree(predict_path)
            shutil.rmtree(predict_result)
            os.mkdir(predict_path)
            os.mkdir(predict_result)
        else:
            os.mkdir(predict_path)
            os.mkdir(predict_result)

        for j in range(0, queue):
            misc.imsave(predict_path + z + '_' + str(j) + '.png', img[:, :, j])

        testGene = testGenerator(predict_path, queue, True)
        results = model.predict_generator(testGene, queue, verbose=0)
        saveResult(predict_path, predict_result, results)





    # if predict_all_flag is True or predict_num > len(images):
    #     predict_num = len(images)
    # testGene = testGenerator(predict_data_path, predict_num, predict_all_flag)
    #
    # results = model.predict_generator(testGene, predict_num, verbose=1)
    # saveResult(predict_path, predict_data_path, results)

