import numpy as np
import skimage.io as io
import os
from keras.preprocessing.image import ImageDataGenerator
import skimage.transform as trans
import tensorflow as tf
# import matplotlib
# matplotlib.use('TkAgg')
# from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True      # 限制gpu初始资源分配

'''
  加载网络的训练集验证集测试集，在训练时，验证集和测试集采用的是同一个文件夹下的图片
'''

Sky = [0, 0, 0]
Building = [128, 0, 0]

# Sky = [128, 128, 128]
# Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    if(flag_multi_class):
        img = histequ_train(img)
        img = img / 255
        # img = img / np.max(img)
        mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1]*new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        # img = histequ_train(img)
        img = img / 255
        # mask = mask / 65535
        # img = img / np.max(img)
        # mask = mask / np.max(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(880, 880), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    print('loading data...')
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)
    # return zip(image_generator, mask_generator)

def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    # for i in range(num_class):
    #     img_out[img == i, :] = i
    return img_out


def histequ_train(gray, nlevels=256):
    """
        训练集直方图均衡化，训练集和测试集的格式不一样

    """
    # Compute histogram
    gray = gray[0, :, :, 0]   #格式转换，训练集gray的shape为[1,880,880,1]，需要转换成[880,880]
    # print(gray[0, :, :, 0].shape)
    gray = gray.astype('int64')
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    # print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    # print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    # uniform_gray = np.zeros(gray.shape, dtype='float32')
    for i in range(height):
        for j in range(width):
            uniform_gray[i, j] = uniform_hist[gray[i, j]]
    # uniform_gray = np.zeros((1, height, width, 1))
    np.reshape(uniform_gray, (1,) + uniform_gray.shape)
    uniform_gray = np.reshape(uniform_gray, uniform_gray.shape + (1,))
    uniform_gray = np.reshape(uniform_gray, (1,)+uniform_gray.shape) #格式转换，shape [880,880]->[1,880,880,1]
    # uniform_gray[:, :, :, 0] = uniform_gray[:, :]
    return uniform_gray

def histequ_test(gray, nlevels=256):
    """
    测试集直方图均衡化，训练集和测试集的格式不一样
    """
    # gray = gray.astype('int64')
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    # print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    # print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity

    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    # uniform_gray = np.zeros(gray.shape, dtype='float32')
    for i in range(height):
        for j in range(width):
            uniform_gray[i, j] = uniform_hist[gray[i, j]]

    return uniform_gray

def testGenerator(test_path, num_image, predict_all=False, target_size=(880, 880), flag_multi_class=False, as_gray=True):
    images = os.listdir(test_path)
    if predict_all is True or num_image > len(images):
        num_image = len(images)

    # print(images)
    # num_image = len(images)
    # print('test:', num_image)
    # if(num_image==0):
    #     num_image = len(images)
    #     print('num_image:', num_image)
    for i in range(num_image):
        img = io.imread(test_path+images[i], as_gray=as_gray)
        img = histequ_test(img)
        img = img / 255
        img = img - np.mean(img)
        # img = img / np.std(img)
        # img = img / np.max(img)
        img = trans.resize(img, target_size, mode='constant')
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img

def saveResult(save_path, test_path, npyfile,flag_multi_class = False, num_class = 2):
    images = os.listdir(test_path)
    print("image:", len(images))
    # if num_image > len(images):
    #    num_image = len(images)
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        # print(np.max(img))
        img[img > 0.4] = 255
        img[img <= 0.4] = 0
        img = img.astype(np.uint8)
        print("i:", i)
        io.imsave(os.path.join(save_path, "pre_"+str(images[i])), img)


if __name__ == '__main__':
    train_data_path = "../data/spine/train/image/"
    train_label_path = "../data/spine/train/label/"
    test_data_path = "../data/spine/test/image/"

    train_data = "../data/spine/train/image1/"
    train_label = "../data/spine/train/label1/"

    """
    下面的程序只是以前测试用的
    """

    # images = os.listdir(train_data)
    # for image_name in images:
    #     if "image" in image_name:
    #         # print('_'.join(("label",image_name.split("_", 1)[1])))
    #         continue
    # train_image = io.imread(train_data+images[0])
    # # io.imshow(train_image)
    # # io.show()
    # imgs_p = cv2.resize(train_image, (2200, 2200), interpolation=cv2.INTER_CUBIC)
    # io.imshow(imgs_p)
    # io.show()

    # data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
    #                      shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
    # save_temp = 'D:/pythoncode/segment/data/spine/train/save'
    # myGene = trainGenerator(2, 'D:/pythoncode/segment/data/spine/train', 'image', 'label', data_gen_args, save_to_dir=False)

    # model = unet(input_size=(880, 880, 1))
    # model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # model.fit_generator(myGene, steps_per_epoch=1000, epochs=5, callbacks=[model_checkpoint])
    # model =load_model("D:/pythoncode/segment/process/unet_membrane.hdf5")
    # testGene = testGenerator("D:/pythoncode/segment/data/spine/test/image1")

    # train_image1 = io.imread(train_data + "image_0_0.png")
    # print(train_image1[500][50:100])
    # for image_show in myGene:
    #     # io.imshow(image_show)
    #     # io.show()
    #     ttt = image_show[0][0][:, :, 0]
    #     print(ttt[300][0:30])
    #     print(np.max(ttt))
    #     print(image_show[1][0].shape)
    #     io.imshow(ttt)
    #     io.show()

    # results = model.predict_generator(testGene, 10, verbose=1)
    # saveResult(save_temp, results)

    # load_model_path = "D:/pythoncode/segment/process/unet_membrane.hdf5"
    # model = load_model(load_model_path)
    # results = model.predict_generator(testGene, 11, verbose=1)
    # saveResult(save_temp, results)

