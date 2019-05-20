from models.model import *
from models.load_data import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import keras
import datetime
# from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# config.gpu_options.allow_growth = True

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model_save_path = 'D:/pythoncode/segment/data/save_models/'
savemodel_threshold = 0.981  # 保存模型的最低准确率
max_acc = 0


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        global max_acc
        if logs.get('val_acc') > savemodel_threshold and logs.get('val_acc') > max_acc:
            nowTime1 = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M_')
            model_name = model_save_path + str(nowTime1) + str('{:.2f}'.format(logs.get('val_acc') * 100))
            with open(model_name + ".txt", 'w') as f:
                f.writelines('model:unet\n')
                f.writelines('Train epoch:\n'+str(len(self.val_acc['epoch'])))
            f.close()
            model.save(model_name+'.h5')
        max_acc = logs.get('val_acc')

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        # plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
        #    plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="upper right")
        print('max valid:', max(self.val_acc['epoch']))
        global max_valid_result
        max_valid_result = max(self.val_acc['epoch'])
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()


if __name__ == '__main__':
    train_path = "D:/pythoncode/segment/data/spine/train/"
    valid_path = "D:/pythoncode/segment/data/spine/valid/"
    train_data = "D:/pythoncode/segment/data/spine/train/image/"
    train_label = "D:/pythoncode/segment/data/spine/train/label/"
    test_data = "D:/pythoncode/segment/data/spine/test/image1/"
    predict_path = "D:/pythoncode/segment/data/spine/test/predict/"
    # train_data = "D:/pythoncode/segment/data/spine/train/image1/"
    # train_label = "D:/pythoncode/segment/data/spine/train/label1/"

    data_gen_args = dict(samplewise_std_normalization=False, samplewise_center=True,  rotation_range=0, width_shift_range=0.05, height_shift_range=0.1,
                         shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='constant', cval=0)

    trainGen = trainGenerator(1, train_path, 'image', 'label', data_gen_args, save_to_dir=False)
    validGen = trainGenerator(1, valid_path, 'image', 'label', data_gen_args, save_to_dir=False)
    model = unet(input_size=(256, 256, 1))
    # model = unet(model_save_path + "unet_spine5.hdf5", input_size=(880, 880, 1))
    model_checkpoint = ModelCheckpoint(model_save_path+"unet_spine.hdf5", monitor='loss', verbose=2, save_best_only=True)
    history = LossHistory()
    model.fit_generator(trainGen, steps_per_epoch=1104*2/8, validation_data=validGen, validation_steps=31,
                        epochs=30, verbose=2, callbacks=[history])                   #validation_steps=126*2
    history.loss_plot('epoch')

    # model = load_model(model_save_path + "2019-05-18_17-13_98.24.h5")

    # testGene = testGenerator(test_data, 4)

    predict_num = 10
    predict_all_flag = True
    predict_data_path = valid_path+'image/'
    images = os.listdir(predict_data_path)
    if predict_all_flag is True or predict_num > len(images):
        predict_num = len(images)
    testGene = testGenerator(predict_data_path, predict_num, predict_all_flag)

    results = model.predict_generator(testGene, predict_num, verbose=1)
    saveResult(predict_path, predict_data_path, results)

