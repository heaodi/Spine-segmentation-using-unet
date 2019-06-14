# import glob
from __future__ import division
import os
import numpy as np
import skimage.io as io
from skimage import transform
import datetime
"""
根据比赛的要求，采用的评判标准，这个程序写的不一定标准，可能需要进一步验证

四、评分标准
Dice 相似性系数是用分割标准和提交结果之间的重叠体素量的两倍除以分
割标准和提交结果中的体素量的总和。PPV 是来衡量 TP 与 FP 之间数量关系。
Sensitivity 灵敏度用来计算 TP 和 FN 的量。我们将用 Dice 相似性系数(DSC)，
Positive Predicted Value(PPV)和 Sensitivity3 个值对分割性能的进行评估
DSC =2𝑇𝑃/(𝐹𝑃+2𝑇𝑃+𝐹𝑁)    (1)
PPV =𝑇𝑃/(𝑇𝑃+𝐹𝑃)          (2)
Sensitivity =𝑇𝑃/(𝑇𝑃+𝐹𝑁)  (3)
其中 TP、FP 和 FN 分别表示真阳性、假阳性和假阴性的数量。
"""

def evalue_score(predict_image_path, predict_label_path):
    image_names = os.listdir(predict_image_path)
    label_names = os.listdir(predict_label_path)
    records = []   # 记录内容[ssi, all_acc, label_src.size, TP, FP, FN]
    print("evalue predict ...")
    for image_name in image_names:
        label = "label_" + "_".join(image_name.split("_")[2:4])
        if label in label_names:
            ssi = predict_image_path+image_name
            ssl = predict_label_path+label
            predict_image = io.imread(ssi, as_gray=True)
            label_src = io.imread(ssl, as_gray=True)
            if predict_image.shape[0] != label_src.shape[0]:
                # print(ssi)
                # print(label_src.shape[0])
                # print("PPV:", PPV)
                # io.imshow(predict_image)
                # io.show()
                # print(predict_image[500])
                predict_image = transform.resize(predict_image, (label_src.shape[0], label_src.shape[1]), mode='constant')
                predict_image[predict_image >= 0.5] = 255
                predict_image[predict_image < 0.5] = 0
                # io.imshow(predict_image)
                # io.show()
            print("MAX:", np.max(predict_image))
            predict_image = predict_image/np.max(predict_image)
            label_src = label_src/255
            predict_one = np.sum(predict_image[:] == 1)
            label_one = np.sum(label_src[:] == 1)
            predict_image = (predict_image == 1)
            label_src = (label_src == 1)
            true_matrix = np.bitwise_and(predict_image, label_src)
            all_acc = np.sum(predict_image[:] == label_src[:])
            ACC = all_acc/label_src.size
            TP = np.sum(true_matrix[:])
            FP = predict_one - TP
            FN = np.sum(np.bitwise_xor(true_matrix, label_src)[:])
            DSC = (2 * TP) / (FP + 2 * TP + FN)
            PPV = TP / (TP + FP)
            SEN = TP / (TP + FN)#记录内容[ssi, all_acc, label_src.size, TP, FP, FN]
            records.append([ssi, all_acc, label_src.size, TP, FP, FN])
            # print("score:", '%.2f' % ACC, '%.2f' % DSC, '%.2f' % PPV, '%.2f' % SEN)
            # io.imshow(label_src)
            # io.show()
    return records


if __name__ == '__main__':
    valid_label = "../data/spine/valid/label/"
    valid_image = "../data/spine/test/predict/"
    result_save_path = "../data/results/"
    score = evalue_score(valid_image, valid_label)
    acc = 0
    size = 0
    TP = 0
    FP = 0
    FN = 0
    for single_score in score:
        # print(single_score)#[ssi, all_acc, label_src.size, TP, FP, FN]
        # acc, size, TP, FP, FN = single_score[1:]
        acc += single_score[1]
        size += single_score[2]
        TP += single_score[3]
        FP += single_score[4]
        FN += single_score[5]
        # print(acc, size, TP, FP, FN)
    DSC = (2 * TP) / (FP + 2 * TP + FN)
    PPV = TP / (TP + FP)
    SEN = TP / (TP + FN)
    ACC = acc / size
    SCORE = (DSC+PPV+SEN)/3
    print("score:", '%.2f' % SCORE)
    print("ACC:", '%.2f' % ACC)
    print("DSC:", '%.2f' % DSC)
    print("PPV:", '%.2f' % PPV)
    print("SEN:", '%.2f' % SEN)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M_')
    with open(result_save_path + str(nowTime) + str('{:.2f}'.format(ACC * 100)) + ".txt", 'w') as f:
        f.writelines('model:unet\n')
        f.writelines('DSC:' + str(DSC) + '\n')
        f.writelines('PPV:' + str(PPV) + '\n')
        f.writelines('SEN:' + str(SEN) + '\n')
        f.writelines('ACC:' + str(ACC) + '\n')
        f.writelines('SCORE:' + str(SCORE) + '\n')


