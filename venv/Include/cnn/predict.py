from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
import matplotlib.pyplot as plt
from matplotlib import ticker
from keras.models import load_model
import time
import os,cv2
import numpy as np

ROWS = 150
COLS = 150


#
# ROWS = 128
# COLS = 128
CHANNELS = 3
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def predict():

    TEST_DIR = 'E:/PycharmProjects/CatVsDog/media/img/'
    result = []

    # model = load_model('my_model.h5')
    model = load_model('E:\PycharmProjects\CatVsDog\\venv\Include\cnn\cats_and_dogs_small_4.h5')

    test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
    count = len(test_images)
    # data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    #
    print("图片网维度：")
    print(data.shape)

    for i, image_file in enumerate(test_images):
        image = read_image(image_file)
        # print()
        data[i] = image
        # data[i] = image.T
        if i % 250 == 0: print('处理 {} of {}'.format(i, count))

    test = data
    predictions = model.predict(test, verbose=0)
    dict = {}
    urls = []
    for i in test_images:
        ss = i.split('/')
        url = '/' + ss[3] + '/' + ss[4] + '/' + ss[5]
        urls.append(url)
    for i in range(0, len(predictions)):
        if predictions[i, 0] >= 0.5:
            print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
            dict[urls[i]] = "图片预测为：Dog！"
        else:
            print('I am {:.2%} sure this is a Cat'.format(1 - predictions[i][0]))
            dict[urls[i]] = "图片预测为：Cat！"
        plt.imshow(test[i])
        # plt.imshow(test[i].T)
        plt.show()
        # time.sleep(2)
    # print(dict)
    # for key,value in dict.items():
    #     print(key + ':' + value)

    return dict

if __name__ == '__main__':
    result = predict()
    for i in result:
        print(i)

