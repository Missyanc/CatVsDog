import os, cv2, random
import numpy as np
import pandas as pd


TRAIN_DIR = 'data/train/'
TEST_DIR = 'E:/PycharmProjects/CatVsDog/media/img/'

# ROWS = 150
# COLS = 150

#
ROWS = 128
COLS = 128
CHANNELS = 3

train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]

train_images = train_dogs[:1000] + train_cats[:1000]
random.shuffle(train_images)
test_images = test_images

def get_testData():
    test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
    test_images = prep_data(test_images)
    return test_images

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
    print(data.shape)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        # print()
        # data[i] = image
        data[i] = image.T
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))
    return data

def load_data():


    train = prep_data(train_images)
    test = prep_data(test_images)

    print("Train shape: {}".format(train.shape))
    print("Test shape: {}".format(test.shape))

    return train,test

def test():
    load_data()

if __name__ == '__main__':
    test()