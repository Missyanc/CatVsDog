import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import Include.cVd.load_data as ld
import numpy as np
import os
import random
import cv2

TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test/'

ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]  # use this for full dataset
train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]

train_images = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_images)
test_images = test_images[:25]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))

    return data


train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

def get_lables():
    labels = []
    for i in train_images:
        if 'dog' in i:
            labels.append(1)
        else:
            labels.append(0)


print(labels)
sns.countplot(labels,palette="Greens_d")
plt.title('Cats and Dogs')
plt.show()


def show_cats_and_dogs(idx):
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(20, 10))
    plt.imshow(pair)
    plt.show()


for idx in range(0, 5):
    show_cats_and_dogs(idx)

cat_num = 0
dog_num = 0
for i in labels:
    if labels[i] == 0:
        cat_num = cat_num + 1
    if labels[i] == 1:
        dog_num = dog_num + 1
print('Total cat number: ', cat_num)
print('Total dog number: ', dog_num)


dog_avg = np.array([dog[0].T for i, dog in enumerate(train) if labels[i] == 1]).mean(axis=0)
plt.imshow(dog_avg)
plt.title('Your Average Dog')
plt.show()
# #
cat_avg = np.array([cat[0].T for i, cat in enumerate(train) if labels[i] == 0]).mean(axis=0)
plt.imshow(cat_avg)
plt.title('Your Average Cat')
plt.show()

# def test():
#     return -1
# if __name__ == '__main__':
#     test()