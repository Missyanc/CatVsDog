# CatVsDog
基于卷积神经网络的猫狗识别 web 应用
## 项目简介

- **该项目是基于 Keras 的猫狗识别 web 应用。**

- **数据集是来自 Kaggle 上的猫狗大赛数据集，其中训练集 train 包含了猫的图片 12500 张以及狗的图片 12500 张，测试集 test 包含了猫狗的图片 12500 张。本项目采用了基于 Keras 的自己构造的 cnn 网络训练以及 Keras 中的 VGG16 卷积神经网络模型来进行训练数据，比较发现自己构造的 cnn 网络训练数据得到的模型不如 VGG16 训练数据得到的模型准确率高，自己训练的模型准确率在 70%-80% 之间，而 VGG16 的准确率在 95% 以上。**

- **最后还用了 Python 的 Django 框架简单的做了个展示页面，其中包括了本人及项目介绍图片的上传及预测结果展示的小功能。**

## 项目实现

### 所用技术

- Keras  搭建卷积神经网络
- Keras VGG16 搭建自己的卷积神经网络
- OpenCV 图像识别
- Django 框架
- Python 文件操作

### 项目流程图

![猫狗大战流程图](https://note.youdao.com/yws/api/personal/file/37C61A75D754431D9BFF254298A30716?method=download&shareKey=52f5c8d42d02f877ce43a40b0e9c7695)

------

## cnn 网络图

#### 图片展示

![cnn网络图](https://note.youdao.com/yws/api/personal/file/29BECEF857DD47E0BD36AD35F71DCBF2?method=download&shareKey=50efb0a881f06fdafeef0bfe78e613eb)

#### 文字代码解释

​	**如上图所示，分别有四个输入不同的 `卷积层-->卷积层-->激活层-->Max池化层  `， 然后用 `Flatten 层` 是将图像展平成一维的列表，然后连接上两组 `Dense 层（全连接层）--> Dropout 层`，最后是一个 `Dense层` 进行最后的二分类。**

```python
from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 
```

> 第一行导入 Sequential 来初始化网络模型作为一个序列化网络。一般有两种基本方法去初始化网络，要么是各种层的序列，要么是用图的方式。 
>
> 第二行 引入 Conv2D 来完成卷积操作。我们这里使用的图像基本是 2 维数组，所以使用 Conv2D，当处理视频文件时要使用 Conv3D, 第三维是时间。 
>
> 第三行 引入 MaxPooling2D 用作池化操作。使用最大池化来保留感兴趣区域中的最大像素值。还有其他池化操作例如 MinPooling, MeanPooling. 
>
> 第四行引入扁平操作，用来转化所有 2 维数组到单个长的连续线性向量。 
>
> 第五行引入 Dense 层，用来完成网络的全连接。

##### **现在创建一个序列的 object**

###### 代码展示

```python 
def catdog():
    model = Sequential()
	# 增加第一个卷积操作： 
    model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model
```

###### 解释：

> `model = Sequential()`
> 增加第一个卷积操作： 
>
> `model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=(3, ROWS, COLS), activation='relu'))`
>
> 网络序列化的意思是我们可以通过 add 增加每一层到 model 上。Conv2D 函数包含 4 个参数，第一个是卷积核的个数也就是 32， 第二个参数时卷积核的大小也就是（3，3），第三个参数是输入大小和图像类型（RGB 还是黑白图）这里表示卷积网络输入 (128,128) 分辨率的图像，3 表示 RGB 图像。第四个参数为使用的激活函数。
>
> 接下来需要完成池化操作在相关特征图（由卷积操作之后产生的 32 个 feature maps）上。池化操作主要意图是减少图像大小。关键就是我们试图为下一层减少总共的节点数。
>
> `model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))`
> 我们使用 2*2 矩阵，最小化像素损失并得到特征定位的精确区域。我们在不损失精度时减少模型复杂度。 
>
> 接下来转化所有池化后图像到一个连续向量通过 Flattening 函数，将 2 维矩阵转化到一维向量。
>
> `model.add(layers.Flatten())`
>
> 然后创建全连接层。扁平后的节点将作为输入连接到全连接层。 
>
> `model.add(layers.Dense(256, activation='relu'))`
>
> Dense 函数创建一个全连接层，”units” 表示该层中定义的节点数，这些单元值一般在输入节点数和输出节点数之间，但只能通过实验来试出最好的值。
>
> `model.add(Activation('sigmoid'))`
>
> 最后初始化输出层，其只包含一个节点，对于二元分类来说。
>
> model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
>
> 现在完成了建立 CNN 模型，需要编译它。

## 项目展示

### 前端展示

- **项目主页**

  ![index](https://note.youdao.com/yws/api/personal/file/C0C269DFD2F5408CBC8DB0D45841DD93?method=download&shareKey=1122a3b48d4c98680d564481d68500de)



- **上传猫狗图片**

  ![uploadimg](https://note.youdao.com/yws/api/personal/file/6C2B2194BBB843FC90F41E268674EA09?method=download&shareKey=8096e48e01e51dcf141d90d1915adcce)



- **查看结果**

  ![result](https://note.youdao.com/yws/api/personal/file/354ECED38AD04AD787A903F0C7F85ACB?method=download&shareKey=eb02e8f928c5febb4d740b8871fc99d7)

  

### 后台展示

![houtai](https://note.youdao.com/yws/api/personal/file/1750AD3D4208482696953891FE867AF6?method=download&shareKey=f365d626d28bc92b5d68b2b86811d8b5)



## 具体关键步骤

### 读取数据

> 对 2000 个图像（8％）进行采样，以便我的电脑能够高效跑起来。
>
> 形成下面的文件夹列表：
>
> ![folder](https://note.youdao.com/yws/api/personal/file/A3F11CCC21204D598B0E716E0C23F1D7?method=download&shareKey=46c3ae67b4ddcde08b5047efa9d7ab3f)

```python
# 数据集解压之后的目录
original_dataset_dir = 'data/train'
# 存放小数据集的目录
base_dir = 'data/small_data'
# os.mkdir(base_dir)

# 建立训练集、验证集、测试集目录
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

# 将猫狗照片按照训练、验证、测试分类
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# 切割数据集
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dat)
```

### 建立模型

参照上面的 CNN 网络解释。

### 训练网络

#### 调整图像像素值

在训练数据之前，需要预处理图像防止过拟合。首先在数据集上做数据增强 image augmentations. 使用 keras.preprocessing 库来合成部分去准备训练集和测试集，字典的名称为所有图像呈现的类别。

```python
# 调整像素值
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,     # 宽度平移
                                   height_shift_range=0.2,   # 高度平移
                                   shear_range=0.2,            # 修剪
                                   zoom_range=0.2,            # 缩放
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator( rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,     # 宽度平移
                                   height_shift_range=0.2,   # 高度平移
                                   shear_range=0.2,            # 修剪
                                   zoom_range=0.2,            # 缩放
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
```

#### 增加容错技术 `CheckPoint`

```python
save_dir = os.path.join(os.getcwd(), 'saved_models')
filepath="model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc',verbose=1,
                            save_best_only=True)
```

#### 利用批量生成器拟合模型

```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[checkpoint])
```

#### 保存模型

```python
model.save('cats_and_dogs_small_2.h5')
```

#### 画出损失以及准确率

```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```

### Web 

利用 Python Django 框架完成了猫狗图片的上传以及预测结果展示结果的简单 Web 页面。

**部分截图如下：**

![show](https://note.youdao.com/yws/api/personal/file/01F5B62998F54A288449491CB26DD3D6?method=download&shareKey=b472c84ab0e983cec328b686260477ea)

## 项目总结

1. 本质上猫狗大战项目是一个二分类问题，是一个相对简单的 cnn 入门问题，所以自己写起来不是那么的困难。而我下一步准备进行更多的 cnn 学习以及训练

2. 在用了以上自己搭建的模型进行训练之后准确率并不是很高，最多只能达到 75% - 85% 之间，后来换了 VGG16 模型进行搭建模型之后，准确率达到了 95%。以下展示了 VGG16 创建的模型源代码。

   ```python
   path='D:/Downloads/vgg16_weights_tf_dim_ordering_tf_kernels_notop/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
   
   conv_base = VGG16(weights=path,
                     include_top=False,
                     input_shape=(150, 150, 3))
   set_trainable = False
   for layer in conv_base.layers:
       if layer.name == 'block5_conv1':
           set_trainable = False
       if set_trainable:
           layer.trainable = True
       else:
           layer.trainable = False
   
   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
   model.add(layers.Dense(1, activation='sigmoid'))
   ```

3. 对于相对较大的数据的处理我的电脑还是相对较弱，训练 2000 张图片并且把 cnn 网络的输入调成了 （150，150，3）都花费了近 13 个小时才训练好模型，其中每个 Batch 的 size 为 32，训练了 100 个 epoch。

4. 对于神经网络的理解还不够深，还需要我更多的学习才行。目前只能初步的用这些写好的框架，而深入的使用以及对这些框架的源码的理解都还未达到。

5. 第一次使用 Python 的 Django 框架，深深被其吸引。其自带数据库的方便以及各个层之间的关联都非常棒。
