# Cat Vs Dog

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

----

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

>第一行导入 Sequential 来初始化网络模型作为一个序列化网络。一般有两种基本方法去初始化网络，要么是各种层的序列，要么是用图的方式。 
>
>第二行 引入 Conv2D 来完成卷积操作。我们这里使用的图像基本是 2 维数组，所以使用 Conv2D，当处理视频文件时要使用 Conv3D, 第三维是时间。 
>
>第三行 引入 MaxPooling2D 用作池化操作。使用最大池化来保留感兴趣区域中的最大像素值。还有其他池化操作例如 MinPooling, MeanPooling. 
>
>第四行引入扁平操作，用来转化所有 2 维数组到单个长的连续线性向量。 
>
>第五行引入 Dense 层，用来完成网络的全连接。

##### **现在创建一个序列的 object**

###### 代码展示

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

