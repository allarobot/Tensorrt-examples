
import tensorflow as tf
import struct
import matplotlib.pyplot as plt
import numpy as np


def read_train_file():
    """
    60000  images and labels for trainning
    :return:
    """
    with open('data/MNIST_data/train-images-idx3-ubyte', 'rb') as f:
        train_image = f.read()
    with open('data/MNIST_data/train-labels-idx1-ubyte', 'rb') as f:
        train_labels = f.read()
    return train_image, train_labels


def read_test_file():
    """
    10000  images and labels for test
    :return:
    """
    with open('data/MNIST_data/train-images-idx3-ubyte', 'rb') as f:
        train_image = f.read()
    with open('data/MNIST_data/train-labels-idx1-ubyte', 'rb') as f:
        train_labels = f.read()
    return train_image, train_labels

'''
读取前n张图片
'''


def get_images(buf, n):
    im = []
    index = struct.calcsize('>IIII')
    for i in range(n):
        temp = struct.unpack_from('>784B', buf, index)
        im.append(np.reshape(temp, (28, 28)))
        index += struct.calcsize('>784B')
    return im


'''
读取前n个标签
'''


def get_labels(buf, n):
    l = []
    index = struct.calcsize('>II')
    for i in range(n):
        temp = struct.unpack_from('>1B', buf, index)
        l.append(temp[0])
        index += struct.calcsize('>1B')
    return l


'''
读取
'''
image, label = read_train_file()
n = 16
train_img = get_images(image, n)
train_label = get_labels(label, n)

'''
显示
'''
for i in range(16):
    plt.subplot(4, 4, 1 + i)
    title = u"label:" + str(train_label[i])
    plt.title(title)
    plt.imshow(train_img[i], cmap='gray')
plt.show()
