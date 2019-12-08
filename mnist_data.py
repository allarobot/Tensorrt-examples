#!/usr/bin/python
# -*- coding:utf-8 -*-
"""


"""


import tensorflow as tf
import struct
import numpy as np


def read_train_file():
    """
    buffer of 60000  images and labels for training
    :return:
    """
    with open('data/MNIST_data/train-images-idx3-ubyte', 'rb') as f:
        train_image = f.read()
    with open('data/MNIST_data/train-labels-idx1-ubyte', 'rb') as f:
        train_labels = f.read()
    return train_image, train_labels


def read_test_file():
    """
    buffer of 10000  images and labels for test
    :return:
    """
    with open('data/MNIST_data/t10k-images-idx3-ubyte', 'rb') as f:
        test_image = f.read()
    with open('data/MNIST_data/t10k-labels-idx1-ubyte', 'rb') as f:
        test_labels = f.read()
    return test_image, test_labels


def get_images(buf, n):
    '''
    get n images from image buffer

    '''

    im = []
    index = struct.calcsize('>IIII')
    for i in range(n):
        temp = struct.unpack_from('>784B', buf, index)
        im.append(np.reshape(temp, (28, 28)))
        index += struct.calcsize('>784B')
    im = np.array(im)
    return im


def get_labels(buf, n):
    '''
    get n labels from label buffer
    '''
    l = []
    index = struct.calcsize('>II')
    for i in range(n):
        temp = struct.unpack_from('>1B', buf, index)
        l.append(temp[0])
        index += struct.calcsize('>1B')
    l = np.array(l) 
    return l


def get_mnist():
    '''
    retrive mnist dataset
    '''
    x1,y1 = read_train_file()
    x2,y2 = read_test_file()
    x_train = get_images(x1,60000)
    x_test = get_images(x2,10000)
    y_train = get_labels(y1,60000)
    y_test = get_labels(y2,10000)
    return x_train,y_train,x_test,y_test

def process_dataset():
    # Import the data
    #(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train,y_train,x_test,y_test = get_mnist()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("x_train shape: ",x_train.shape," x_test shape: ",x_test.shape)
    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
	
    #import matplotlib.pyplot as plt
    #'''
    #读取
    #'''
    #image, label = read_train_file()
    #n = 16
    #train_img = get_images(image, n)
    #train_label = get_labels(label, n)

    #'''
    #显示
    #'''
    #for i in range(16):
    #    plt.subplot(4, 4, 1 + i)
    #    title = u"label:" + str(train_label[i])
    #    plt.title(title)
    #    plt.imshow(train_img[i], cmap='gray')
    #plt.show()

    '''
    save test picture
    '''
    import cv2
    from random import randint

    image_buffer, label_buffer = read_test_file()
    n = 10000 #randint(0,10000)
    images = get_images(image_buffer, n)
    labels = get_labels(label_buffer, n)
    for i in range(n):
        img = images[i]
        #cv2.resize(img,(500,500))
        #print("image shape before resize: {}\n".format(img.shape))
        cv2.imwrite("data/{}_{}.jpg".format(i,labels[i]),img)




