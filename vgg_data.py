#!/usr/bin/python
# -*- coding:utf-8 -*-
"""


"""
import glob
import cv2
import os,sys
import numpy as np
from config import VGGCONFIG as CONFIG
def process_dataset():
    print("in function")
    sys.path.append("C://Users/Administrator//Documents//GitKraken//tensorrt_examples")
    print("\n".join(sys.path))

    train_files = glob.glob("data/INRIAPerson/Train/*/*")
    train_labels = []
    train_images = []
    for file in train_files:
        print(file)
        try:
            im = cv2.imread(file, cv2.IMREAD_COLOR)
            train_images.append(cv2.resize(im,(224,224)))
            if 'pos' in file:
                train_labels.append(1)
            else:
                train_labels.append(0)
        except:
            pass
    print(train_files)

    test_files = glob.glob("data/INRIAPerson/Test/*/*")
    test_labels = []
    test_images = []
    for file in test_files:
        print(file)
        try:
            im = cv2.imread(file, cv2.IMREAD_COLOR)
            test_images.append(cv2.resize(im,(224,224)))
            if 'pos' in file:
                test_labels.append(1)
            else:
                test_labels.append(0)
        except:
            pass
    print(test_files)

    size = len(train_labels)
    train_images = np.reshape(train_images, (size,224,224,3))
    size = len(test_labels)
    test_images = np.resize(test_images, (size,224,224,3))

    print(train_labels,len(train_labels),test_labels,len(test_labels))
    return train_images,np.array(train_labels),test_images,np.array(test_labels)

def get_test_dataset(filepath="data/INRIAPerson/Test/*/*"):
    print("in function")
    print("\n".join(sys.path))

    test_files = glob.glob(filepath)
    test_labels = []
    test_images = []
    for file in test_files:
        try:
            im = cv2.imread(file, cv2.IMREAD_COLOR)
            test_images.append(cv2.resize(im,(224,224)))
            if 'pos' in file:
                test_labels.append(1)
            else:
                test_labels.append(0)
        except:
            pass
    size = len(test_labels)
    test_images = np.resize(test_images, (size,224,224,3))

    return test_images,np.array(test_labels)

if __name__  ==  "__main__":


    print("create dataset")
    results = process_dataset()
    cv2.imshow("",results[0][0])
    cv2.waitKey(-1)