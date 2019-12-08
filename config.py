#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
config file for inference nets

"""

class MNISTCONFIG():
    MODEL_PB_FILE = "lenet5.pb"
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    INPUT_CHANNEL = 1
    OUTPUT_NAME = "dense_1/Softmax"


class VGGCONFIG():
    MODEL_PB_FILE = "vgg16cam.pb"
    MODEL_FILE = "vgg16cam.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (3, 224, 224)
    INPUT_CHANNEL = 3
    OUTPUT_NAME = "dense/Softmax"


class BSICONFIG():
    MODEL_PB_FILE = "opt_vggcam.pb"
    MODEL_FILE = "opt_vggcam.uff"
    INPUT_NAME ="input_tensor"
    INPUT_SHAPE = (3, 256, 256)
    INPUT_CHANNEL = 3
    OUTPUT_NAME = "softmax_tensor"

