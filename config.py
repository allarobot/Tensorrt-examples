#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
config file for inference nets

"""

class MNISTCONFIG(object):
    MODEL_PB_FILE = "models/lenet5.pb"
    MODEL_FILE = "models/lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"


class MNISTCONFIG(object):
    MODEL_PB_FILE = "models/vgg16cam.pb"
    MODEL_FILE = "models/vgg16cam.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 242, 242)
    OUTPUT_NAME = "dense_1/Softmax"

