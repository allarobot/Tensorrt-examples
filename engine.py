#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
config file for inference nets

"""
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import tensorflow as tf
import numpy as np
import common
import numpy as np
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))