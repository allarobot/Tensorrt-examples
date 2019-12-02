#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This sample uses a UFF vgg model to create a TensorRT Inference Engine

"""
import  tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers.core import Flatten, Dense, Dropout, Lambda
from config import VGG16CAMCONFIG

def global_average_pooling(x):
    return tf.keras.backend.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

class VGG16CAM():
    def __init__(self):
        self.filename = VGG16CAMCONFIG.MODEL_PB_FILE

    def create_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=[None,None, 1]))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(64,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64,(3,3),activation=tf.nn.relu))
        self.model.add(MaxPooling2D((2,2),strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128,(3,3),activation=tf.nn.relu))
        self.model.add(MaxPooling2D((2,2),strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256,(3,3),activation=tf.nn.relu))
        self.model.add(MaxPooling2D((2,2),strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512,(3,3),activation=tf.nn.relu))
        self.model.add(MaxPooling2D((2,2),strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512,(3,3),activation=tf.nn.relu))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512,(3,3),activation=tf.nn.relu))


        self.model.add(Lambda(global_average_pooling,
              output_shape=global_average_pooling_shape))
        self.model.add(Dense(2, activation='softmax', init='uniform'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def save(self):
        # First freeze the graph and remove training nodes.
        output_names = self.model.output.op.name
        sess = tf.keras.backend.get_session()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
        # Save the model
        with open(self.filename, "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())