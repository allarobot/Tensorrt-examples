#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This file contains functions for training a TensorFlow model
"""

import tensorflow as tf
from config import MNISTCONFIG as CONFIG

class Mnist():
    def __init__(self):
        # super(Mnist,self).__init__()
        self.filename = CONFIG.MODEL_PB_FILE

    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=[28,28, 1]))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
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

def main():
    import mnist_data
    import cv2

    o_mnist = Mnist()
    x_train, y_train, x_test, y_test = mnist_data.process_dataset()
    o_mnist.create_model()
    # Train the model on the data
    print(x_train.shape,y_train.shape)
    o_mnist.model.fit(x_train, y_train, epochs = 5, verbose = 1)
    # Evaluate the model on test data
    o_mnist.model.evaluate(x_test, y_test)

    x = cv2.imread("data/1_L.jpg",cv2.IMREAD_UNCHANGED)
    x = x.reshape((1,28,28,1))
    y =o_mnist.model.predict(x)
    print("predict result y: ",y)
    o_mnist.save()

if __name__ == '__main__':
    main()

