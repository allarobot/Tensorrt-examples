#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
"""
import tensorflow as tf
import numpy as np

class CNN(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print("Node:",node)

        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(np.float32, shape=[None, 224, 224, 3], name='input')
            #self.dropout_rate = tf.placeholder(tf.float32, shape=[], name='dropout_rate')
            tf.import_graph_def(graph_def, {'input_1': self.input})

        self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        # the last one would be output
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print("Layer:",layer)

        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph=self.graph)

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/dense/Softmax:0")
        output = self.sess.run(output_tensor, feed_dict={self.input: data})

        return output

if __name__ == "__main__":
    from vgg_data import *
    import cv2
    net = CNN("models/vgg16cam_2.pb")

    #get data for testing
    x_test, y_test = get_test_dataset()
    idx = [i for i in range(len(y_test))]
    np.random.shuffle(idx)
    x,y = x_test[idx[0:10]],y_test[idx[0:10]]
    output = net.test(x)
    for n,item in enumerate(output):
        pred = np.argmax(item)
        print("results: ",item, "predicted: ", pred,"actual: ",y[n])