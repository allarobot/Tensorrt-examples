#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
# This sample uses a UFF MNIST model to create a TensorRT Inference Engine

"""

from random import randint
#from PIL import Image
import cv2
import numpy as np
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import tensorrt as trt
import common
from config import MNISTCONFIG as CONFIG
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(model_file):
    print("model file: ",model_file)
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        #builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(CONFIG.INPUT_NAME, CONFIG.INPUT_SHAPE)
        parser.register_output(CONFIG.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


if __name__ == '__main__':
  
    import jetson.inference
    import jetson.utils
    import argparse

    # parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile",type=str, help="filename of the model")
    parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
    parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
#    parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")
    opt = parser.parse_args()

    # Using VideoCapture to access image stream
    cap = cv2.VideoCapture(0)

    model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, CONFIG.MODEL_FILE)
    print("model file: ",model_file)
    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            while True:
                _,img = cap.read()
                img2 = cv2.cvtColor(cv2.resize(img,CONFIG.INPUT_SHAPE[1:]),cv2.COLOR_BGR2GRAY)

                img2 = np.array(img2).ravel()
                np.copyto(inputs[0].host, 1.0 - img2 / 255.0)

                #********************************************************************************
                # The common.do_inference function will return a list of outputs - we only have one in this case.
                [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                pred = np.argmax(output)
                cv2.putText(img,"Status: #{} detected".format(pred),(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                cv2.imshow("",img)

                if chr(cv2.waitKey(1) & 0xff) == 'q':
                    break

                print("Prediction: " + str(pred))

            cv2.destroyAllWindows()
    cap.release()






