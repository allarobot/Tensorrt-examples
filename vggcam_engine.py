#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This sample uses a UFF MNIST model to create a TensorRT Inference Engine
"""
from engine import  *

from config import VGGCONFIG

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Vggcam():
    def __init__(self):
        self._config = VGGCONFIG

    def build_engine(self,model_file):
        # For more information on TRT basics, refer to the introductory samples.
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            # builder.max_workspace_size = common.GiB(1)
            # Parse the Uff Network
            parser.register_input(self._config.INPUT_NAME, self._config.INPUT_SHAPE)
            parser.register_output(self._config.OUTPUT_NAME)
            parser.parse(model_file, network)
            # Build and return an engine.
            self._engine = builder.build_cuda_engine(network)
        self._inputs, self._outputs, self._bindings, self._stream = common.allocate_buffers(self._engine)
        self._context = self._engine.create_execution_context()
        self.pagelocked_buffer = self._inputs[0].host


    def trt_predict(self, img):
        img = np.array(img).ravel()
        np.copyto(self.pagelocked_buffer, 1.0 - img / 255.0)
        [output] = common.do_inference(self._context, bindings=self._bindings, inputs=self._inputs,
                                       outputs=self._outputs, stream=self._stream)
        pred = np.argmax(output)
        return pred
