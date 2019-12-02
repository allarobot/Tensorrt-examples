#!/usr/bin/python
# -*- coding:utf-8 -*-
"""


"""


import cv2
import numpy as np
from inference_engines import mnist

if __name__ == '__main__':
  
    import jetson.inference
    import jetson.utils
    import argparse

    #*****************
    #parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="filename of the image to process")
    opt = parser.parse_args()

    #print("image loaded: ",img.shape,width,height)
    net = mnist()

    # create the camera and display
    font = jetson.utils.cudaFont()
    #camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
    display = jetson.utils.glDisplay()

    # load an image (into shared CPU/GPU memory)
    #img, width, height = jetson.utils.loadImageRGBA(opt.filename)
    img = cv2.imread(opt.filename,cv2.IMREAD_UNCHANGED)
    img = np.array(cv2.resize(img,(28,28)))
    width,height = img.shape
    # process frames until user exits
    while display.IsOpen():

	# capture the image
        img2, width, height = jetson.utils.loadImageRGBA(opt.filename)

        # classify
        result = net.predict(img)

        img_enlarge = cv2.resize(img,(400,400))
	# overlay the result on the image	
        font.OverlayText(img2, width, width, "predict result is {} for image {}".format(result,opt.filename), 5, 5, font.White, font.Gray40)
	
	# render the image
        display.RenderOnce(img2, width, width)

	# update the title bar
        display.SetTitle("BSI assistant")






