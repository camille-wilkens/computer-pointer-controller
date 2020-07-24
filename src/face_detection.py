import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

import logging as log


class FaceDetect:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device, extensions=None,threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        '''
        core = IECore()
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        p_image = self.preprocess_input(image)

        self.net.start_async(request_id=0, inputs = {self.input_name:p_image})
        if self.net.requests[0].wait(-1) == 0:      
            results = self.net.requests[0].outputs[self.output_name]
            face_coords = self.preprocess_outputs(results,image)
            if len(face_coords)==0:
                log.error("No Face is detected")
                return 0,0
            face_coords = face_coords[0]
            cropped_face = image[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
        return face_coords, cropped_face
            

    def preprocess_outputs(self, outputs,image):
        '''
        TODO: This method needs to be completed by you
        '''

        faces_coords = []
        outs = outputs[0][0]
        for box in outs:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                faces_coords.append([xmin, ymin, xmax, ymax])
        return faces_coords


    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        resized_frame = image.transpose((2, 0, 1)).reshape((n, c, h, w))
 
        return resized_frame

