import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class HeadPoseDetect:
    '''
    Class for the Head Pose Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
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
        infer_network = self.net.start_async(request_id=0, inputs = {self.input_name:p_image})
        if self.net.requests[0].wait(-1) == 0:     
            result = self.net.requests[0].outputs 
            ypr = self.preprocess_outputs(result)
 
        return ypr
    
    
    def preprocess_outputs(self, outputs):
        #Reference: https://knowledge.udacity.com/questions/242566
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        processed_output = []

        processed_output.append(outputs['angle_y_fc'].tolist()[0][0])
        processed_output.append(outputs['angle_p_fc'].tolist()[0][0])
        processed_output.append(outputs['angle_r_fc'].tolist()[0][0])

        return processed_output



    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        n,c,h,w = self.input_shape
        # Pre-process the frame
        p_frame = cv2.resize(image, (w,h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(n,c,h,w)
        return p_frame

