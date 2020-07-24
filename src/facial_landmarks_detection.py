import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys




class FacialLandmarksDetect:
    '''
    Class for the FacialLandmarksDetectn Model.
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
        left_eye = []
        right_eye = []
        p_image = self.preprocess_input(image)
        self.net.start_async(request_id=0, inputs = {self.input_name:p_image})
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs[self.output_name]
            outputs= outputs[0]
            left_eye, right_eye,eye_coords = self.get_eyes(outputs, image)
            
        return left_eye, right_eye, eye_coords

    
    def get_eyes(self, outputs, image):
        '''
        TODO: This method needs to be completed by you
        Draw bounding boxes onto the frame.
        '''
    	#Eye Box code from - https://knowledge.udacity.com/questions/245775

        initial_h = image.shape[0]
        initial_w = image.shape[1]
        

        xl,yl = outputs[0][0]*initial_w,outputs[1][0]*initial_h
        xr,yr = outputs[2][0]*initial_w,outputs[3][0]*initial_h
        
        # make box for left eye 
        xlmin = int(xl-10)
        ylmin = int(yl-10)
        xlmax = int(xl+10)
        ylmax = int(yl+10)
        left_eye =  image[ylmin:ylmax, xlmin:xlmax]
        
        # make box for right eye 
        xrmin = int(xr-20)
        yrmin = int(yr-20)
        xrmax = int(xr+20)
        yrmax = int(yr+20)
        right_eye =  image[yrmin:yrmax, xrmin:xrmax]
        eye_coords = [[xlmin,ylmin,xlmax,ylmax],[xrmin,yrmin,xrmax,yrmax]]

        return  left_eye, right_eye, eye_coords


    def preprocess_outputs(self, outputs, image):
        '''
        TODO: This method needs to be completed by you
        '''
            
        output=[]
        for out in outputs[0][0]:  
            if out[2] > self.threshold:
                output.append(out)
                
            
        return output



    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        n,c,h,w = self.input_shape
        p_frame = cv2.resize(image, (w,h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(n,c,h,w)
        return p_frame


