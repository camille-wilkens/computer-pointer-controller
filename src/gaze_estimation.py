import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

import logging as log

import math


class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, model_name, device, extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.input_shape=None
        self.network = None
        self.model_name = None
      


        try:
            self.core = IECore()
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            log.error("Gaze Estimation could not be initialized, please check path",e)




      
    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        '''

        try:
            self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
            self.input = next(iter(self.network.inputs))
        except Exception as e:
            log.error('Gaze Estimation Model could not be loaded',e)
    

    def predict(self, left_eye_image,right_eye_image, head_pose_angle):
        '''
        TODO: This method needs to be completed by you
        '''
        p_left_eye_image  = self.preprocess_input(left_eye_image)
        p_right_eye_image  = self.preprocess_input(right_eye_image)
        
          
        self.network.start_async(request_id=0,inputs= {'left_eye_image': p_left_eye_image,'right_eye_image': p_right_eye_image,'head_pose_angles': head_pose_angle} )

        if self.network.requests[0].wait(-1) == 0:
            outputs = self.network.requests[0].outputs
            coords, gaze = self.preprocess_outputs(outputs,head_pose_angle)
        
        return coords, gaze   
    

    


    def preprocess_outputs(self, outputs, head_pose):
        '''
        TODO: This method needs to be completed by you
        '''
        #Reference: https://knowledge.udacity.com/questions/254779

        gaze_vector = outputs["gaze_vector"][0]
        roll = head_pose[2] 
        gaze_vector =  gaze_vector / np.linalg.norm(gaze_vector)
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)
        tmpX = gaze_vector[0] * cs + gaze_vector[1] * sn
        tmpY = gaze_vector[0] * sn + gaze_vector[1] * cs
        return (tmpX,tmpY)




    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        #Reference: https://knowledge.udacity.com/questions/188206
        

        try:
            image = cv2.resize(image,(60,60) )
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
        except Exception as e:
            log.error("Error While preprocessing Image in " + str(self.model_name) + str(e))
        return image