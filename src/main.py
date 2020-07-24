import numpy as np
import time 
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
from argparse import ArgumentParser
import sys
import logging as log
import math

from input_feeder import InputFeeder
from face_detection import FaceDetect
from facial_landmarks_detection import FacialLandmarksDetect
from head_pose_estimation import HeadPoseDetect
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

def argparser():

    parser = ArgumentParser()

    parser.add_argument("-fm", "--face_detection_model", required=True, type=str, 
                        help="Please enter the full path to the intel face detection model.")
                        
    parser.add_argument("-lm", "--landmark_detection_model", required=True, type=str, 
                        help="Please enter the full path to the intel landmark detection model.")
                        
    parser.add_argument("-hm", "--head_pose_estimation_model", required=True, type=str, 
    			help="Please enter the full path to the intel head pose estimation model.")
    
    parser.add_argument("-gm", "--gaze_estimation_model", required=True, type=str, 
    			help="Please enter the full path to the intel head pose estimation model.")
    			
    parser.add_argument("-d", "--device", required=False, default='CPU',
                        help="Please enter the target device, ie CPU, GPU, MYRIAD, or FPGA")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Please enter the full path to video or image file or enter CAM for camera")
                        
                        
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    parser.add_argument("-ext", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="Absolute path to a shared library with the kernel implementation")
                        

    parser.add_argument("-flag", "--preview_flag", required=False, nargs='+',
                        default=[],
                        help="Please enter flag to view visualizations: ff fl fh fg (if multiple options - seperate by space "
                             "ff for Face Detection Model, fl for Facial Landmark Detection Model"
                             "fh for Head Pose Estimation Model & fg for Gaze Estimation Model.")

    return parser


def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    #Code Reference - https://knowledge.udacity.com/questions/171017
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                   [0, 1, 0],
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame

def build_camera_matrix(center_of_face, focal_length):
    # Code Reference - https://knowledge.udacity.com/questions/171017
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix

def main():
    args = argparser().parse_args()
    
    log.basicConfig(filename='log.log',level=log.INFO)
    

    device=args.device
    threshold=args.prob_threshold

    extension = args.cpu_extension
    preview_flags = args.preview_flag
    
    input_file_path = args.input
    # Initialize Models
    log.info("------------------------Program Started-------------------------------------")
    face=FaceDetect(args.face_detection_model, args.device,  args.cpu_extension,args.prob_threshold)
    landmark=FacialLandmarksDetect(args.landmark_detection_model, args.device,  args.cpu_extension)
    head_pose=HeadPoseDetect(args.head_pose_estimation_model, args.device,  args.cpu_extension)
    gaze_estimation=GazeEstimation(args.gaze_estimation_model, args.device,  args.cpu_extension)

    # Load models
    log.info("Loading Models")
    start_time = time.time()
    face.load_model()
    log.info("Face detection model loaded: time: {:.3f} ms".format((time.time() - start_time) * 1000))
    
    landmark_start = time.time()
    landmark.load_model()
    log.info("Facial landmarks detection model loaded: time: {:.3f} ms".format((time.time() - landmark_start) * 1000))
    
    head_start = time.time()
    head_pose.load_model()
    log.info("Head pose estimation model loaded: time: {:.3f} ms".format((time.time() - head_start) * 1000))
    
    gaze_start = time.time()
    gaze_estimation.load_model()
    log.info("Gaze estimation model loaded: time: {:.3f} ms".format((time.time() - gaze_start) * 1000))
    
    load_total_time = time.time() - start_time
    log.info("Time to load all models: time: {:.3f} ms".format(load_total_time * 1000))
    log.info("All models are loaded successfully..")


    if input_file_path.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_file_path):
            log.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_file_path)

 
    log.info("Initialize Mouse")
    mouse = MouseController(precision='low', speed='fast')

  
    feeder.load_data()
    log.info("Starting Inference on Video")
    start_time = time.time()
    counter=0
    for ret, frame in feeder.next_batch():	
    	
        if not ret:
            break

        key = cv2.waitKey(60)
        counter= counter+1

        face_coords,face_image = face.predict(frame.copy())
        left_eye,right_eye,eye_coords = landmark.predict(face_image)
        hp_angles = head_pose.predict(face_image)
        gaze_coords  =  gaze_estimation.predict(left_eye, right_eye, hp_angles)
        # Settings from https://knowledge.udacity.com/questions/171017
        focal_length = 950.0
        scale = 50
        center_of_face = (face_image.shape[1] / 2, face_image.shape[0] / 2)
        
        yaw = hp_angles[0]
        pitch = hp_angles[1]
        roll = hp_angles[2]


        if len(preview_flags) != 0:

            if 'ff' in preview_flags and len(preview_flags)==1:
                preview_window = frame
            else:
                 preview_window = face_image.copy()
                
   
            if 'ff' in preview_flags  and len(preview_flags)==1:
                cv2.rectangle(frame, (face_coords[0], face_coords[1]),
                                  (face_coords[2], face_coords[3]), (0, 250, 0), 3)
                         
                    
            elif 'fl' in preview_flags and len(preview_flags)==1:
    
                cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                              (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                              (150, 0, 150))
                   
            elif 'fh' in preview_flags and len(preview_flags)==1:
                
                cv2.putText(preview_window,"Pose Angles: Yaw:{:.10f} | Pitch:{:.10f} |Roll:{:.10f}".format(hp_angles[0],hp_angles[1],
                                                                             hp_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 0, 0), 1   )
            elif 'fg' in preview_flags and len(preview_flags)==1:
                
                draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
    

            elif 'fg' in preview_flags and 'fh' in preview_flags and len(preview_flags)==2:
                #Gaze
                draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
                #Head Pose
                cv2.putText(preview_window,"Pose Angles: Yaw:{:.10f} | Pitch:{:.10f} |Roll:{:.10f}".format(hp_angles[0],hp_angles[1],
                                                                             hp_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .3, (255, 0, 0), 1   )
                
            elif 'ff' in preview_flags and 'fh' in preview_flags and len(preview_flags)==2:
                #face
                cv2.rectangle(frame, (face_coords[0], face_coords[1]),
                                  (face_coords[2], face_coords[3]), (0, 250, 0), 3)
                #Head Pose
                cv2.putText(preview_window,"Pose Angles: Yaw:{:.10f} | Pitch:{:.10f} |Roll:{:.10f}".format(hp_angles[0],hp_angles[1],
                                                                             hp_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .3, (255, 0, 0), 1   )
    
            elif 'ff' in preview_flags and 'fl' in preview_flags and len(preview_flags)==2:
                #face
                cv2.rectangle(frame, (face_coords[0], face_coords[1]),
                                  (face_coords[2], face_coords[3]), (0, 250, 0), 3)
                #eye
                cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                              (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                              (150, 0, 150))
                
            elif 'fh' in preview_flags and 'fl' in preview_flags and 'fg' in preview_flags and len(preview_flags)==3:
                #Head Pose
                cv2.putText(preview_window,"Pose Angles: Yaw:{:.10f} | Pitch:{:.10f} |Roll:{:.10f}".format(hp_angles[0],hp_angles[1],
                                                                             hp_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .3, (255, 0, 0), 1   )

                #eye
                cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                              (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                              (150, 0, 150))
                #Gaze
                draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)

                
                    
            elif 'ff' in preview_flags and 'fg' in preview_flags and len(preview_flags)==2:
                #face
                cv2.rectangle(frame, (face_coords[0], face_coords[1]),
                                  (face_coords[2], face_coords[3]), (0, 250, 0), 3)
                #gaze
                draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)

            elif 'fl' in preview_flags and 'fh' in preview_flags and len(preview_flags)==2:
                #eye
                cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                              (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                              (150, 0, 150))
                #head pose
                cv2.putText(preview_window,"Pose Angles: Yaw:{:.10f} | Pitch:{:.10f} |Roll:{:.10f}".format(hp_angles[0],hp_angles[1],
                                                                             hp_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .2, (255, 0, 0), 1   )

            elif 'fg' in preview_flags and 'fl' in preview_flags and len(preview_flags)==2:
                #gaze
                draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
                #eye
                cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                              (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                              (150, 0, 150))
                
            else: 
                #face
                cv2.rectangle(frame, (face_coords[0], face_coords[1]),
                                  (face_coords[2], face_coords[3]), (0, 250, 0), 3)
                #gaze
                draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
                #eye
                cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                              (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                              (150, 0, 150))
                #head pose
                cv2.putText(preview_window,"Pose Angles: Yaw:{:.10f} | Pitch:{:.10f} |Roll:{:.10f}".format(hp_angles[0],hp_angles[1],
                                                                             hp_angles[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .3, (255, 0, 0), 1   )


        if len(preview_flags) != 0:
            preview_image = np.hstack((cv2.resize(frame, (1500, 1500)), cv2.resize(preview_window, (1500, 1500))))
        else:
            preview_image = cv2.resize(frame, (1500, 1500))

        cv2.imshow('Visualization', preview_image)
        

        mouse.move(gaze_coords[0],gaze_coords[1])
       
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break  
    inference_time = round(time.time() - start_time, 1)
    fps = int(counter) / inference_time
    log.info("Counter {} seconds".format(counter))
    log.info("Total Inference Time {} seconds".format(inference_time))
    log.info("fps {} frame/second".format(fps))
    log.info("Video has completed")
    log.info("---------------------------------Program has ended ----------------------------------------")

    feeder.close()
    cv2.destroyAllWindows()
    
    
    



if __name__ == '__main__':
    main()