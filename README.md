# Computer Pointer Controller

In this project, I used a gaze detection model to control the mouse pointer of my computer.  I used the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project demonstrates how to run multiple models in the same machine and coordinate the flow of data between those models.

###How it works

You will be using the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The gaze estimation model requires three inputs:

    The head pose
    The left eye image
    The right eye image.
    
To get these inputs, you will have to use three other OpenVino models:

    Face Detection
    Head Pose Estimation
    Facial Landmarks Detection

## Project Set Up and Installation

  1) Install Intel® Distribution of OpenVINO™ toolkit 
      https://docs.openvinotoolkit.org/latest/
      Note:  Please make sure to install all dependencies listed in the OpenVino documentation - ie CMake, Python & Microsoft Visual Studio

  2) Inititialize OpenVino Environment
      For Windows using Command Prompt:

        cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
        setupvars.bat

  3) Unzip My Project files from computer-pointer-controller.zip and extract in project directory: 
      c:\Users\cwilkens\Udacity\project3\computer-pointer-controller 

          Directory Structure:
          README.md
          requirements.txt
          +---bin
          |       demo.avi
          +---src
          |   |   face_detection.py
          |   |   facial_landmarks_detection.py
          |   |   gaze_estimation.py
          |   |   head_pose_estimation.py
          |   |   input_feeder.py
          |   |   log.log
          |   |   main.py
          |   |   model.py
          |   |   mouse_controller.py
          |   |   stats.txt
          |   |   utils.py

  4) Create Virtual Environment from command prompt:
      virtualenv project3

  5) Install Dependicies - pip install requirements.txt
      image==1.5.27
      ipdb==0.12.3
      ipython==7.10.2
      numpy==1.17.4
      Pillow==6.2.1
      requests==2.22.0
      virtualenv==16.7.9

  6) Download Models
     For Windows, using command prompt as admin create models directory structure:
     
          cd C:\Program Files (x86)\IntelSWTools\openvino\
          mkdir models\intel
          cd C:\Program Files (x86)\IntelSWTools\openvino\models\intel

      ###face-detection-adas-binary-0001
          python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "face-detection-adas-binary-0001"

      ###landmarks-regression-retail-0009
          python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "landmarks-regression-retail-0009"

      ###head-pose-estimation-adas-0001
        python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "head-pose-estimation-adas-0001"

      ###gaze-estimation-adas-0002
        python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "gaze-estimation-adas-0002"

## Demo
From the main project directory:  (project3) c:\Users\cwilkens\Udacity\project3\computer-pointer-controller
  Change directory to \src directory
    (project3) c:\Users\cwilkens\Udacity\project3\computer-pointer-controller\src>

  Command to Run Video File with no Flagging:

      python main.py -fm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -gm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i c:\Users\cwilkens\Udacity\project3\computer-pointer-controller\bin\demo.avi -d CPU

  Command to Run Camera with no Flagging:
  
       python main.py -fm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -gm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i CAM -d CPU


## Documentation
  To see full list of run options type python main.py -h

    (project3) c:\Users\cwilkens\Udacity\project3\computer-pointer-controller\src>python main.py -h
        usage: main.py [-h] -fm FACE_DETECTION_MODEL -lm LANDMARK_DETECTION_MODEL -hm
                       HEAD_POSE_ESTIMATION_MODEL -gm GAZE_ESTIMATION_MODEL
                       [-d DEVICE] -i INPUT [-pt PROB_THRESHOLD] [-l CPU_EXTENSION]
                       [-flag PREVIEW_FLAG [PREVIEW_FLAG ...]]

                        optional arguments:
                          -h, --help            show this help message and exit
                          -fm FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                                                Please enter the full path to the intel face detection
                                                model.
                          -lm LANDMARK_DETECTION_MODEL, --landmark_detection_model LANDMARK_DETECTION_MODEL
                                                Please enter the full path to the intel landmark
                                                detection model.
                          -hm HEAD_POSE_ESTIMATION_MODEL, --head_pose_estimation_model HEAD_POSE_ESTIMATION_MODEL
                                                Please enter the full path to the intel head pose
                                                estimation model.
                          -gm GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                                                Please enter the full path to the intel head pose
                                                estimation model.
                          -d DEVICE, --device DEVICE
                                                Please enter the target device, ie CPU, GPU, MYRIAD,
                                                or FPGA
                          -i INPUT, --input INPUT
                                                Please enter the full path to video or image file or
                                                enter CAM for camera
                          -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                                                Probability threshold for detections filtering(0.5 by
                                                default)
                          -ext CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                                               Absolute path to a shared library with the kernel
                                                implementation
                          -flag PREVIEW_FLAG [PREVIEW_FLAG ...], --preview_flag PREVIEW_FLAG [PREVIEW_FLAG ...]
                                                Please enter flag to view visualizations: ff fl fh fg
                                                (if multiple options - seperate by spaceff for Face
                                                Detection Model, fl for Facial Landmark Detection
                                                Modelfh for Head Pose Estimation Model & fg for Gaze
                                                Estimation Model.

## Benchmarks
  Tests were performed on Intel Core i7-7820HK CPU @ 2.9Ghz, 2901 Mhz,4 Core(s), 8 Logical Processors(s)

   ###FP32
   Command to Run:
   
          python main.py -fm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009" -hm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001" -gm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002"  -i c:\Users\cwilkens\Udacity\project3\computer-pointer-controller\bin\demo.avi -d CPU 
          
  FP32 Output (from log.log file):
  
        INFO:root:------------------------Program Started-------------------------------------
        INFO:root:Loading Models
        INFO:root:Face detection model loaded: time: 323.452 ms
        INFO:root:Facial landmarks detection model loaded: time: 114.708 ms
        INFO:root:Head pose estimation model loaded: time: 147.235 ms
        INFO:root:Gaze estimation model loaded: time: 324.035 ms
        INFO:root:Time to load all models: time: 911.493 ms
        INFO:root:All models are loaded successfully..
        INFO:root:Initialize Mouse
        INFO:root:Starting Inference on Video
        INFO:root:Counter 59 seconds
        INFO:root:Total Inference Time 100.4 seconds
        INFO:root:fps 0.5876494023904382 frame/second
        INFO:root:Video has completed
        INFO:root:---------------------------------Program has ended ----------------------------------------

  ###FP16
  Command to Run:
  
      python main.py -fm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -gm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i c:\Users\cwilkens\Udacity\project3\computer-pointer-controller\bin\demo.avi -d CPU 
 
 FP16 Output (from log.log file):  
 
          INFO:root:------------------------Program Started-------------------------------------
          INFO:root:Loading Models
          INFO:root:Face detection model loaded: time: 445.123 ms
          INFO:root:Facial landmarks detection model loaded: time: 181.196 ms
          INFO:root:Head pose estimation model loaded: time: 294.523 ms
          INFO:root:Gaze estimation model loaded: time: 132.251 ms
          INFO:root:Time to load all models: time: 1054.124 ms
          INFO:root:All models are loaded successfully..
          INFO:root:Initialize Mouse
          INFO:root:Starting Inference on Video
          INFO:root:Counter 59 seconds
          INFO:root:Total Inference Time 99.4 seconds
          INFO:root:fps 0.5935613682092555 frame/second
          INFO:root:Video has completed
          INFO:root:---------------------------------Program has ended ----------------------------------------

  ###INT8 
  Command to Run:
  
    python main.py -fm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009" -hm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\head-pose-estimation-adas-0001\FP32-INT8\head-pose-estimation-adas-0001" -gm "C:\Program Files (x86)\IntelSWTools\openvino\models\intel\gaze-estimation-adas-0002\FP32-INT8\gaze-estimation-adas-0002"  -i c:\Users\cwilkens\Udacity\project3\computer-pointer-controller\bin\demo.avi -d CPU 
    
  FP16 Output (from log.log file):   
  
      INFO:root:------------------------Program Started-------------------------------------
      INFO:root:Loading Models
      INFO:root:Face detection model loaded: time: 339.574 ms
      INFO:root:Facial landmarks detection model loaded: time: 246.454 ms
      INFO:root:Head pose estimation model loaded: time: 266.209 ms
      INFO:root:Gaze estimation model loaded: time: 170.644 ms
      INFO:root:Time to load all models: time: 1023.892 ms
      INFO:root:All models are loaded successfully..
      INFO:root:Initialize Mouse
      INFO:root:Starting Inference on Video
      INFO:root:Counter 59 seconds
      INFO:root:Total Inference Time 99.8 seconds
      INFO:root:fps 0.591182364729459 frame/second
      INFO:root:Video has completed
      INFO:root:---------------------------------Program has ended ----------------------------------------

##Load Times:
  face-detection-adas-binary-0001 (FP32-INT)

  Model	                        | FP16	     |  FP32	      |INT8		   
  ------------------------------- |----------|------------- |----------	   
  face-detection-adas-binary-0001 |     	   |323.452 ms    |  		    	
  landmarks-regression-retail-0009|181.196 ms|114.708 ms    |246.454 ms         		     
  head-pose-estimation-adas-0001	|294.523 ms|147.235 ms    |266.209 ms	     
  gaze-estimation-adas-0002     	|132.251 ms|324.035 ms	  |170.644 ms	     


##Total Load Times, Inference Times & FPS:	

  Model	|Total Load Times |   Inference	  |FPS	   
  ------|-----------------|-------------- |---------------------------------	 
  FP16	| 1054.124 ms 	  |  99.8 seconds |0.591182364729459 frame/second		    	
  FP32	| 911.493 ms      |  100.4 seconds|0.5876494023904382 frame/second         		     
  INT8	| 1023.892 ms     |  99.8 seconds |0.591182364729459 frame/second  



## Results

 FP32 had the fastest model loading times however FP16 & INT8 had better inference times and frames per second.  I was expecting FP16 to load the fastest due to less precision in the model FP32.  I also noticed that when loading the same model face-detection-adas-binary-0001 (FP32-INT) my loading times varied across the FP16, FP32 & INT8 program runs. Additional test runs on multiple hardware types may change these initial results.


## Stand Out Suggestions

These tests were performed on a CPU device, it would be interesting to run these models using a varitey of environments to validate performance of these models across variety of hardware types.    

### Edge Cases

Lighting played a big impact with the camera.  Due to poor lighting, if a face was not recognized the mouse kept going in the same direction.  


