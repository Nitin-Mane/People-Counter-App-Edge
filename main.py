##################################################################################
#
#                                 People Counter App Script
#
##################################################################################
'''
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
#
##################################################################################
#
# Running the code file from the Project Workspace, Remote system and Edge Device. 

'''
Execution command at the command prompt after initalizing mqtt and Mosca server with ffserver - 

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

'''
###################################################################################

# Load the library 
import os # Operating system commands for directory path and reading files
import sys # System opertation library for the passing specified arguments
import time # Time initialing libary 
import socket # transliteration call interface for two network.
import json # Reading json file 
import cv2 # Computer vision library for image processing

# Mqtt library 
import logging as log # Log data for the storing 
import paho.mqtt.client as mqtt # MQTT API libary recall

# Model load libary 
from argparse import ArgumentParser # Passing the command parser input and output
from inference import Network # loading model and IR 

#other library 
#import smtplib
#import time
#from datetime import datetime
#from email.mime.image import MIMEImage
#from email.mime.multipart import MIMEMultipart

# MQTT server environment variables
'''
This code generate a local host server address for the app to run in the backend. 

[doc](http://mosquitto.org/man/mqtt-7.html)

Hostname: System name 
IPaddress: host remote ip address
MQTT host: remote available address
MQTT port: use port TCP/ UDP selection
MQTT keep interval: Rechecking instance running active or inactive. 

'''
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def ssd_out(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count

def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))

def build_argparser():
    """
    Parse command line arguments.
    Model - intel/person-detection-retail-0013/FP16/<model path>
    input - resources/<video_file> 
    cpu_extension - libcpu_extension_sse4.so
    device - CPU, GPU, FPGA or MYRIA
    Probability threshold - 0.5 or 0.6 (better performance)
    
    :return: command line arguments
    output: video frame  
    """
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    '''
    Paho MQTT Connect function 
    
    the remote MQTT request generate a client connect port to send data packet.
    
    
    Input: 
            MQTT_HOST : local host available ip address
            MQTT_PORT : local host port for communication protocol 
            MQQT_KEEPALIVE_INTERVAL : refresh mode
            
    Output: 
           Clint IP Address : local host ip address 127.0.0.1/3001 or 0.0.0.0/3001
    '''
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Flag for the input image
    single_image_mode = False # image mode to the set for loop
    
    # person detection parameter 
    current_request_id = 0 # real-time request id for counter
    previous_counts = 0 # last time counts of the frame 
    sum_count = 0 # total counts of the person in the frame set zero for later use
    up_timer = 0 # start timer mode set 0 
    
    # Initialise the class
    infer_network = Network() # setting up the class request
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold # recall probalility threshold from command
    
    # Initialize the Inference Engine
    infer_network = Network() # load the model in the IR network 

    # Load the network model into the IE network
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          current_request_id, 
                                          args.cpu_extension)[1]
    
    
    # Checks for live feed
    '''
    the live feed is used for the webcam or real-time input image or video
    
    Video Input technique - Opencv library
    [doc](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html)
    
    CAM: Camera mode 
    
    Image format: only jpg and bmp  
    
    output: frames
    '''
    if args.input == 'CAM':
        #setting up the camera mode 0 for intial stage
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    
    # Checks for video file from the input command
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    # turning on the the video capture command using opencv
    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        # input data from args for the processing
        cap.open(args.input)

    # If input data not matched to the video format or given any other data format. 
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source") #debug 
    
    #Setting the video processing technique
    '''
    global input is used for using variables in other function
    intial_w : width of the frame 
    inital_h : height of the frame 
    pro_threshold : input threshold for the better performance 
    
    
    '''
    # Global variables 
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    # calculating the the width and height of the frame.  
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # conditions for the captured frames for pre-processing
    while cap.isOpened():
        
        #Reading the next frame from the video or camera
        flag, frame = cap.read() 
        # flag is important for the breaking the loop of the function
        if not flag:
            break
        key_pressed = cv2.waitKey(60) # if pressed esc then the script will break
        
        # Pre-process the frame
        image = cv2.resize(frame, (w, h)) # getting the frame width and height and resizing it
        
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1)) # transpose for converting 
        image = image.reshape((n, c, h, w)) # reshaping the frames
        
        # Inference start time 
        inf_start = time.time() # Starting timer 
        
        # Perform inference on the frame
        infer_network.exec_net(current_request_id, image) # input frame

        if infer_network.wait(current_request_id) == 0:
            
            det_time = time.time() - inf_start
            
            result = infer_network.get_output(current_request_id)
            #if args.perf_counts:
            perf_count = infer_network.performance_counter(current_request_id)
            #performance_counts(perf_count)

            frame, current_count = ssd_out(frame, result)
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            
            count_person = "Person count : {:.1f}"\
                               .format(current_count)
            
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            cv2.putText(frame, count_person, (15, 35),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            if current_count == 1:
                room_state = "busy"
                
            else:
                room_state = "empty"
                
            room_status = "Registration  Room Status : {}"\
                                 .format(room_state)
            
            cv2.putText(frame, room_status, (15, 55),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            # When new person come in the video frame
            if current_count > previous_counts:
                up_timer = time.time()
                sum_count = sum_count + current_count - previous_counts
                client.publish("person", json.dumps({"total": sum_count}))

            # Person duration time in the video frame is calculated
            if current_count < previous_counts:
                duration = int(time.time() - up_timer)
                
                # Publishing the messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            previous_counts = current_count
            
            ## Mail information
            #toaddr = '< Sender_mail_id>'
            #me = '<my_mail_id'
            #Subject='security alert'

            ### current_count, sum_count and duration to the MQTT server ###
            #if current_count > 1:
            #    print("Alert! multiple person are detected ")
            #    #camera warm-up time
            #    time.sleep(2)
            #    
            #    time.sleep(10)
                
            #    subject='Security alert!!'
            #    msg = MIMEMultipart()
                
            #    msg['Subject'] = subject
            #    msg['From'] = me
            #    msg['To'] = toaddr
                
            #    fp= open(frame,'rb')
            #    img = MIMEImage(fp.read())
            #    fp.close()
            #    msg.attach(img)

            #    server = smtplib.SMTP('smtp.gmail.com',587)
            #    server.starttls()
            #    server.login(user = '<User_mail_id>',password='<PASSWORD>')
            #    server.send_message(msg)
            
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            

def main():
    """
    Load the network and parse the output.
    
    args: 
          Setting up the MQTT protocol

    :return: Communication server ip address with port
    """
    
    
    # Grab command line args
    # Input commands for the passing arguments 
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt() # Starting the MQTT Server
    
    # Perform inference on the input stream
    infer_on_stream(args, client) # Inference with the input data with the client server


if __name__ == '__main__':
    #main file for execution
    main()
