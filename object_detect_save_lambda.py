
#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
""" Amazon object detection lambda sample, modified to pass a cropped image to object detection and then pass any detected object to a second model"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import boto3
import logging
import urllib
import zipfile

IMAGE_STORE_BUCKET = os.environ['IMAGE_STORE_BUCKET']

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (640, 640)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 640, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'
        output_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
                      7 : 'car', 8 : 'cat', 9 : 'chair', 10 : 'cow', 11 : 'dining table',
                      12 : 'dog', 13 : 'horse', 14 : 'motorbike', 15 : 'person',
                      16 : 'pottedplant', 17 : 'sheep', 18 : 'sofa', 19 : 'train',
                      20 : 'tvmonitor'}
        
        #set save path and image index to save images that have objects detected
        full_image_index = 0
        cropped_image_index = 0
        s3 = boto3.resource('s3')
        store_bucket = s3.Bucket(IMAGE_STORE_BUCKET)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
       
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
        # Load the model onto the GPU.
        print('Loading object detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        print('Object detection model loaded')
        # Set the threshold for detection
        detection_threshold = 0.20
        # The height and width of the training set images
        input_height = 300
        input_width = 300
        clip_width =640
        clip_height = 640
        clip_left_offset = 1266
        clip_top_offset = 900
        image_pad = 20
        image_keep_list = [3, 8, 12]
        
        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            
            # clip the frame to the clip width and height set by constant.  Clip width and
            # height are a compromise of detail vs. image window, as the larger an image 
            # kept after cropping, the more detail will be lost scaling it down to the 
            # trained image size (300x300)
            frame = frame[clip_top_offset:clip_top_offset + clip_height, 
                            clip_left_offset:clip_left_offset + clip_width].copy()
            
            # copy raw (cropped) frame so images passed to second model will not have
            #the rectangle used for the visual display
            clean_frame = frame.copy()
            #print(json.dumps(frame.shape))
            
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API.
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0]) / float(input_height)
            xscale = float(frame.shape[1]) / float(input_width)
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}

            # Get the detected objects and probabilities
            for obj in parsed_inference_results[model_type]:
                if obj['prob'] > detection_threshold and obj['label'] in image_keep_list:
                    # Add bounding boxes to full resolution frame
                    xmin = int(xscale * obj['xmin'])
                    ymin = int(yscale * obj['ymin'])
                    xmax = int(xscale * obj['xmax'])
                    ymax = int(yscale * obj['ymax'])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 5).
                    text_offset = 15
                    cv2.putText(frame, "{}: {:.2f}%".format(output_map[obj['label']],
                                                               obj['prob'] * 100),
                                (xmin, ymin-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 20), 2)
                    # Store label and probability to send to cloud
                    cloud_output[output_map[obj['label']]] = obj['prob']
                    
                    #labels I don't want gobs of pictures of. This is redundant to the 
                    #image_keep_list but I used both methods in different experiments
                    #and I expect to continue experiments in the future so for future 
                    # convenience it stays
                    excluded_labels = [4, 9, 16] 
                    
                    #limits of cropped images
                    cxmin = max(0, xmin - image_pad)
                    cxmax = min(xmax + image_pad, clip_width)
                    cymin = max(0, ymin - image_pad)
                    cymax = min(ymax + image_pad, clip_height)

                    #save the section of frame where the rectangle is
                    if (cxmax > cxmin) and (cymax > cymin) and (obj['label'] not in excluded_labels):
                        logger.info('crop to box: xmin={}, xmax={}, ymin={}, ymax={}'.format(xmin, xmax, ymin, ymax))
                        cropped_image = clean_frame[cymin:cymax, cxmin:cxmax].copy()
                        #print(json.dumps(cropped_image.shape))
                        cropped_image = cv2.resize(cropped_image, (224, 224))
                        #print(json.dumps(cropped_image.shape))
                        ret, jpeg = cv2.imencode('.jpg', cropped_image)
                        if ret:
                            print('storing cropped image')
                            filename = "cropped_image_{}.jpg".format(cropped_image_index)
                            store_bucket.put_object(Key = filename, Body = jpeg.tostring())
                            print('uploaded file {}'.format(filename))
                            cropped_image_index += 1
                        else:
                            print('jpeg encoding failed')

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

infinite_infer_run()
