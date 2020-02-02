# AWSDeeplens_Classification
This is an excercise in emualating the performance of a large, costly general purpose model with a system of small, specialized models that can be deployed on EDGE devices. 
A single shot object detection model is used to identify potential items if interest and isolate them, and then an image of only the isolated object is passed to a classification model.   The system is deployed on an AWS Deeplens camera.  I used squirrels and birds in my backyard for convenience, and because it avoids any issues of privacy I would face if I aim the camera at sidewalks or streets.

I use the example ssd object detection model provided by AWS, as I see no benefit to training my own model.  Instead, the performance of the model is improved by limiting the field of view to a small block in an area of interest.  That reduces the need to scale down the input image to match the training image size and/or it allows the model to be trained on a smaller size image set, reducing training cost.  The full camera range can easily be covered by implementing a digital scan, capturing small segments across the range of the camera and passing each to the object detection model in sequence.  However, I did not implement this after the first few experiments becuase in the case of my yard, most interesting action happens within a small range anyway. 
 
I used the ssd algorithm to capture images directly from the camera, then combined them with a selection of images from the Caltech UCSD CUB_200_2011 bird image dataset (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) to generate the classification model.

## File Description
The system is implemented in the form of two AWS Lambda routines in two files:
- object_detect_save_lambda.py is an AWS  Lambda routine that runs locally on the camera.  This isolates the area of interest and passes it to the object detection routine.  If any objects of interest are detected, the region containing that object is cropped, scaled, and saved to an S3 Bucket.
- File_put_Lambda.py is a second Lamda routine that exists on AWS itself and triggers each time an object is placed in the S3 bucket.  This opens the file from S3, passes it through the classification model, and logs the result to a csv file.

TODO:  The implementation through S3 is useful for evaluating what the model is seeing and how classification works, but it is not fully EDGE deployed.  Next is to work out Python version incompatibilities on the camera implementation, and move the classification model to an Intel Movidius Neural Compute Stick directly on the camera.

Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001.
