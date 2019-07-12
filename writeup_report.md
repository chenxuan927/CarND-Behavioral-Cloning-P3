# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_1.jpg
[image2]: ./examples/recovery_left_1.jpg
[image3]: ./examples/recovery_left_2.jpg
[image4]: ./examples/recovery_left_3.jpg
[image5]: ./examples/recovery_right_1.jpg
[image6]: ./examples/recovery_right_2.jpg
[image7]: ./examples/recovery_right_3.jpg
[image8]: ./examples/flip_1.jpg
[image9]: ./examples/flip_2.jpg
[image10]: ./examples/history.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 74-89) 

The model includes RELU layers to introduce nonlinearity (model.py line 77-81), and the data is normalized in the model using a Keras lambda layer (code line 75). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 84, line 86, and line 88). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 5, line 66, line 67). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was kind of "try and test" approach. 

My first step was to use a convolution neural network model similar to the nVidia model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model by adding some dropout layers between fully connected layers.

Then I tried to train the network with different epochs and check when the model was over or under fitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I re-train the model with more data collection as stated in section 3 Creation of the Training Set & Training Process. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-89) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)                    |          Output Shape      |     Param #   |                     
|:------------------------------:|:--------------------------:|:--------------:|
| lambda_1 (Lambda)              |    (None, 160, 320, 3)     |   0            |
| cropping2d_1 (Cropping2D)      |    (None, 65, 320, 3)      |   0            |
| conv2d_1 (Conv2D)              |    (None, 31, 158, 24)     |   1824         |   
| conv2d_2 (Conv2D)              |    (None, 14, 77, 36)      |   21636        |  
| conv2d_3 (Conv2D)              |    (None, 5, 37, 48)       |   43248        |  
| conv2d_4 (Conv2D)              |    (None, 3, 35, 64)       |   27712        |
| conv2d_5 (Conv2D)              |    (None, 1, 33, 64)       |   36928        |
| flatten_1 (Flatten)            |    (None, 2112)            |   0            |   
| dense_1 (Dense)                |    (None, 100)             |   211300       |
| dropout_1 (Dropout)            |    (None, 100)             |   0            |  
| dense_2 (Dense)                |    (None, 50)              |   5050         |
| dropout_2 (Dropout)            |    (None, 50)              |   0            |   
| dense_3 (Dense)                |    (None, 10)              |   510          |
| dropout_3 (Dropout)            |    (None, 10)              |   0            | 
| dense_4 (Dense)                |    (None, 1)               |   11           |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I record three laps on track one, see below for details. 
- lap 1: clockwise center lane driving
- lap 2: counter-clockwise center lane driving
- lap 3: clockwise recovery driving from the side

The two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to what to do when it’s off on the side of the road. 

These images show what a recovery looks like starting from left:

![alt text][image2]
![alt text][image3]
![alt text][image4]

These images show what a recovery looks like starting from right:

![alt text][image5]
![alt text][image6]
![alt text][image7]

To augment the data sat, I also flipped images and angles thinking that this would help not bias to either left or right turn. For example, here is an image that has then been flipped:

![alt text][image8]
![alt text][image9]

I use multiple cameras to capture images including a center camera, left camera and right camera. During training, the left and right camera images are fed to the model as if they were from the center camera. This way, the model can learn how to steer if the car drifts off to the left or the right. I used the left and right images with a steering correction of 0.2.

After the collection process, I had 8867 number of data points. I then preprocessed this data by converting the color from BGR to RGB because training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the following figure I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image10]
