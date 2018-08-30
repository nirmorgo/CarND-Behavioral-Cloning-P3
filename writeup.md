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

[image1]: ./examples/center.jpg "center driving"
[image2]: ./examples/left_recovery.jpg "left recovery"
[image3]: ./examples/right_recovery.jpg "right recovery"
[image4]: ./examples/before_flip.jpg "before flip"
[image5]: ./examples/after_flip.jpg "after flip"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* data_handling.py contains functions for loading and augmenting the data
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

My model consists of 3 convolutional layers with 3X3 filters and 32 output channels and each convolution has a RELU activation and is followed by a 2X2 max pooling layer (code lines 14-19).

After the third convolution we flatten the output (code line 20) and use 2 fully connected layers with 128 and 64 outputs, with RELU activations, and than another fully connected layer with no activation, that we have the final model output (lines 21-25).

As a loss function for the model, i have used MSE since this is not a classification problem, but a regression problem (with a single output)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after the fully connected layers in order to reduce overfitting (model.py lines 22 and 24). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 28).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road doing turns and driving through all the different terain types.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutions in order to extract features from the road images, and use these features in order to calculate a single output for that determines the angle that the vehicle need to use. the net was trained as a regrresor

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because it is simple, and robust. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I add two dropout layers after the fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I collected additional data for training and used data augmentation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of 3 convolutional layers with 3X3 filters and 32 output channels and each convolution has a RELU activation and is followed by a 2X2 max pooling layer (code lines 14-19).

After the third convolution we flatten the output (code line 20) and use 2 fully connected layers with 128 and 64 outputs, with RELU activations, and than another fully connected layer with no activation, that we have the final model output (lines 21-25).

As a loss function for the model, i have used MSE since this is not a classification problem, but a regression problem (with a single output)

The input data is cropped from top and bottom in order to focus only on the interesting part of the image, and then it is normalized between 0-1 (lines 12-13)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself when needed:

![alt text][image2]
![alt text][image3]


To augment the data sat, I also flipped images and angles thinking that this would give more veriety to my dataset For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

I also drove in the opposite direction of the road in order to have more driving scenarios covered in my dataset.

To utilize the left and right cameras, i added them to the dataset as additional data points with corrected steering angle (i used 0.2 as a correction factor). This part of the code is described in data_handling.py

After the collection process, I had ~70k number of data points. I then preprocessed this data by cropping the 55 pixels from the top and 25 pixels from the bottom and normalizing the images between 0 to 1.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the validation set loss value stopped improving. I used an adam optimizer so that manually training the learning rate wasn't necessary.
