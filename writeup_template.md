# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals /
steps of this project are the following:
* Use the simulator to collect data of
good driving behavior
* Build, a convolution neural network in Keras that
predicts steering angles from images
* Train and validate the model with a
training and validation set
* Test that the model successfully drives around
track one without leaving the road
* Summarize the results with a written report
[//]: # (Image References)

## Rubric Points
### Here I will consider the [rubric
points](https://review.udacity.com/#!/rubrics/432/view) individually and
describe how I addressed each point in my implementation.  

---
### Files
Submitted & Code Quality

#### 1. Submission includes all required files and can
be used to run the simulator in autonomous mode

My project includes the
following files:
* model.py containing the script to create and train the
model
* drive.py for driving the car in autonomous mode
* model.h5 containing a
trained convolution neural network 
* writeup_template.md summarizing the results
#### 2. Submission includes functional code
Using the Udacity provided simulator
and my drive.py file, the car can be driven autonomously around the track by
executing
```python
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model.py file contains
the code for training and saving the convolution neural network. The file shows
the pipeline I used for training and validating the model, and it contains
comments to explain how the code works.

### Model Architecture and Training
Strategy

#### 1. An appropriate model architecture has been employed

My model
consists of a convolution neural network with 3x3 and 5x5 filter sizes and
depths between 24 and 128

The model includes RELU layers to introduce
nonlinearity , and the data is normalized in the model using a Keras lambda
layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains
dropout layers in order to reduce overfitting . 

The model was trained and
validated on different data sets to ensure that the model was not overfitting .
The model was tested by running it through the simulator and ensuring that the
vehicle could stay on the track.

#### 3. Model parameter tuning

The model used
an adam optimizer, so the learning rate was not tuned manually .

#### 4.
Appropriate training data

Training data was chosen to keep the vehicle driving
on the road. I used a combination of center lane driving.

### Model
Architecture and Training Strategy

#### 1. Solution Design Approach
我首先使用了Tesla的相关架构，并添加了relu和dropout层，防止过度拟合

In order to gauge how well the model
was working, I split my image and steering angle data into a training and
validation set. I found that my first model had a low mean squared error on the
training set but a high mean squared error on the validation set. This implied
that the model was overfitting. 

To combat the overfitting, added relu and
dropout


The final step was to run the simulator to see how well the car was
driving around track one. There were a few spots where the vehicle fell off the
track... to improve the driving behavior in these cases, 我增加了训练数据集

At the end
of the process, the vehicle is able to drive autonomously around the track
without leaving the road.

#### 2. Final Model Architecture

The final model
architecture (in model.py) consisted of a convolution neural network...
Here is a visualization of the layers (note: visualizing the architecture is
optional according to the project rubric)


| Layer         		|     Description	        					        |
|:---------------------:|:-----------------------------------------------------:|
| Input         		| 80x320x3 RGB image   							        | 
| Convolution1  5x5     | 2x2 stride, SAME padding, outputs 40x160x24           |
| RELU					|                                                       |
| Convolution2  5X5     | 2x2 stride, SAME padding, outputs 20x80x36            |
| RELU					|                                                       |
| Convolution3  5X5     | 2x2 stride, SAME padding, outputs 10x40x48            |
| RELU					|                                                       |
| Convolution4  3X3     | 1x1 stride, VALID padding, outputs 8x38x64            |
| RELU					|                                                       |
| Convolution5  3X3     | 1x1 stride, VALID padding, outputs 6x36x64            |
| RELU					|                                                       |
| Flatten       		| outputs 13824											|
| Dropout				|												        |
| Dense     	      	| outputs 180											|
| RELU					|                                                       |
| Dense     	      	| outputs 80											|
| Dense     	      	| outputs 10											|
| RELU					|                                                       |
| Dense     	      	| outputs 1											    |

#### 3. Creation
of the Training Set & Training Process
为了获得良好的驾驶行为，我是使用了左侧和右侧的摄像头图片来补偿数据，并对中心图片进行了左右镜像，同样，将方向盘角度也翻转（angle*-1.0）.
在预处理这些数据后，我使用此训练数据来训练模型。验证集有助于确定模型是否超出或不合适。理想的时期数量是6，我使用了adam优化器，因此无需手动训练学习率。