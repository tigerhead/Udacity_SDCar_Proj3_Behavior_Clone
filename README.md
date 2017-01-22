## Udacity_SDCar_Proj3_Behavior_Clone
Udacity Self-Driven Car Nano-degree program Term2 Project 3 Behavior Cloning

# Project Introduction

In this project, a drive simulator is provided to collect images captured by carmeras installed on the test car and corresponding steering angles. A deep learning network is built and trained with collectig data. As a test of the performance of the network, the steering angle will be calcuated using trained network model and the simulator will use calculated steering angle to drive the test.

# Train data
Udacity project provided a standard training data collected on track 1. I collect extra driving data for 2 more laps on track one and manually created some recovery train data by drving car to the edge and recording images turning back to center of the road. Considering there are large amount of zero steering angle already, I only added no-zero steering data to provided dataset. 

Total number of images in the data set: 27018. 
Number of Center camera image: 9006
Number of Left camera image: 9006
Number of Right camera image: 9006
Number of Steering sample: 9006

# Deep learning neuron network architecure
I craeted two models. Both of them were built based on a paper published by Nvidia and turked it. Here is my final network model for model one(model.py):

Input image shape: 80 X 160 X 3

Layer 1:

Convolutional using 5X5X3 filler and 24 maps. The output shape should be 76x156x24.
Activation: Relu
Pooling: Max pooling with 2X2 kernel and stride = 1. The output shape should be 38x78x24.

Layer 2:

Convolutional using 5X5X1 filter and 36 maps. The output shape should be 34x74x36.
Activation: Relu.
Pooling: Max pooling with 2X2 kernel and stride = 1 The output shape should be 17x37x36.
Dropout: 0.5

Layer 3:

Convolutional using 5X5X1 filter and 48 maps. The output shape should be 13x33x48.
Activation: Relu.
Pooling: Max pooling with 2X2 kenel and stride = 1 The output shape should be 6x16x48.
Dropout: 0.5

Layer 4:

Convolutional using 3X3X1 filter and 64 maps. The output shape should be 4x14x64.
Activation: Relu.

Layer 5:

Convolutional using 3X3X1 filter and 128 maps. The output shape should be 2x12x128.
Activation: Relu.
Dropout: 0.5

Flatten: Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. Output is 3072

Layer 6: 

Fully Connected with 1164 neurons. Output is 1164.
Activation: Relu.

Layer 7: 

Fully Connected with 100 neurons. Output is 100.
Activation: Relu.

Layer 8:

Fully Connected with 50 neurons. Output is 50.

Activation: Relu.

Layer 9:

Fully Connected with 20 neurons. Output is 20.
Activation: Relu
Dropout: 0.5

Layer 10(readout layer): Fully Connected  1 neurons. Output is 1.

Model two has one more convolutional layer using 1X1X3 filter with 3 maps at top of model 1.

# Image preprocessing

Model 1 use minimum reprocess:
1. Randomly flip image vertically to simulate driving in reverse situation
2. Add fixed angle adjustment for left and right camera images. For left image, add 0.27 and right image, add -0.27
3. Normalize image between -0.5 and 0.5
4. Resize the original image size from 160 X 320 to 80 X 160

Model 2 use more augmentation techniques to create image in different situations:
1. Randomly adjust brightness of the image to simulate different light condition.
2. Randomly add shadow to simulate shade situation.
3. Use horizontal and vertical shift to simulate leveling situation
4. Corp the bottom part of the image to cut the dashboard
5. Resize image to 64 X 64

# Model Training

I spend lot of time on training. I played with learning rate and adjusted network architecture. I found out that loss error can only be used as a rogh reference. Lower error doesn't necessarily mean it work better in simulator test. I used checkpoint to save weighs for each epoch. And try each one of them to pick better weight. After many hours experiments, I finally got one weights for model 1 which drove track 1 smoothly. But model one failed easily in track 2, after reading post in forum and I finally found Vivek's blog about augmentation. I followed his blog, did the same augmentation in the blog(listed in Image preprocessing section), and worked well in both track 1 and 2. And I also tried to fine tuning model by load the save model and train it again further more using lower learning rate.

# Test Result

I recorded test result of model 1 on Track 1 and Model 2 on track 2. 

Track 1 video driven by model 1: https://youtu.be/GVTYq-stYwA

Tack 2 video driven by model 2: https://youtu.be/TvAb9CSlehM

#Some notes:

To execute model.py or model2.py, put all images under data/IMG directory. And for model1.py, it save weighss in models_1_01 directory and model2.py save weights in models_2_01 directory.

# Reference 
1. End to End Learning for Self-Driving Cars, Nvidia  http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
2. Vivek Yadav Blog https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.9y77pio6s
3. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
