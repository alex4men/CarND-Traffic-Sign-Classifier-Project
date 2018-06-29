# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/distribution.png "Distribution"
[image3]: ./examples/predictions.png "Predictions"
[image4]: ./test_images/keepRight.jpg "Keep Right"
[image5]: ./test_images/noEntry.jpg "No Entry"
[image6]: ./test_images/noVehicles.jpg "No Vehicles"
[image7]: ./test_images/yield.jpg "Yield"
[image8]: ./test_images/turnRightAhead.jpg "Turn Right Ahead"
[image9]: ./test_images/noWaiting.jpg "No waiting"
[image10]: ./examples/activations.png "Activations"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/alex4men/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The project uses [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Find more info in the link.

I used the python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

I used pandas to read traffic sign names from the signnames.csv file.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. All traffic sign classes contained in the dataset are represented below:

![alt text][image1]

Here are distributions of the classes in the datasets:

![alt text][image2]

As we can see, they are quite similar.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I just normalized the image data from the range 0..255 to the range -1.0..1.0 by using the formula x = (x - 127.5) / 127.5. So, the image data has almost zero mean and equal variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet-5 architecture with some small changes. The main changes are different input and output dimensions and dropout layers after the first two fully connected layers. My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| outputs 400        									|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|		keep probability 0.75							|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout					|		keep probability 0.75							|
| Fully connected	- Output	| outputs 43       									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with batch_size = 128, learning rate = 0.001, keep probability = 0.75 on both dropout layers. I trained the model for 10 epochs, which seems the most optimal, because futher the model starts to overfit.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.946
* test set accuracy of 0.93

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
> I used the LeNet-5 architecture, because handwritten digits recognition task is similar with this task and I haven't known other architectures so well for that moment.

* What were some problems with the initial architecture?
> The accuracy was about 84% on validation dataset or worse.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
> First, I tried to add a dropout layer after one of the fully connected layers. The accuracy on the validation dataset went up, but was far away from 0.93. I experimented with different keep probabilities, but still haven't reached the desired accuracy. Then I added dropout on the first two fully connected layers and made keep probability 0.75 and it helped!

* Which parameters were tuned? How were they adjusted and why?
> I tuned dropout keep probability in the range from 0.5 to 0.95. The most optimal is 0.75. Dropout is the most significant parameter in the model. Other parameters, like batch size, learn rate and number of epochs haven't changed much.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
> Convolution layer works good because of the weight sharing and translation invariance. The CNN learns itself what features to extract from the image. Dropout layer helps the model to generalize the knowledge (avoid overfitting). With dropout layers the model acts as an ensemble of many smaller models.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five traffic signs that I shot in Innopolis, Russia, which are similar to German traffic signs:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

And just for curiosity I took one sign which was not in the original dataset at all:

![alt text][image9]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep right      		| Keep right   									|
| No entry     			| No entry 										|
| No vehicles	      		| No vehicles					 				|
| Yield					| Yield											|
| Turn right ahead			| Turn right ahead      							|
| No waiting			| Turn left ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%. The 6th sign was predicted as "Turn left ahead" and the second association was "Keep right", which is more similar for me.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31st cell of the Ipython notebook.

![alt text][image3]

For the first image, the model is relatively sure that this is a keep right sign (probability of 0.9995), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .9995         			| Keep right   									|
| .0004     				| Turn left ahead 										|
| 3.63e-12					| End of no passing											|
| 7.33e-13	      			| Yield					 				|
| 6.74e-13				    | Go straight or right      							|

The "No vehicles" sign was predicted with the least certainity — only 91.21% for "No vehicles" and 8.76% for "Keep right".

The rest of the images with top 5 predictions can be seen on the image above. They all were predicted with almost 100% certainity, except for the "No waiting" sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image10]

The output of the first two convolutional layers of the model with "Keep right" sign as an input. We see on the first layer, it detects edges of straight lines and circles. On the second layer it detects diagonal line segments.
