# **Traffic Sign Recognition** 

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

[histogram]: ./visualizations/histogram.png "Histogram"
[sign_examples]: ./visualizations/sign_examples.png "Sign Examples"
[children_crossing]: ./new-images/children_crossing.png "Children Crossing"
[no_entry]: ./new-images/no_entry.png "No Entry"
[road_work]: ./new-images/road_work.png "Road Work"
[speed_limit_20]: ./new-images/speed_limit_20.png "Speed Limit 20"
[straight_or_right]: ./new-images/straight_or_right.png "Straight or Right"
[softmax0]: ./visualizations/softmax0.png "Softmax 0"
[softmax1]: ./visualizations/softmax1.png "Softmax 1"
[softmax2]: ./visualizations/softmax2.png "Softmax 2"
[softmax3]: ./visualizations/softmax3.png "Softmax 3"
[softmax4]: ./visualizations/softmax4.png "Softmax 4"
[network_state]: ./visualizations/network_state.png "Network State"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vkai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used vanilla Python and the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here are some exploratory visualizations of the data set. Below is a histogram of all the images available, showing the occurences of each class of sign. 

![Histogram][histogram]

It appears we have many more examples of certain signs compared to others. This is probably reflective of real life conditions (there are more 30 km/h roads than 20 km/h roads), but for the purposes of training our model, we may want to have plenty of examples of all signs.

Below is an example of each type of sign to help us understand what each sign is supposed to look like.

![Sign Examples][sign_examples]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

My original approach to preprocess the data involved a grayscaling step and normalization step. I chose to grayscale because I believed the color of the signs would not be as important as the shapes and contents of the signs. Grayscaling would also hopefully reduce the color noise from the areas surrounding the sign. I used the cv2 grayscaling technique used in Project 1.

As a last step, I normalized the image data to bring the data to zero mean and equal variance. I used the cv2 normalize method to normalize the data to between -1.0 and 1.0

As I worked through the model training, I found that the network achieved a higher validation accuracy with the original RGB images compared to grayscale images. I ended up removing the grayscaling step and simply normalizing the RGB values.

#### 2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		    | Description	        					                | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x3 RGB image   							            | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					        |												                        |
| Average pooling	      | 2x2 stride, outputs 14x14x6 				          |
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU          		    |         									                    |
| Average pooling				| 2x2 stride, outputs 5x5x16       							|
|	Flatten					      |	outputs 400 params											      |
|	Fully connected				|	outputs 120 params											      |
|	RELU          				|	                  											      |
|	Dropout          			|	0.5 keep prob                  								|
|	Fully connected				|	outputs 84 params											        |
|	RELU          				|	                  											      |
|	Dropout          			|	0.5 keep prob                  								|
|	Fully connected				|	outputs 43 params											        |


This model is based off the LeNet-5 lab completed in lecture. The layer sizes are the same as LeNet-5. The improvements came when changing the max pooling layers to average pooling and including dropout in the first and second fully connected layers. I used the same pooling strides for the average pooling, and I used a 0.5 keep probability for the dropout during model training.


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

I divided the training over 10 epochs with a batch size of 128. To train the model, I first shuffled the training set with each epoch to prevent developing any dependencies on the ordering of the dataset. I used the AdamOptimizer with a learning rate of 0.005, a departure from the LeNet-5 lab from lecture. I found that tuning the learning rate from 0.001 to 0.005 improved the accuracy notably. In order to gauge the degree of over/under fitting, I also monitored the training accuracy along with the validation accuracy during each training epoch. 

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eight and ninth cells of the Ipython notebook.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.961
* test set accuracy of 0.934
 
I started the approach to the solution with the LeNet-5 architecture from the LeNet lab completed in lecture. I found this architecture to be appropriate because it was designed for interpretting images. The convolution layers look at patches of the input image and can consider information from an area of pixels together, rather than simply looking at individual pixels independently. 

After starting with the LeNet-5 architecture, I tried tuning various hyperparameters to see how well the original model would perform. This was an iterative trial and error process as there were not many hyperparameters to adjust. I settled on increasing the learning rate to 0.005 from 0.001, finding that this gave the best performance increase.

Throughout this process I also monitored the training and validation accuracies to check for over/under fitting. In this way, I was able to find that the training accuracy was significantly higher than the validation accuracy, indicating that the model was over fitting to the training data. To remedy this, I took a couple tips from the lectures on reducing over fitting. I first swapped out the max pooling layers in the original LeNet architecture for average pooling layers. I then included two dropout layers after both inner fully connected layers. I found that all these adjustments significantly improved the validation accuracy in the end.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Children Crossing][children_crossing] ![No Entry][no_entry] ![Road Work][road_work] 
![Speed Limit 20][speed_limit_20] ![Straight or Right][straight_or_right]

I tried to choose images that did not have as many occurences in the training set to see if the model would still perform well on lesser seen images. I found that the Children Crossing sign I showed as an example above had a yellow border, whereas the Children Crossing signs from the web had a red border. Because I trained on the original RGB images, this could potentially affect the prediction of this new sign. 

I was also concerned for the performance of the triangular signs with a "busy" interior image, like the Children Crossing or Road Work. The small resolution of the images (32x32px) may not allow the neural network to pick up the intricacies of what was going on in the sign. The model could confuse these images for one another.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			          | Prediction	        					                                                    | 
|:-------------------:|:---------------------------------------------------------------------------------:| 
| Children Crossing   | General Caution - 0.302, Road Work - 0.177, Dangerous Curve to the Right - 0.120	| 
| No Entry     			  | No Entry - 1.000 										                                              |
| Road Work					  | Road Work - 1.000											                                            |
| 20 km/h	      		  | 120 km/h - 0.366, 30 km/h - 0.323, 20 km/h - 0.301					 				              |
| Straight or Right		| Straight or Right - 1.000      						                                      	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is poor compared to the test accuracy of 93.4%, however, we've only tested 5 additional images. Testing a new image of each sign class should hopefully yield a better representation of the model's performance.

The model was extremely sure of the No Entry, Road Work, and Straight or Right signs. As noted above, the model was unable to appropriately identify the Children Crossing sign, instead believing it to be the General Caution, Road Work, or Dangerous Curve signs. These are all triangular, red-bordered signs with some black interior content, as expected. 

The model seemed to be evenly split about it's decision on the 20 km/h sign, confusing it with the 120 km/h and 30 km/h signs equally. Perhaps the resolution of the text in the images is not enough for the model to train on the actual text. The new 20 km/h sign is also viewed at somewhat of an angle. Perhaps the model needs to train on more slightly distorted images from varying perspectives.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for visualizing the softmax probabilities on my final model is located in the 13th, 14th, and 15th cells of the Ipython notebook.

| Number | Image			          | Softmax Probabilities	(prediction, probability)					                                      | 
|:------:|:--------------------:|:---------------------------------------------------------------------------------------------:| 
| 0      | Children Crossing   | (18, 0.30197272), (25, 0.17671449), (20, 0.1202525), (19, 0.062454697), (27, 0.052557904)      |
| 1      | No Entry            | (17, 1.0), (14, 7.1997834e-18), (0, 3.6721973e-35), (36, 1.7810773e-35), (1, 0.0)              |
| 2      | Road Work           | (25, 1.0), (5, 2.1701042e-10), (19, 1.2962265e-10), (20, 1.2609405e-10), (31, 7.0323677e-11)   |
| 3      | 20 km/h             | (8, 0.36602476), (1, 0.32322204), (0, 0.3006396), (28, 0.0097880997), (4, 0.0001600767)        |
| 4      | Straight or Right   | (36, 0.9998017), (34, 0.00014898), (38, 3.113754e-05), (35, 9.110127e-06), (17, 4.542099e-06)  |

![Children Crossing][softmax0]
![No Entry][softmax1]
![Road Work][softmax2]
![20 km/h][softmax3]
![Straight or Right][softmax4]

### Visualize the Neural Network's State with Test Images

#### 1. Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images.

![Network State][network_state]

Shown above is the network state of the second convolutional layer for the No Entry sign. The No Entry sign is pretty unique amongst the other signs in the dataset, providing a nice visual of what excites the neural network about this image. Each of the feature maps clearly show the neural network finding the white bar across the center of the sign. FeatureMap 11 and FeatureMap 14 are also interesting in how they outline the rounded areas above and below the white bar.