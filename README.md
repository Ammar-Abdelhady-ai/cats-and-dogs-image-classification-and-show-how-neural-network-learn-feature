# cats-and-dogs-image-classification-and-show-how-neural-network-learn-feature
cats and dogs image classification and show how neural network learn feature from image

### Visualizing what convnets learn and,
### Mack medel to classification Cats and Dogs image

It is often said that deep learning models are "black boxes", learning representations that are difficult to extract and present in a human-readable form.

While this is partially true for certain types of deep learning models, it is definitely not true for convnets. The representations learned by convnets are highly amenable to visualization, in large part because they are representations of visual concepts. Since 2013, a wide array of techniques have been developed for visualizing and interpreting these representations. We won't survey all of them, but we will cover three of the most accessible and useful ones:

Visualizing intermediate convnet outputs ("intermediate activations"). This is useful to understand how successive convnet layers transform their input, and to get a first idea of the meaning of individual convnet filters.
Visualizing convnets filters. This is useful to understand precisely what visual pattern or concept each filter in a convnet is receptive to.
Visualizing heatmaps of class activation in an image. This is useful to understand which part of an image where identified as belonging to a given class, and thus allows to localize objects in images.
For the first method -- activation visualization -- we will use the small convnet that we trained from scratch on the cat vs. dog classification problem two sections ago. 
For the next two methods, we will use the InceptionV3 model

### Key convolution property


It's critical to understand what does a feature map represent with respect to the input image.

Baiscally a 4x4 feature map tensor, can be resized and layed over the input, in that case it's clear it segments the image into 4 areas. Those areas are essential in almost all CV tasks. In SS, those are superpixels. In Object detection, they help search for the boxes. In classification, they also search for the class to look for in different areas efficiently.

### Visualizing intermediate activations

Visualizing intermediate activations consists in displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input (the output of a layer is often called its "activation", the output of the activation function). This gives a view into how an input is decomposed unto the different filters learned by the network.

These feature maps we want to visualize have 3 dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel, as a 2D image.

#### ImageDataGenerator is used for getting the input of the original data and further, 
#### it makes the transformation of this data on a random basis and gives the output resultant containing 
#### only the data that is newly transformed.
#### and mack Data augmentation 

#### Inception v3 is an image recognition model that has been shown to attain greater than 78.1% accuracy 
#### on the ImageNet dataset. The model is the culmination of 
#### many ideas developed by multiple researchers over the years

#### trainable to False moves all the layer's weights from trainable to non-trainable. 
#### This is called "freezing" the layer: the state of a frozen layer
#### won't be updated during training (either when training with fit() or 
#### when training with any custom loop that relies on 
#### trainable_weights to apply gradient updates)

#### use callback function to :
#### save model and avoid over fitting and save time in early stop.

#### Keras functional model to get the output of intermediate layer In order to extract the feature maps we want to look at, we will create a Keras model that takes batches of images as input, and outputs the activations of all convolution and pooling layers.

#### To do this, we will use the Keras class Model. A Model is instantiated using two arguments: an input tensor (or list of input tensors), and an output tensor (or list of output tensors).

#### The resulting class is a Keras model, just like the Sequential models that you are familiar with, mapping the specified inputs to the specified outputs. What sets the Model class apart is that it allows for models with ___multiple outputs__, unlike Sequential.

#### When fed an image input, this model returns the values of the layer activations in the original model. 

#### This is the first time you encounter a multi-output model: until now the models you have seen only had exactly one input and one output. 

#### In the general case, a model could have any number of inputs and outputs. 
#### This one has one input and 8 outputs, one output per layer activation.

#### This channel appears to encode a diagonal edge detector. 
#### Let's try the 3th channel -- but note that your own channels may vary, 
#### since the specific filters learned by convolution layers are not deterministic

![image](https://user-images.githubusercontent.com/76500493/230624809-6fc7f211-59be-4197-8153-71d4fe6ee17e.png)

#### This one looks like a "bright green dot" detector, 
#### useful to encode cat eyes. At this point, 
#### let's go and plot a complete visualization of all the activations in the network.
#### We'll extract and plot every channel in each of our 8 activation maps, 
#### and we will stack the results in one big image tensor, 
#### with channels stacked side by side.

### Visualizing heatmaps of class activation


#### We will introduce one more visualization technique, one that is useful for understanding which parts of a given image led a convnet to its final classification decision. This is helpful for "debugging" the decision process of a convnet, in particular in case of a classification mistake. It also allows you to locate specific objects in an image.

#### This general category of techniques is called "Class Activation Map" (CAM) visualization, and consists in producing heatmaps of "class activation" over input images. A "class activation" heatmap is a 2D grid of scores associated with an specific output class, computed for every location in any input image, indicating how important each location is with respect to the class considered. For instance, given a image fed into one of our "cat vs. dog" convnet, Class Activation Map visualization allows us to generate a heatmap for the class "cat", indicating how cat-like different parts of the image are, and likewise for the class "dog", indicating how dog-like differents parts of the image are

###  Grad-CAM
The specific implementation we will use is the one described in Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization.

It is very simple: it consists in taking the output feature map of a convolution layer given an input image, and weighing every channel in that feature map by the gradient of the class with respect to the channel.

Intuitively, one way to understand this trick is that we are weighting a spatial map of "how intensely the input image activates different channels" by "how important each channel is with regard to the class", resulting in a spatial map of "how intensely the input image activates the class".

#### Let's consider the following image of Cat, 
#### possible a mother and its cub, 
#### strolling in the savanna (under a Creative Commons license):

![image](https://user-images.githubusercontent.com/76500493/230624995-24fd490d-7703-4682-a752-e629273c0f5b.png)

#### The conv2d_259 output is batch_sizexx8x8x320 = 1x8x8x320

#### Now, we can merge the block5_conv3 outputs (batch_sizex8x8x320),
#### by weighting and sum over the 320 dimension with the grads to get one heatmap 1x8x8.

For visualization purpose, we will also normalize the heatmap between 0 and 1:

![image](https://user-images.githubusercontent.com/76500493/230625078-c145036a-82a8-4961-bb30-c9c08e2c515e.png)

Finally, we will use OpenCV to generate an image that superimposes the original image with the heatmap we just obtained.

The resulting heatmap is 8x8, with each block has a receptive field over the input 224x224 image.

You can think of it as a drop of ink spilled over to cover bigger area.

So each block of the 8x8 corresponds to an area of size = 224/8x224/8 in the input image.

![image](https://user-images.githubusercontent.com/76500493/230625152-90aca9dc-547e-4fc3-83dc-6a378c538750.png)

#### This visualisation technique answers two important questions:

#####  - Why did the network think this image contained an cat?
##### - Where is the cat located in the picture?


#### In particular, it is interesting to note that the ears of cat cub are strongly activated: this is probably how the network can tell the difference between cats type(Egyptian_cat, tiger_cat, )


