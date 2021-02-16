# Image Classification
Existing knowledge of PyTorch and linear regression can be implemented for a new problem: *image classification*. The training dataset will be the MNIST Handwritten Digits Database. It consists of several grayscale images of handwritten digits (0 to 9) and has a label for each image indicating the digit it reperesents. 

Here are some of the images in the MNIST dataset:

![mnist](https://i.imgur.com/CAYnuo1.jpg)

First things first, install and import `torch` and `torchvision`. 

`torchvision` contains some utilities for working with image data. It also provides helped classes to download and import datasets like MNIST automatically. 
```
import torch
import torchvision
from torchvision.datasets import MNIST
```
Download training dataset
```
dataset = MNIST(root='data/', download=True)
```
Upon checking the size of the dataset `len(dataset)` it shown that there are 60,000 images. These 60,000 images will be used to train the model. 

There is also an additional 'test set' of 10,000 images which will be used for evaluating the model and reporting metrics. This set can be created using `MNIST` class and passing `train = False` to the constructor.
```
test_dataset = MNIST(root='data/', train=False)
```
Here's a sample element from the training dataset `dataset[0]`. It contains a 28 by 28 pixel image and a label. 
```
(<PIL.Image.Image image mode=L size=28x28 at 0x7F625B9FD710>, 5)
```
In order to view the image, the `matplotlib` library can be used (also used for plotting and graphing data science in Python)
```
import matplotlib.pyplot as plt
%matplotlib inline
```
Here is one of the images from the dataset:
```
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)
```
Label: 5

![mnistimg](https://miro.medium.com/max/704/1*GlZuwzO-dMXxyEHO65Ha_A.png)

These images are relatively small in size, and it is challengeing to recognize the digits even to the human eye. 

Since PyTorch doesn't know how to work with images, they must be converted into tensors. This can be done by specifying a transform while creating the dataset. The `torchvision.transforms`module contains many predefined transformation functions that can be applied to the images as they are loaded. The `ToTenser` will transform the images into PyTorch tensors. 
```
import torchvision.transforms as transforms
```
MNIST Dataset (images and labels)
```
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())
```
Convert into a 1x28x28 tensor:
```
img_tensor, label = dataset[0]
```
The first dimension tracks color channels. The second and third dimensions represents the height and width of the image in pixels, respectively. Since images in the MNIST dataset are grayscale, there's just one channel. Other datasets have images with color, in which case there are three channels: red, green, and blue (RGB).

Here are some sample values inside the tensor `print(img_tensor[0,10:15,10:15])`
```
tensor([[0.0039, 0.6039, 0.9922, 0.3529, 0.0000],
        [0.0000, 0.5451, 0.9922, 0.7451, 0.0078],
        [0.0000, 0.0431, 0.7451, 0.9922, 0.2745],
        [0.0000, 0.0000, 0.1373, 0.9451, 0.8824],
        [0.0000, 0.0000, 0.0000, 0.3176, 0.9412]])
```
The values range from 0 to 1, with 0 representing black, 1 white, and the values in between different shades of grey. `print(torch.max(img_tensor), torch.min(img_tensor))`
```
tensor(1.) tensor(0.)
```
Using `plt.imshow`, the tensor can be plotted as an image:
```
plt.imshow(img_tensor[0,10:15,10:15], cmap='gray');
```
![imshow](https://lh3.googleusercontent.com/fze6zrNtkU4KK79A4gOeOxP5PlW4CiKirOJjmj2Ezw3SDYnxhXHYM5-x8hjmxxbYp8Kl=s85)

### Training sets and Validation sets
When building real-world machine learning models, it is common to split the dataset into three parts:
1. **Training set** - used to train the model (ex: compute the loss and adjust the model's weights using gradient descent)
2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc), and pick the best version of the model
3. **Test set** - used to compare different models or approaches and report the model's final accuracy

In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images.


Since there's no predefined validation set, the 60,000 images must be split into training and validation datasets. 10,000 randomly chosen images can be set aside for validation by using the `random_spilt` method from PyTorch.
```
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000])
```
