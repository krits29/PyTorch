# Linear Regression
Linear regression is one of the foundational algorithms used in machine learning. It uses a linear approach to model the relationship between multiple variables.

![linear](https://backlog.com/wp-blog-app/uploads/2019/12/Nulab-Gradient-descent-for-linear-regression-using-Golang-Blog.png)

# Example Model
Walking through an example model which can predict the crop yields for apples and oranges by considering regional variables like the average temperature, rainfall, and humidity.

**Target Variables:** apples (ton), oranges (ton)

**Input Variables:** temperature (F), rainfall (mm), humidity (%)

**Training Data:**
Region | Temp (F) | Rainfall (mm) | Humidity (%) | Apples (ton) | Oranges (ton)
-------|----------|---------------|--------------|--------------|--------------
Kanto | 73 | 67 | 43 | 56 | 70
Johto | 91 | 88 | 64 | 81 | 101
Hoenn | 87 | 124 | 58 | 119 | 133
Sinnoh | 102 | 43 | 37 | 22 | 37
Unova | 69 | 96 | 70 | 103 | 119

Using the linear regression model, the target variable can be estimated.

The estimation is calculated by adding the input variables multipled by a **weight**. Additionally, there is a constant offset, which is called the **bias**.

```
yield_apple = (w11 * temp) + (w12 * rainfall) + (w13 * humidity) + b1
yield_orange = (w21 * temp) + (w22 * rainfall) + (w23 * humidity) + b2
```
This results in a **linear or planar function** of the input variables: temperature, rainfall, and humidity.
![planar](https://i.imgur.com/4DJ9f8X.png)

**Learning**

Using the training data, the model will learn what the set of weights `w11, w12, w13, w21, w22, w23` are. After that, the model can be fed new data about a different region and then make accurate predictions about the target variable given the specific new data.

**Training**

The model will be trained by slightly adjusting the weights many many times, each time getting it closer and closer to the given target value. This optimization technique is called *gradient descent*.


### Training the Data
The training data can be represented using two matrices: `inputs` and `targets` 

One row for the set of values. One column for each variable.

Input (temp, rainfall, humidity)
```
inputs = torch.tensor([[73, 67, 43], 
                       [91, 88, 64], 
                       [87, 134, 58], 
                       [102, 43, 37], 
                       [69, 96, 70]])
```
Targets (apples, oranges)
```
targets = torch.tensor([[56, 70], 
                        [81, 101], 
                        [119, 133], 
                        [22, 37], 
                        [103, 119]])
```
Additionally, the weights and biases can be respresented using matrices. They will start off as random values. Using `torch.randn` a random tensor can be created given a particular shape. In the case of the weights, there will be two rows (one for apple and one for oranges) and three columns for each input variable (temp, rainfall, humidity). In the case of the biases, there will just be two bias values, one for apples and one for oranges.
```
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
```
Coming back to the initial definition of the model that was previously stated:
```
yield_apple = (w11 * temp) + (w12 * rainfall) + (w13 * humidity) + b1
yield_orange = (w21 * temp) + (w22 * rainfall) + (w23 * humidity) + b2
```
Now with the matrices that were created for the data, weights, and biases, this definition can be applied using simple matrix multiplication and addition.
![model](https://i.imgur.com/WGXLFvA.png)

The weight matrix need to be *transposed* (flip flopped dimensions) in order to properly perform the matrix multiplication.

The model can be defined as:
```
def model(x):
  return x @ w.t() + b
```
The @ performs the matrix multiplication. The `.t()` will return a transpose. The `x` in the parameter is the input data into the model.

Test it out:

`preds = model(inputs)`

returns
```
tensor([[-43.9569, -21.1025],
        [-55.7975, -28.1628],
        [-70.6863,  11.5154],
        [-44.2982, -54.6685],
        [-51.9732, -10.9839]], grad_fn=<AddBackward0>)
```
However, when compared to the actual values of the target variable, there is a HUGE difference:
```
tensor([[ 56,  70],
        [ 81, 101],
        [119, 133],
        [ 22,  37],
        [103, 119]])
```
### Loss function
Before improving the model, it's important to evaluate how well the current model is performing. To compare the model's predictions and the actual targets, the following method is used:
- Calculate the difference beween the `preds` matrix and the `targets` matrix
- Remove negative values by squaring all the elements of the difference matrix
- Then calculate the average of all those elements in the final matrix

This is called the **Mean Squared Error (MSE)** and it results in a single number.

MSE loss
```
def mse(t1, t2):
   diff = t1 - t2
   return torch.sum(diff * diff) / diff.numel()
```
Apply the MSE loss function on the current predictions of the model.
```
loss = mse(preds, targets)
```
Which results in `tensor(15813.8125, grad_fn=<DivBackward0>)`

The result can be interpreted as how off the model is at predicitng the target values. So in this case, each predicted element differs from the target element by the sqrt of the loss result.

*The lower the loss, the better the model.*

### Computing Gradients
Using PyTorch, the gradient (derivative) of the loss can be automatically computed. 
```
loss.backward()
```
The gradients are now stored in the `.grad` property of each respective tensor (because they are with respect to the weights).

Remember that the weights were just a random tensor before. It was just a tensor matrix filled up with random values initially (which is why the model was so off in the first place).

This is what it was:

`w` = 
```
tensor([[-0.2910, -0.3450,  0.0305],
        [-0.6528,  0.7386, -0.5153]], requires_grad=True)
```
The gradient is stored in the `.grad` property:

`w.grad = `
```
tensor([[-10740.7393, -12376.3008,  -7471.2300],
        [ -9458.5078, -10033.8672,  -6344.1094]])
```

### Adjusting weights and baises
Since the loss is a function (a quadratic function in particular), the shape of the graph can show where the set of weights for the loss is the lowest. The graph can be plotted with respect to a weight or bias:
![positivegradients](https://i.imgur.com/WLzJ4xP.png)
A couple of things to note in the diagram above:
- the gradient is positive (aka the slope)
- an increase of the weight element will increase the loss
- a decrease of the weight element will decrease the loss
![negativegradients](https://i.imgur.com/dvG2fxU.png)

In this diagram above:
- the gradient is negative (aka the slope)
- an increase of the weight element will decrease the loss
- a decrease of the weight element will increase the loss

These observations are key for the *gradient descent* algorithm used to immprove the model. 

The loss will be slightly reduced as the weight element is adjusted.

Using `torch.no_grad` this can be implemented, and PyTorch will not track, calculate, or modify the gradients when updating the weights and biases.
```
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
```
The gradients are multiplied with a very small number (10^-5) so that the weights are not modified by too large of an amount. This number is called the *learning rate* of the algorithm.

Then quickly verify if the loss is actually any lower:
```
loss = mse(preds, targets)
print(loss)
```
which results in:
```
tensor(15813.8125, grad_fn=<DivBackward0>)
```
Lastly, reset the gradients to zero by using `.zero_()` so that PyTorch won't automatically accumulate the gradients. 
```
w.grad.zero_()
b.grad.zero_()
```
### Training with gradient descent
By using the gradient descent optimization algorithm, the loss is reduced and the model is improved. 

Training the model steps:
1. Generate predictions (using whatever the initial random numbers were)
2. Calculate the loss
3. Compute gradients (with respect to the weights and biases)
4. Adjust the weights (subtracting small quantities proportional to the gradient)
5. Reset the gradients to zero

Once again in code form:
```
# Generate predictions
preds = model(inputs)

# Calculate the loss
loss = mse(preds, targets)

# Compute gradients
loss.backward()

# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
    
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
```
The new weights and biases after the adjustment came out to be:
```
tensor([[-0.0961, -0.1190,  0.1667],
        [-0.4805,  0.9211, -0.3996]], requires_grad=True)
tensor([-0.9137, -0.7759], requires_grad=True)
```
The overall loss that got calculated is now lower: `tensor(7357.4829, grad_fn=<DivBackward0>)`

There is already quite a reduction in the loss with one step of the gradient descent optimizaton.

### Training with multiple epochs
The steps for gradient descent can now be repeated to further adjust the weights/biases several times to get closer and closer to the target.

An *epoch* is each iteration.

The easiest way is to use a for loop. Here the model is trained for 100 epochs:
```
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
```
Quickly verifying that the loss is lower: `tensor(130.3513, grad_fn=<DivBackward0>)`

After 100 epochs, the loss is now much lower than the initial value. 

So, at this point, the model's predictions shoul dbe pretty close to the targets.

Preds:
```
tensor([[ 60.8975,  70.5663],
        [ 83.9699,  92.9066],
        [108.6802, 150.1993],
        [ 43.5842,  38.4608],
        [ 91.6760, 104.6360]], grad_fn=<AddBackward0>)
```
Targets:
```
tensor([[ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.]])
```
And they are definitely quite close. With a few more epochs, the results can get even closer.
# PyTorch Built-ins
PyTorch provides several built-in functions to do some of the same steps above because linear regression is quite common. 

First things first, `torch.nn` package from PyTorch should be imported. This contains utillity classes for building neural networks.
```
import torch.nn as nn
```

As before, the inputs and the targets can be represented as matrices.
Input (temp, rainfall, humidity)
```
inputs = torch.tensor([[73, 67, 43], 
                       [91, 88, 64], 
                       [87, 134, 58], 
                       [102, 43, 37], 
                       [69, 96, 70], 
                       [74, 66, 43], 
                       [91, 87, 65], 
                       [88, 134, 59], 
                       [101, 44, 37], 
                       [68, 96, 71], 
                       [73, 66, 44], 
                       [92, 87, 64], 
                       [87, 135, 57], 
                       [103, 43, 36], 
                       [68, 97, 70]])
```
Targets (apples, oranges)
```
targets = torch.tensor([[56, 70], 
                        [81, 101], 
                        [119, 133], 
                        [22, 37], 
                        [103, 119],
                        [57, 69], 
                        [80, 102], 
                        [118, 132], 
                        [21, 38], 
                        [104, 118], 
                        [57, 69], 
                        [82, 100], 
                        [118, 134], 
                        [20, 38], 
                        [102, 120]])
```
This time, there are more training examples to show how to work with large datasets in small batches.

### Dataset and DataLoader
Creating a `TensorDataset` allows access to rows from `inputs` and `targets` as tuples and also provides standard APIs for working different types of datasets in PyTorch.
```
from torch.utils.data import TensorDataset
```
By using `TensorDataset`, a small section of the training data can be accessed (using array indexing notation `[0:3]`). It will return a tuple with two elements. The first contains the input variables for the selected rows. The second contains the targets.

Define the dataset:
```
train_ds = TensorDataset(inputs, targets)
```
Printing `train_ds` for the first 3 elements `[0:3]`
```
(tensor([[ 73.,  67.,  43.],
         [ 91.,  88.,  64.],
         [ 87., 134.,  58.]]),
 tensor([[ 56.,  70.],
         [ 81., 101.],
         [119., 133.]]))
```
Another built in function is `DataLoader` which splits the data into batches of predefined size while training. It can also shuffle data or do random sampling.
```
from torch.utils.data import DataLoader
```
Define the data loader:
```
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
```
The data loader can be used in a `for` loop. In each itereation, the data loader will return one batch of data with whatever the given batch size it. It can also shuffle the data before creating the batch, if `shuffle` is set to `True`. Shuffling is useful to randomize the input to the optimiaztion, which in turn leads to a faster reduction in loss.
```
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break
```
### nn.Linear
Using the `nn.Linear` class from PyTorch, the weight and biases can be automatically initialiazed to define the model.

Define the model:
```
model = nn.Linear(3, 2)
```
Now `model.weight` can be used:
```
tensor([[ 0.1304, -0.1898,  0.2187],
        [ 0.2360,  0.4139, -0.4540]], requires_grad=True)
```
And `model.bias` as well:
```
tensor([0.3457, 0.3883], requires_grad=True)
```
### Parameters

PyTorch models also have a helpful `.parameters` method, which returns a list containing all the weights and bias matrices present in the model. For the linear regression model, it contains of one weight matrix and one bias matrix.

Parameters:
```
list(model.parameters())
```
Returns:
```
[Parameter containing:
 tensor([[ 0.1304, -0.1898,  0.2187],
         [ 0.2360,  0.4139, -0.4540]], requires_grad=True),
 Parameter containing:
 tensor([0.3457, 0.3883], requires_grad=True)]
 ```
 
### Predictions
Predictions can be generated the same way as before.
 
Generate Predictions:
```
preds = model(inputs)
```
### Loss Function
Instead of defining the loss function manually, the built-in loss function `mse_loss` can be used.

The `nn.functional` package contains many useful functions and other utilities.
```
import torch.nn.functional as nnfunc
```
Define loss function:
```
loss_fn = nnfunc.mse_loss
```
Compute the loss for current predictions:
```
loss = loss_fn(model(inputs), targets)
```
### Optimizer
Instead of manually manipulating the model's weights and biases using gradients, the optimizer `optim.SGD` can be used. SGD is short for "stochastic gradient descent". The term *stochastic* indicates that samples are selected in random batches instead of as a single group.

Define optimizer:
```
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
```
Note that model.parameters() is passed as an argument to `optim.SGD` so that the optimizer knows which matrices should be modified during the update step. Also, the learning rate can be specifeid that controls the amount by which the parameters are modified.

### Train the model
All the steps from before will be the same, the only difference is that there will be batches of data to process instead of training the entire data in every iteration.

Same steps overall:
1. Generate predictions
2. Calculate the loss
3. Compute gradients with respect to the weights and biases
4. Adjust the weights by subtracting a small quantity proportional to the gradient
5. Reset the gradients to zero

The utility function `fit` will be used to train the model fro a given number of epochs. The data loader defined earlier will also be used to get batches of data for every iteration. 
```
# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```
Instead of manually updating the parameters (weights and biases), `opt.step` can perform the update and `opt.zero_grad` resets the gradients to zero. The print statement keeps track of the training process by printing the loss from the previous batch of data for every 10th epoch. Then `loss.item` returns the actual value stored in the tensor.

Train the model for 100 epochs:
```
fit(100, model, loss_fn, opt, train_dl)
```
Printed outcomes:
```
Epoch [10/100], Loss: 818.6476
Epoch [20/100], Loss: 335.3347
Epoch [30/100], Loss: 190.3544
Epoch [40/100], Loss: 131.6701
Epoch [50/100], Loss: 77.0783
Epoch [60/100], Loss: 151.5671
Epoch [70/100], Loss: 151.0817
Epoch [80/100], Loss: 67.6262
Epoch [90/100], Loss: 53.6205
Epoch [100/100], Loss: 33.4517
```
Generate predictions using the model:
```
preds = model(inputs)
```
preds:
```
tensor([[ 58.4229,  72.0145],
        [ 82.1525,  95.1376],
        [115.8955, 142.6296],
        [ 28.6805,  46.0115],
        [ 97.5243, 104.3522],
        [ 57.3792,  70.9543],
        [ 81.9342,  94.1737],
        [116.2036, 142.6871],
        [ 29.7242,  47.0717],
        [ 98.3498, 104.4486],
        [ 58.2047,  71.0507],
        [ 81.1088,  94.0774],
        [116.1137, 143.5935],
        [ 27.8550,  45.9152],
        [ 98.5680, 105.4124]], grad_fn=<AddmmBackward>)
```
compare with targets:
```
tensor([[ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.],
        [ 57.,  69.],
        [ 80., 102.],
        [118., 132.],
        [ 21.,  38.],
        [104., 118.],
        [ 57.,  69.],
        [ 82., 100.],
        [118., 134.],
        [ 20.,  38.],
        [102., 120.]])
```
Indeed, the predictions are quite close to the targets. It's become a reasonably good trained model to predict crop yields for apples and oranges by looking at the average temperature, rainfall, and humidity in a region. Now it can be used to make predictions of crop yields for new regions by passing a batch containing a single row of input.

New input:
```
model(torch.tensor([[75, 63, 44.]]))
```
Result:
```
tensor([[55.3323, 67.8895]], grad_fn=<AddmmBackward>)
```
The predicted yield of apples is 54.3 tons per hectare, and for oranges it's 68.3 tons per hectare.
