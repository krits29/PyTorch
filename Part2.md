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

### Training multiple epochs
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

