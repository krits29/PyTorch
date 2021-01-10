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
