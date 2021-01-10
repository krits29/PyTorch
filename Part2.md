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

