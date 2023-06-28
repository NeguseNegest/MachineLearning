import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

def intercept_slope():
    Temprature = np.array([2, 3, 4, 5, 6, 7, 8])
    Humidity = np.array([68, 75, 83, 89, 92, 95, 98])
    n = len(Temprature)
    sum_x = np.sum(Temprature)
    sum_y = np.sum(Humidity)
    sum_xx = np.sum(Temprature * Temprature)
    sum_xy = np.sum(Temprature * Humidity)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    return print(f"Intercept is: {intercept}, Slope is: {slope}")

intercept_slope()

"""Task 1 solution 
Intercept is: 60.892857142857146, Slope is: 4.964285714285714
the results seem to indicate that we have an increase of 4 points in per hour studied"""

"-----------------------TASK 2-----------------------"

def gradient_desc_linear():
    w_0=0
    w_1=0

    X=np.array([1,2,3,4,5])
    Y=np.array([3,5,7,9,11])
    n=int(len(X))#Number of data points
    a=0.01 #Learning rate
    iterations=2
    for i in range(iterations):
        Predict=w_0*X+w_1
        dw_1=(-2/n)*sum(X*(Y-Predict))
        dw_2=(-2/n)*sum(Y-Predict)
        w_0-=a*dw_1
        w_1-=a*dw_2
        
    return print(f"Task 2, intercept value is :{w_0},slope is : {w_1}")


"----------------Task 3------------"


def z_score_norm():
    age = [28, 35, 42, 25, 30]
    income = [6000, 70000, 80000, 55000, 65000]
    n = len(age)
    mean_age = sum(age) / n
    mean_income = sum(income) / n
    sdeviation_age = math.sqrt(sum((x - mean_age) ** 2 for x in age) / (n - 1))
    sdeviation_income = math.sqrt(sum((x - mean_income) ** 2 for x in income) / (n - 1))
    
    for i in range(n):
        age[i] = (age[i] - mean_age) / sdeviation_age
        income[i] = (income[i] - mean_income) / sdeviation_income
    
    return age, income


"---------------------Task 4----------------"
"""MSE for model1 :53.9999999999999,MSE for model2 :35.89991335626337,
The results seem to indicate that the model2 is better since its mean squared error is less by almost 20, 
it is therfore better to use the model2 if we want accurate predictions. """

def univariat_linear():
    Temperature = np.array([25, 30, 35, 20, 28])
    Humidity = np.array([50, 60, 70, 40, 55])/100
    Energy_Consumption = np.array([200, 250, 300, 180, 220])
    X = np.column_stack((Temperature, Humidity))
    y = Energy_Consumption
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse1 = mean_squared_error(y, predictions)
    return mse1
    
    
def multivariat_linear():
    Temperature = np.array([25, 30, 35, 20, 28])
    Humidity = np.array([50, 60, 70, 40, 55])/100
    Energy_Consumption = np.array([200, 250, 300, 180, 220])
    Synthetic_Feature = Temperature * Humidity

    X_synthetic = Synthetic_Feature.reshape(-1, 1)
    y = Energy_Consumption
    model = LinearRegression()
    model.fit(X_synthetic, y)
    predictions = model.predict(X_synthetic)

    mse2 = mean_squared_error(y, predictions)

    
    return mse2


def main():
    gradient_desc_linear()
    intercept_slope()
    model1=univariat_linear()
    model2=multivariat_linear()
    print(z_score_norm())
    print(f"MSE for model1:{model1},MSE for model2:{model2}")

main()
