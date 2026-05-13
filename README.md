# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the salary dataset using pandas.read_csv().

2.Separate the input features and target variable, then convert categorical data using get_dummies().

3.Split the dataset into training and testing sets using train_test_split().

4.Create and train the DecisionTreeRegressor model using the training data.

5.Visualize the trained decision tree using plot_tree() and display the graph using matplotlib

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Mithun Kumar V
RegisterNumber:  212225040236
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
data = pd.read_csv("Salary (1).csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()


```

## Output:
<img width="1501" height="757" alt="image" src="https://github.com/user-attachments/assets/8c2d526d-683e-424d-be43-943670e7874b" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
