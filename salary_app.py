import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write("""
# Salary Prediction Model

Below are our salary predictions based on experience.
""")

# Import the dataset
dataset = pd.read_excel("salary.xlsx")

# Split the data into independent (X) and dependent (y) variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values 

# Split the dataset into training and testing sets (80-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Import and fit the Simple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test the model & create a predicted table 
y_pred = regressor.predict(X_test)

# Visualize training data points
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(X_train, y_train, color='red') 
ax.plot(X_train, regressor.predict(X_train), color='blue')
ax.set_title('Salary vs Experience (Training set)')
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
st.pyplot(fig)

# Visualize test data points 
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(X_test, y_test, color='red') 
ax.plot(X_train, regressor.predict(X_train), color='blue')
ax.set_title('Salary vs Experience (Testing set)')
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
st.pyplot(fig)

# Display model performance
bias = regressor.score(X_train, y_train)
st.write("Training Phase R²:", bias)

variance = regressor.score(X_test, y_test)
st.write("Testing Phase R²:", variance)
