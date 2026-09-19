# Multiple Linear Regression using OLS

## Overview

This project implements **Multiple Linear Regression from scratch** using the **Ordinary Least Squares (OLS) Normal Equation**.

The goal is to understand how multiple linear regression works mathematically and implement the complete process using **NumPy**, without relying on a machine learning library such as Scikit-learn.

The model uses multiple independent variables to predict a continuous dependent variable.

## Dataset

The example dataset contains:

- **Area** – size of the house
- **Bedrooms** – number of bedrooms
- **Price** – target variable

Example:

| Area | Bedrooms | Price |
|---:|---:|---:|
| 1000 | 2 | 200 |
| 1500 | 3 | 300 |
| 2000 | 4 | 400 |
| 2500 | 4 | 450 |
| 3000 | 5 | 550 |

## Multiple Linear Regression

The model is represented as:

ŷ = β₀ + β₁X₁ + β₂X₂

Where:

- **ŷ** = predicted value
- **β₀** = intercept
- **β₁** = coefficient of the first predictor
- **β₂** = coefficient of the second predictor
- **X₁, X₂** = input variables

## OLS Normal Equation

The regression coefficients are calculated using the Normal Equation:

β = (XᵀX)⁻¹Xᵀy

Before applying the equation, a column of ones is added to the feature matrix to account for the intercept.

```python
X_b = np.c_[np.ones(X.shape[0]), X]

Implementation Steps

The project follows these steps:

Create the dataset
Add the intercept column
Calculate the transpose of the feature matrix
Calculate XᵀX
Calculate the inverse of XᵀX
Calculate Xᵀy
Calculate regression coefficients using the Normal Equation
Generate predicted values
Calculate residuals
Calculate SSE
Calculate SST
Calculate R-squared
Calculate Adjusted R-squared
Visualize actual vs. predicted values
Model Evaluation
Residuals

Residuals represent the difference between the actual and predicted values:

Residual = y - ŷ

Sum of Squared Errors (SSE)

SSE measures the total squared residual error:

SSE = Σ(y - ŷ)²

Total Sum of Squares (SST)

SST measures the total variation in the actual target values around their mean:

SST = Σ(y - ȳ)²

R-squared

R-squared measures the proportion of variation in the target variable explained by the regression model:

R² = 1 - (SSE / SST)

Adjusted R-squared

Adjusted R-squared accounts for the number of predictors included in the model:

Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)

Where:

n = number of observations
p = number of predictors
Visualization

The project includes an Actual vs. Predicted plot.

The x-axis represents the actual target values, while the y-axis represents the predicted values.

A perfect prediction would lie on the diagonal line where:

Predicted = Actual

The closer the points are to this line, the closer the predictions are to the actual values.

Technologies Used
Python
NumPy
Matplotlib
Key Concepts Learned

This project helped me understand:

Multiple Linear Regression
Ordinary Least Squares (OLS)
Normal Equation
Intercept
Regression coefficients
Matrix multiplication
Matrix transpose
Matrix inverse
Predictions
Residuals
SSE
SST
R-squared
Adjusted R-squared
Actual vs. Predicted visualization

Project Structure
Multiple-Linear-Regression-OLS/
│
├── multiple_linear_regression_ols.py
└── README.md

