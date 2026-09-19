import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
# dataset
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 4],
    [3000, 5]
])

y = np.array([
    200,
    300,
    400,
    450,
    550
])

#add intercept column

X_b = np.c_[np.ones(X.shape[0]), X]

#Normal Equation
X_T = X_b.T

XTX = X_T @ X_b

XTX_inv = np.linalg.inv(XTX)

XTy = X_T @ y

beta = XTX_inv @ XTy


# 4. Display coefficients
print("Coefficcients:")
print(beta)


#5.Predictions

y_pred = X_b @ beta

print("\nPredictions:")
print(y_pred)


#Residuals
residuals = y - y_pred
print("\nResiduals:")
print(residuals)

#SSE

SSE = np.sum(residuals ** 2)

print("\nSSE:", SSE)

#SST
y_mean = np.mean(y)

SST = np.sum((y - y_mean) ** 2)

print("SST:", SST)

#R-squared

R_squared = 1 - (SSE / SST)
print("R-squared:", R_squared)

#Adjsuted r_squared

n = X.shape[0]  # number of observations
p = X.shape[1]  # number of predictors

adjusted_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - p - 1)

print("Adjusted R-squared:", adjusted_R_squared)

plt.scatter(y, y_pred, color='blue', label='Data points')

plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Perfect Prediction Line')
plt.show()


plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted y")
plt.legend()


# Area vs Price
plt.scatter(X[:, 0], y)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# Bedrooms vs Price
plt.scatter(X[:, 1], y)
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs Price")
plt.show()
