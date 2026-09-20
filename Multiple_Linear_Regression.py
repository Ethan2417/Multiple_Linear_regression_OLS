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
print("Intercept:", beta[0])
print("Coefficients:", beta[1:])

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

# 3D visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Actual data points
ax.scatter(
    X[:, 0],
    X[:, 1],
    y,
    color="blue",
    label="Actual values"
)

# Create grid for regression plane
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)

X1, X2 = np.meshgrid(x1_range, x2_range)

# Calculate predicted y for every point on the grid
Y_pred = (
    beta[0]
    + beta[1] * X1
    + beta[2] * X2
)

# Plot regression plane
ax.plot_surface(
    X1,
    X2,
    Y_pred,
    alpha=0.5
)

ax.set_xlabel("House Size")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")

ax.set_title("Multiple Linear Regression - 3D")

plt.show()


