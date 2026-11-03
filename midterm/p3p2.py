import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]).reshape(-1, 1)
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

model = LinearRegression().fit(X, Y)
print("scikit-learn implementation:")
print(f"intercept: {model.intercept_:.6f}")
print(f"slope:     {model.coef_[0]:.6f}")


# bonus cuML solution
try:
    from cuml.linear_model import LinearRegression as cuLinearRegression
    cu_model = cuLinearRegression().fit(X, Y)
    print("\ncuML implementation:")
    print(f"intercept: {float(cu_model.intercept_):.6f}")
    print(f"slope:     {float(cu_model.coef_[0]):.6f}")
except Exception as e:
    print(f"\ncuML implementation skipped: ({type(e).__name__})")