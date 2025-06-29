import numpy as np
import pandas as pd
from statsmodels.tools.tools import add_constant


rng = np.random.default_rng(787)

n = 1000
X = rng.uniform(0.5, 2, n)
beta_0 = 0.5
beta_1 = -0.75
epsilon = rng.normal(0, 1, n)
Y = beta_0 + beta_1 * X + epsilon
data = pd.DataFrame({"X": X, "Y": Y})


X = add_constant(data["X"].values)

theta = np.array([0.0, 0.0, 1.0])
max_iter = 100
tol = 1e-6

for i in range(max_iter):
    beta = theta[:2]
    sigma2 = theta[2]
    residuals = Y - X @ beta

    grad_beta = (X.T @ residuals) / sigma2
    grad_sigma2 = -n / (2 * sigma2) + 0.5 * np.sum(residuals**2) / sigma2**2
    grad = np.concatenate((grad_beta, [grad_sigma2]))

    H_beta = -(X.T @ X) / sigma2
    H_cross = -(X.T @ residuals) / sigma2**2
    H_sigma2 = n / (2 * sigma2**2) - np.sum(residuals**2) / sigma2**3

    H = np.zeros((3, 3))
    H[:2, :2] = H_beta
    H[:2, 2] = H_cross
    H[2, :2] = H_cross
    H[2, 2] = H_sigma2

    delta = np.linalg.solve(H, grad)
    theta_new = theta - delta

    if np.linalg.norm(delta) < tol:
        break
    theta = theta_new

print("Estimated parameters:", theta)
