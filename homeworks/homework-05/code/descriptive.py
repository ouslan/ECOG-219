import numpy as np
import pandas as pd


rng = np.random.default_rng(787)

n = 1000
X = rng.uniform(0.5, 2, n)
beta_0 = 0.5
beta_1 = -0.75
epsilon = rng.normal(0, 1, n)
Y = beta_0 + beta_1 * X + epsilon
data = pd.DataFrame({"X": X, "Y": Y})
data.describe()
