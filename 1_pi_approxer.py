import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import cupy as cp

# 高速化のため qupy でGPGPU行列演算させる
def condition(l, y_c, theta):
    return l * cp.cos(theta) / (2 * y_c) > 1

def pi_approxer(d: int, l: int, count_pairs: int = 10**2):
    count_pairs_sqrt = int(np.sqrt(count_pairs))
    theta_array = np.random.uniform(0, np.pi / 2, count_pairs_sqrt).astype(np.float16)
    y_c_array = np.random.uniform(0, d / 2, count_pairs_sqrt).astype(np.float16)

    theta_array = cp.asarray(theta_array, dtype=cp.float16)
    y_c_array = cp.asarray(y_c_array, dtype=cp.float16)

    # Use numpy broadcasting to calculate the condition for all pairs
    touched = condition(l, y_c_array[:, np.newaxis], theta_array)

    # Count the number of True values
    count_touched = np.sum(touched)
    count_all_pairs = count_pairs

    posibility = count_touched / count_all_pairs
    pi = 2 * l / d / posibility
    return pi

count_throw_array = np.logspace(2, 8, 100).astype(int)
pi_array = np.array([pi_approxer(8, 2, count_throw).get() for count_throw in count_throw_array])