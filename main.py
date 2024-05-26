import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import cupy as cp

# define parameterss
d = 8
l_list = [1, 2, 6, 8]

# Generate thera array to plot on graph
theta_array = np.arange(0.01, np.pi/2, np.pi/256)
# Generate y_c array to plot on graph
y_c_array = np.arange(0.01, d/2, d/256)

def condition(l, y_c, theta):
    cond = l * np.cos(theta) / (2 * y_c) > 1
    return cond

def touched_patterns(l, y_c_array, theta_array):
    patterns = []
    for y_c in y_c_array:
        for theta in theta_array:
            if condition(l, y_c, theta):
                patterns.append([l, y_c, theta])
    return patterns

patterns_dict = {l: [] for l in l_list}
print(isinstance(patterns_dict, dict))

df = pl.DataFrame()
for l in patterns_dict.keys():
    patterns_dict[l] = touched_patterns(l, y_c_array, theta_array)
    
    df_temp = pl.DataFrame(patterns_dict[l])
    df = df.vstack(df_temp)
df.columns = ['l', 'y_c', 'theta']
print(df.shape)
print(df.head(10))

df = df.sort('l', descending=True)
# Plot
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(12, 8))
# sns.scatterplot(data=df, x='y_c', y='theta', hue='l', palette='deep')
# plt.show()

theta_array = np.random.uniform(0, np.pi/2, 10^2)
y_c_array = np.random.uniform(0, d/2, 10^2)

"""
1-2 で求めた確率 p(l) / l = 1/ (pi * d) となるので
y_c, theta の一様乱数を生成して数値的に p(l) の近似値を得れば
1/ (pi *d ) すなわり pi の近似値を得ることができる。
"""

all_pairs = []

theta_array = np.random.uniform(0, np.pi/2, 10^2)
y_c_array = np.random.uniform(0, d/2, 10^2) # d = 8

for theta in theta_array:
  for y_c in y_c_array:
    all_pairs.append([theta, y_c])

count_all_pairs = len(all_pairs)

l = 2

# count_touched = len(touched_patterns(l, theta_array, y_c_array))
# posibility = count_touched / count_all_pairs
# pi = 2 * l / d / posibility
# print(pi)

"""
1-2 で求めた確率 p(l) / l = 1/ (pi * d) となるので
y_c, theta の一様乱数を生成して数値的に p(l) の近似値を得れば
1/ (pi *d ) すなわり pi の近似値を得ることができる。
"""

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
pi_array = np.array([pi_approxer(8, 2, count_throw) for count_throw in count_throw_array])

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(x=count_throw_array, y=pi_array)
plt.xscale('log')
plt.show()