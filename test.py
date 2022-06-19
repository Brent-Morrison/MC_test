from io import StringIO
import numpy as np
import pandas as pd
from numba import jit
from mc_test import *
from utils import *

# Prices
file_content1 = """
 1.00 	 1.00 	 1.00 	 1.00 	 1.00 	 1.00 	 1.00 	 1.00 	 1.00 	 1.00 
 1.11 	 0.84 	 1.40 	 0.75 	 0.93 	 0.83 	 0.90 	 1.09 	 0.91 	 1.05 
 1.01 	 0.94 	 0.70 	 0.84 	 1.11 	 0.66 	 1.06 	 1.22 	 1.11 	 1.08 
 1.18 	 0.93 	 1.05 	 0.83 	 1.11 	 0.62 	 1.15 	 1.36 	 0.97 	 0.98 
 1.41 	 0.76 	 0.45 	 0.70 	 1.07 	 0.63 	 1.44 	 1.47 	 0.87 	 1.03 
 1.61 	 0.82 	 0.40 	 0.86 	 1.09 	 0.56 	 1.67 	 1.56 	 0.97 	 1.01 
 1.36 	 0.63 	 0.86 	 0.90 	 0.89 	 0.50 	 1.91 	 1.33 	 1.14 	 0.80 
 1.13 	 0.57 	 0.74 	 0.95 	 1.05 	 0.41 	 1.55 	 1.52 	 1.12 	 0.70 
 1.20 	 0.51 	 0.64 	 0.97 	 1.29 	 0.40 	 1.76 	 1.17 	 1.29 	 0.59 
 1.23 	 0.55 	 0.73 	 0.98 	 1.09 	 0.33 	 1.95 	 1.05 	 1.29 	 0.72 
 1.45 	 0.55 	 0.88 	 0.79 	 1.12 	 0.25 	 1.66 	 0.94 	 1.30 	 0.76 
 1.72 	 0.44 	 1.03 	 0.98 	 0.92 	 0.31 	 1.85 	 1.14 	 1.40 	 0.88 
 1.61 	 0.50 	 1.01 	 1.18 	 0.90 	 0.33 	 1.76 	 1.13 	 1.32 	 0.95 
 1.63 	 0.39 	 0.87 	 1.34 	 0.88 	 0.40 	 1.99 	 1.22 	 1.15 	 0.81 
 1.36 	 0.43 	 0.88 	 1.41 	 0.79 	 0.38 	 2.47 	 1.41 	 0.89 	 0.83 
 1.12 	 0.50 	 0.67 	 1.66 	 0.72 	 0.30 	 2.86 	 1.29 	 0.72 	 0.88 
 1.33 	 0.50 	 0.77 	 1.50 	 0.60 	 0.27 	 3.50 	 1.16 	 0.55 	 0.92 
 1.45 	 0.42 	 0.67 	 1.53 	 0.71 	 0.31 	 2.88 	 0.97 	 0.51 	 1.00 
 1.50 	 0.46 	 0.66 	 1.19 	 0.65 	 0.23 	 2.86 	 0.80 	 0.60 	 1.07 
 1.37 	 0.51 	 0.77 	 1.39 	 0.56 	 0.19 	 2.60 	 0.94 	 0.47 	 1.30 
"""

# Positions
file_content2 = """
1	1	1	1	0	0	0	0	0	0
0	0	1	1	1	1	0	0	0	0
0	0	0	1	1	1	1	1	0	0
0	0	0	0	0	0	1	1	1	1
0	0	0	0	1	1	1	1	0	0
0	0	1	1	1	1	0	0	0	0
1	1	1	1	0	0	0	0	0	0
0	0	1	1	1	1	0	0	0	0
0	0	0	1	1	1	1	1	0	0
0	0	0	0	0	0	1	1	1	1
0	0	0	0	1	1	1	1	0	0
0	0	1	1	1	1	0	0	0	0
1	1	1	1	0	0	0	0	0	0
0	0	1	1	1	1	0	0	0	0
0	0	0	1	1	1	1	1	0	0
0	0	0	0	0	0	1	1	1	1
0	0	0	0	1	1	1	1	0	0
0	0	1	1	1	1	0	0	0	0
1	1	1	1	0	0	0	0	0	0
0	0	1	1	1	1	0	0	0	0
"""

# Positions
file_content3 = """
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0
"""

# Load dummy data
with StringIO(file_content1) as f:
  prices1 = np.loadtxt(f, dtype=float)

with StringIO(file_content2) as f:
  positions1 = np.loadtxt(f, dtype=float)

with StringIO(file_content3) as f:
  positions2 = np.loadtxt(f, dtype=float)

# Load csv data
prices = np.genfromtxt(r'C:\Users\brent\Documents\R\Misc_scripts\prices_mtrx.csv', skip_header=1, delimiter=',', dtype=float)
positions = np.genfromtxt(r'C:\Users\brent\Documents\R\Misc_scripts\positions_mtrx.csv', skip_header=1, delimiter=',', dtype=float)


# Test non-jit function
np_test_df = monte_carlo_backtest_np(
  prices1, 
  positions1, 
  seed_capital = 100.0, 
  max_positions = 6, 
  rndm = False,
  verbose = True
  )

print('max_drawdown: '.ljust(20), np_test_df[0])
print('cagr: '.ljust(20), np_test_df[1])
print('vol: '.ljust(20), np_test_df[2])
print('pfolio_val: '.ljust(20), np_test_df[3])


# Test jit function
test_df = monte_carlo_backtest(
  prices1, 
  positions1, 
  seed_capital = 100.0, 
  max_positions = 6, 
  rndm = False,
  verbose = True
  )

print('\n')
print('max_drawdown: '.ljust(20), test_df[0])
print('cagr: '.ljust(20), test_df[1])
print('vol: '.ljust(20), test_df[2])
print('pfolio_val: '.ljust(20), test_df[3])



# Test looped jit function
import time
tic = time.perf_counter()
test_df1 = monte_carlo_backtest1(
  prices1, 
  positions1, 
  seed_capital = 100.0, 
  max_positions = 4,
  iter = 100,
  rndm = False
  ) 
toc = time.perf_counter()
print('Elapsed time: ', round(toc - tic,2), 'seconds')

# Extract position data from df
test_df1.iloc[:,5]
test_df1.iloc[:,4]




# Utils test
# ----------

from utils import *
import time

array_idx_0 = None
array_idx_1 = np.array([1,0,1,0])

arr0 = np.nonzero(array_idx_0)[0]
arr1 = np.nonzero(array_idx_1)[0]
arr2 = np.array([0,1,2,2,3,4])
arr3 = np.random.choice(arr2, size=3, replace=False)
arr4 = np.array([], dtype=np.int64)
arr5 = np.random.choice(arr2, size=0, replace=False)

print('DATA')
print('arr0: ', arr0, 'type: ', arr0.dtype)
print('arr1: ', arr1, 'type: ', arr1.dtype)
print('arr2: ', arr2, 'type: ', arr2.dtype)
print('arr3: ', arr3, 'type: ', arr3.dtype)
print('arr4: ', arr4, 'type: ', arr4.dtype)
print('arr5: ', arr5, 'type: ', arr5.dtype)
print('\n')
print('UNION')
print('un_34: ', union(arr3, arr5))
print('un_43: ', union(arr4, arr3))
print('un_12: ', union(arr1, arr2))
print('un_21: ', union(arr2, arr1))
print('\n')
print('INTERSECTION')
print('is_34: ', intersect(arr3, arr4))
print('is_43: ', intersect(arr4, arr3))
print('is_12: ', intersect(arr1, arr2))
print('is_21: ', intersect(arr2, arr1))
print('\n')
print('SET DIFFERENCE')
print('sd_02: ', setdiff(arr0, arr2))
print('sd_20: ', setdiff(arr2, arr0))
print('sd_12: ', setdiff(arr1, arr2))
print('sd_21: ', setdiff(arr2, arr1))


# https://github.com/numba/numba/issues/2648
@jit(nopython=True)
def rnd1(x, decimals, out):
  return np.round_(x, decimals, out)

@jit(nopython=True)
def rnd2(x, decimals):
  return np.round_(x, decimals)



x=np.arange(10.) + 0.2
print(x)
print(np.round_(x))
y=np.empty_like(x)
print(rnd1(x, 0, y))


print(rnd2(x, 0))