from io import StringIO
import numpy as np
import pandas as pd
from numba import jit
from mc_test import *

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
  prices = np.loadtxt(f, dtype=float)

with StringIO(file_content2) as f:
  positions = np.loadtxt(f, dtype=float)

# Load csv data
prices = np.genfromtxt(r'C:\Users\brent\Documents\R\Misc_scripts\prices_mtrx.csv', skip_header=1, delimiter=',', dtype=float)
positions = np.genfromtxt(r'C:\Users\brent\Documents\R\Misc_scripts\positions_mtrx.csv', skip_header=1, delimiter=',', dtype=float)

# Test
test_df = monte_carlo_backtest(
  prices, 
  positions, 
  seed_capital = 1000, 
  max_positions = 5, 
  rndm = True,
  verbose = False
  )

# Test
test_df1 = monte_carlo_backtest1(
  prices, 
  positions, 
  seed_capital = 100, 
  max_positions = 5,
  iter = 10000,
  rndm = True
  ) 