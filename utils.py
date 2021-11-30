import numpy as np
from numba import jit

@jit(nopython=True)
def intersect(arr1, arr2):
  # Set types
  arr1 = arr1.astype(np.int64)
  arr2 = arr2.astype(np.int64)
  
  if len(arr1) == 0:
    x = arr1
  elif len(arr2) == 0:
    x = arr2
  else:
    x = np.array(list(set(arr1) & set(arr2)))
  return x



@jit(nopython=True)
def setdiff(arr1, arr2):
  # Set types
  arr1 = arr1.astype(np.int64)
  arr2 = arr2.astype(np.int64)
  
  if len(arr1) == 0:
    x = arr2
  elif len(arr2) == 0:
    x = arr1
  else:
    x = np.array(list(set(arr1) ^ set(arr2)))
  return x