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



@jit(nopython=True)
def union(arr1, arr2):
  # Set types
  arr1 = arr1.astype(np.int64)
  arr2 = arr2.astype(np.int64)
  
  if len(arr1) == 0:
    x = arr2
  elif len(arr2) == 0:
    x = arr1
  else:
    x = np.array(list(set(arr1) | set(arr2)))
  return x



@jit(nopython=True)
def max_dd(x):
  run_max = x[0]
  roll_max = np.empty_like(x)
  dd_perc = np.empty_like(x)
  for i, vltn in enumerate(x):
    if vltn >= run_max:
      run_max = vltn
    roll_max[i] = run_max
    dd_perc[i] = (roll_max[i] - x[i]) / roll_max[i]
    #print(i, np.round(x[i],2), np.round(roll_max[i],2), np.round(dd_perc[i],5))  
  return max(dd_perc)


def group_choice_np(x, s, r):
  res = []
  for i in np.unique(x):
    print(i)
    arr = np.random.choice(np.where(x==i)[0], size=s, replace=r)#.flatten()
    print(arr)
    res = np.concatenate((res, arr))#, axis=None)
  return res


@jit(nopython=True)
def group_choice(x, s, r):
  
  # Initialise empty list - messing to type as int64
  res = np.random.choice(x, size=0, replace=False)
  res = res.astype(np.int64)

  # Sample by group
  for i in np.unique(x):
    print(i)
    arr = np.random.choice(np.where(x==i)[0], size=s, replace=r)#.flatten()
    print(arr)
    res = union(res, arr)
  
  return res

