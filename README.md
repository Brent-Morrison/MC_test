# MC_test
Monte Carlo back test.  A randomised back test procedure for stock trading strategies.  

## Use case
Trading strategies implemented over a large investable universe can generate signals for many stocks.  

A classic example of this is a retail investor seeking to replicate an academic "anomaly" portfolio.  For 
example a momentum portfolio will invest in the top 10% of stocks experiencing the largest 12 month price
increase.  If the universe under which this is implemented contains 1,000 stocks (as is typical in assessing 
the significance of these anomalies), the strategy calls for 100 stocks to be held.  This is obviously 
unrealistic for a portfolio seeded with €20k cash.  In this example an investor may desire to hold at most 
10 positions.  

How do we form an expectation of the performance of choosing 10 positions at random from a population of 100
stocks available for selection (we assume indifference within the 100 stocks available for selection).  Does performance attributable to stock 1 through 10 differ to that for 91 though 100?

We can answer this questions using simulation.  

The functions in this project perform Monte Carlo simulations on stocks for which a 
trading signal has been generated.  Specifically, if there are more tradings signals generated than desired
holdings, a random selection of available stocks are selected.  

Naively implemented such a routine can result in excessive turnover, 100% turnover in some instances.  An example of this is evident in the example provided above, in period 1 stocks 1 though 10 are selected (this determined via the randomised selection procedure) and held, then disposed of, and then stocks 91 through 100 are randomly selected and held in period 2.  This is not a realistic investment approach should say stocks 1 through 8 remain a buy signal in period 2.  These would continue to be held in order to avoid transaction costs (remember we are indifferent to selection across stocks that have a buy signal).  To mitigate this unrealistic turnover, the simulation process enforces a rule such that should a stock be held in period t, and it remains buy signal in period t+1, it will not be disposed of.  Closing out the example above, 2 positions would become available to be randomly sampled in period 2 from the remaining population of buy positions, maintaining the portfolio position size of 10 stocks.  


## Future development
Portfolio rebalancing has not been implemented.


## Performance
The functions have been written in NumPy and accelerated with Numba.  

Numba requires a rigid type structure and does not support all NumPy functions.  As such, making things work can be tricky.  Some issues encountered with Numba:  

- Numba supports only one advanced index.  This requires implementing loops as opposed to vectorised operations in certain circumstances.  For example this code...
    ```python
    open_positions_reval = np.sum(price_change[open_positions_idx] * holding[r-1,open_positions_idx])
    ``` 
    ...needs to be expressed as:  
    ```python
    open_positions_reval = 0
    for i in open_positions_idx:
        open_positions_reval = open_positions_reval + (price_change[i] * holding[r-1,i])
    ```
    Reference [1](https://github.com/numba/numba/issues/2157) and [2](https://github.com/numba/numba/issues/5389).  

- Numba does not support the ```np.c_``` function.  ```np.concatenate``` has been used as a replacement.  

- The use of ```np.random.choice``` throws this error *"TypeError: np.random.choice() first argument should be int or array, got range_state_int64"*.  This requires the first argument being changed from ```range(positions.shape[1])``` to ```np.arange(positions.shape[1])```.  

- ```setdiff1d``` and ```intersect1d``` are unsupported.  In replacing these with a set operation, for example ```np.array(list(set(open_positions_idx) & set(available_positions_idx)), dtype=int)```, an additional error is thrown.  Numba needs to be able to infer a type, it can not do so in the case of an empty list (as can be the case with this intersection).  See [here](https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-has-an-untyped-list-problem).  One workaround is to use a [typed list](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#typed-list).  Alternatively, a set intersection function can be defined.  An example can be found [here](https://stackoverflow.com/questions/59959207/intersection-of-two-lists-in-numba).  

- Back to ```np.concatenate```, this...
    ```python
    if sample_idx is not None:
      c = np.concatenate((sample_idx, hold_position_idx))
    else:
      c = hold_position_idx
    ``` 
    ... is not going to work.  This is because there are *"code-paths that try to assign different types of arrays to the same variable"*, the ```None``` type is not compatible with an array.  Further explanation [here](https://stackoverflow.com/questions/51754268/using-numpy-vstack-in-numba).

- Accumulate operations on NumPy ufuncs is not supported.  Therefore maximum drawdown, which was calculated with code as so...  
    ```python
    np.max((np.maximum.accumulate(portfolio_vltn, axis=0) - portfolio_vltn) / np.maximum.accumulate(portfolio_vltn, axis=0))
    ```
    ...has been replaced with a custom function.  
    ```python
    def max_dd(x):
      run_max = x[0]
      roll_max = np.empty_like(x)
      dd_perc = np.empty_like(x)
      for i, vltn in enumerate(x):
        if vltn >= run_max:
          run_max = vltn
        roll_max[i] = run_max
        dd_perc[i] = (roll_max[i] - x[i]) / roll_max[i]
      return max(dd_perc)
  ``` 

- See [this](https://stackoverflow.com/questions/67160311/does-numba-support-in-built-python-function-e-g-setitem) re setitem functionality

## Other issues  
- Conda environment and debugging.  It turns out NumPy does not import correctly when using the Visual Studio code debugger.  Workaround [here](https://github.com/microsoft/vscode-python/issues/13500).