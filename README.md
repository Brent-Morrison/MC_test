# MC_test
Monte Carlo back test.  A randomised back test procedure for stock trading strategies.  

## Use case
Trading strategies implemented over a large stocks universe can generate signals for many stocks.  

A classic example of this is a retail investor seeking to replicate an academic "anomaly" portfolio.  For 
example a momentum portfolio will invest in the top 10% of stocks experiencing the largest 12 month price
increase.  If the universe under which this is implemented contains 1,000 stocks (as is typical in assessing 
the significance of these anomalies), the strategy calls for 100 stocks to be held.  This is obviously 
unrealistic for a portfolio seeded with €20k cash.  In this example an investor may desire to hold at most 
10 positions.  

How to form an expectation of the performance of choosing 10 positions at random from a population of 100
stocks available for selection (we assume indifference within the 100 stocks available for selection). 

We can answer this questions using simulation.  

The functions in this project allow for Monte Carlo simulations to be performed on stocks for which a 
trading signal has been generated.  Specifically, if there are more tradings signals generated than desired
holdings, a random selection of available stocks are selected.  

This could result in excessive turnover, 100% turnover in some circumstances.  To mitigate this, a rule is enforced such that should a stock be held in period t, and it remains buy signal in period t+1, it will not be disposed of.  

## Performance
The functions has been written in NumPy and acclerated with Numba.  

Some issues encountered with Numba:  

- Numba supports only one advanced index.  This requires implementing loops as opposed to vectorised operations in certain circumstances.  
For example this code...
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

- Numba does not support the ```np.c_``` function.  ```np.concatenate``` is a replacement for this.  

## Other issues  
- Conda environment and debugging.  It turns out NumPy does not import correctly when using the Visual Studio code debugger.  Workaround [here](https://github.com/microsoft/vscode-python/issues/13500)