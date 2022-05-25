# Note 1 : Function not supported by Numba
# Note 2 : "NotImplementedError: only one advanced index supported"


# Imports
import numpy as np
import pandas as pd
from numba import jit
from utils import intersect, setdiff, union, max_dd


# Backtest function (numba - nopython)
# -------------------------------------------------------------------------------------------------

@jit(nopython=True)
def monte_carlo_backtest(
  prices, 
  positions, 
  seed_capital, 
  max_positions, 
  rndm = False,
  verbose = False
  ):

  """Calculate monthly portfolio returns and calculate summary performance measures
  for the period provided.  Portfolio selection can be fully randomised or partially randomised based on 
  position entry provided.

  Arguments
  ---------
  prices : array
    Stock prices.  Columns represent stocks and rows months.  Must be fully populated and
    not contain zeroes or null values
  
  positions : array
    Represent the positions to be entered into.  Valid values are 0 and 1.  
    1 = enter position.  Must be same shape as 'prices'.
  
  seed_capital : integer
    Initial portfolio capital.

  max_positions : integer
    The maximum number of stocks to be held at any time. 
  
  rndm : boolean
    True = randomly chose positions to enter ignoring those flagged in 'positions'
    False = chose positions to enter per that flagged in 'positions'
  """


  # Error capture
  # -------------

  if 0 in prices:
    raise ValueError("The price data contains zeros.  This will trigger a division by zero.")

  if prices.shape != positions.shape:
    raise ValueError("The price data and position data must be of the same shape.")

  for r in range(positions.shape[0]):
    row = positions[r,:]
    if np.sum(row) == 0:
      raise ValueError("At least one row of the 'positions' data has no trade signal")
  

  # Convert types for Numba compatability
  # -------------------------------------
  max_positions_flt = np.array([max_positions], dtype=np.float64)

  # Create array for holdings of open positions (signified with 1), this creates as float64
  holding = np.zeros(shape = positions.shape)
  holding = np.concatenate((holding, np.ones((positions.shape[0],1))), axis=1)
  
  # Seed opening cash
  holding[0,-1] = seed_capital

  # Create array for position valuation
  valuation = np.zeros(shape = positions.shape)
  valuation = np.concatenate((valuation, np.ones((positions.shape[0],1))), axis=1)

  # Add column for cash to prices array
  prices = np.concatenate((prices, np.ones((positions.shape[0],1))), axis=1)

  # Loop over all rows
  for r in range(positions.shape[0]):
    
    # Store prior (open) positions in array
    if r == 0:
      prior_pos = np.empty(0, dtype=np.float64)
    else:
      prior_pos = holding[r-1,:-1]
    
    # Index of open positions
    open_positions_idx = np.nonzero(prior_pos)[0]

    # Capture revaluation of opening positions
    price_change = (prices[r,:] - prices[r-1,:]) 
    if r == 0:
      open_positions_reval = 0 
    else:
      open_positions_reval = 0
      for i in open_positions_idx:
        open_positions_reval = open_positions_reval + (price_change[i] * holding[r-1,i])


    # Current prospective positions available to open (this will never be None)
    # -------------------------------------------------------------------------
    # TO DO - if a stock is no longer available to trade it should be excluded from the 
    # population available for selection. How to do this?
    if rndm:
      available_positions_idx = np.random.choice(np.arange(positions.shape[1]), size=max_positions, replace=False)
    else:
      available_positions_idx = np.nonzero(positions[r,:])[0]


    # If the current available open positions (ie. those with a buy signal) are the    
    # same stocks as held in the prior period, continue to hold those stocks
    # -----------------------------------------------------------------------------
    hold_position_idx = intersect(open_positions_idx, available_positions_idx)


    # Index of positions that are available to sample.  Stocks with a buy signal, ex those 
    # that are already held and remain a buy per above, ie. in  'hold_position_idx'
    # ------------------------------------------------------------------------------------
    avail_position_to_sample_idx = setdiff(available_positions_idx, hold_position_idx)


    # Set the number of stocks to sample, minimum of max positions and stocks available to sample (this coud be nil) 
    adj_sample_size = min(max_positions - len(hold_position_idx), len(avail_position_to_sample_idx))

    # Randomly choose one of the available positions to enter
    if adj_sample_size > 0 and len(avail_position_to_sample_idx) > 0:
      sample_idx = np.random.choice(avail_position_to_sample_idx, size=adj_sample_size, replace=False)
    else:
      sample_idx = np.random.choice(avail_position_to_sample_idx, size=0, replace=False)
    
    # Join positions derived from hold and sample
    c = union(sample_idx, hold_position_idx)

    # Assign positions indicator to holding matrix
    for i in c:
      holding[r,i] = 1

    # Create array for current versus prior holdings
    if r == 0:
      prior_holding = np.zeros(shape = positions.shape[1])
    else:
      prior_holding = holding[r-1,:-1]

    current_holding = holding[r,:-1]
    compare_holding = np.stack((prior_holding,current_holding))

    # Index of sales, prior holding is 1 and current 0 (purchases are in "sample_idx")
    sales_idx = np.where((compare_holding[0] != 0) & (compare_holding[1] == 0))[0]

    # Carry forward holding from prior period
    for i in hold_position_idx:
      holding[r,i] = holding[r-1,i]

    open_cash = holding[r,-1]

    # Update cash balance for sales 
    if len(sales_idx) != 0:
      proceed_sale = 0
      for i in sales_idx:
        proceed_sale = proceed_sale + (prices[r,i] * holding[r-1,i])
    else:
      proceed_sale = 0
    holding[r,-1] = holding[r,-1] + proceed_sale

    # Size purchases
    cash_avail = holding[r,-1]
    if len(sample_idx) != 0:
      for i in sample_idx:
        new_hold = holding[r,i] * cash_avail / max_positions_flt / prices[r,i] 
        new_hold = np.floor(new_hold)
        holding[r,i] = new_hold[0]
    
    # Update cash balance for purchases
    if len(sample_idx) != 0:
      cost_purch = 0
      for i in sample_idx:
        cost_purch = cost_purch + (holding[r,i] * prices[r,i])
    else:
      cost_purch = 0
    holding[r,-1] = holding[r,-1] - cost_purch

    # Roll forward cash
    if r < positions.shape[0]-1:
      holding[r+1,-1] = holding[r,-1]

    # Assign valuation 
    valuation[r,:] = prices[r,:] * holding[r,:]

    if verbose:
      print('LOOP: ',r)
      print('Cash')
      print('1a. open cash: '.ljust(30), open_cash)
      print('1b. sale proceeds: '.ljust(30), proceed_sale)
      print('1c. cost purchased: '.ljust(30), cost_purch)
      print('1d. closing cash: '.ljust(30), holding[r,-1])
      print('\n')
      print('Holdings index')
      print('2a. open holding: '.ljust(30), open_positions_idx) # np.sort(open_positions_idx)
      print('2b. available: '.ljust(30), available_positions_idx)
      print('2c. retain: '.ljust(30), hold_position_idx)
      print('2d. sell: '.ljust(30), sales_idx)
      print('2e. available to purchase: '.ljust(30), avail_position_to_sample_idx)
      print('2f. purchased: '.ljust(30), sample_idx)
      print('2g. closing holding: '.ljust(30), c)
      print('\n')
      print('Valuation')
      print('3a. opening value: '.ljust(30), np.sum(valuation[r-1,:]))
      print('3b. revaluation: '.ljust(30), open_positions_reval)
      print('3c. closing value: '.ljust(30), np.sum(valuation[r,:]))
      print('\n')
  

  # Performance metrics
  # -------------------

  # Portfolio valuation
  portfolio_vltn = np.sum(valuation, axis=1)

  # Maximum drawdown percentage
  max_drawdown = max_dd(portfolio_vltn)

  # Compound monthly growth rate
  cmgr = np.power(portfolio_vltn[-1] / portfolio_vltn[0], 1 / (len(portfolio_vltn) - 1)) - 1

  # Annual growth rate
  cagr = np.power(1 + cmgr, 12) - 1

  # Volatility
  volatility = np.std(np.ediff1d(np.log(portfolio_vltn))) * np.sqrt(12)

  return max_drawdown, cagr, volatility, portfolio_vltn















# Backtest function (python)
# -------------------------------------------------------------------------------------------------

def monte_carlo_backtest_np(
  prices, 
  positions, 
  seed_capital, 
  max_positions, 
  rndm = False,
  verbose = False
  ):

  """Calculate monthly portfolio returns and calculate summary performance measures
  for the period provided.  Portfolio selection can be fully randomised or partially randomised based on 
  position entry provided.

  Arguments
  ---------
  prices : array
    Stock prices.  Columns represent stocks and rows months.  Must be fully populated and
    not contain zeroes or null values
  
  positions : array
    Represent the positions to be entered into.  Valid values are 0 and 1.  
    1 = enter position.  Must be same shape as 'prices'.
  
  seed_capital : integer
    Initial portfolio capital.

  max_positions : integer
    The maximum number of stocks to be held at any time. 
  
  rndm : boolean
    True = randomly chose positions to enter ignoring those flagged in 'positions'
    False = chose positions to enter per that flagged in 'positions'

  verbose : boolean
    Print transaction details
  """


  # Error capture
  # -------------

  if 0 in prices:
    raise ValueError("The price data contains zeros.  This will trigger a division by zero.")

  if prices.shape != positions.shape:
    raise ValueError("The price data and position data must be of the same shape.")

  for r in range(positions.shape[0]):
    row = positions[r,:]
    if np.sum(row) == 0:
      raise ValueError("At least one row of the 'positions' data has no trade signal")


  # Create array for holdings of open positions (signified with 1) and valuation thereof
  holding = np.zeros(shape = positions.shape)
  holding = np.concatenate((holding, np.ones((positions.shape[0],1))), axis=1)
  
  # Seed opening cash
  holding[0,-1] = seed_capital

  # Create array for position valuation
  valuation = np.zeros(shape = positions.shape)
  valuation = np.concatenate((valuation, np.ones((positions.shape[0],1))), axis=1)

  # Add column for cash to prices array
  prices = np.concatenate((prices, np.ones((positions.shape[0],1))), axis=1)

  # Loop over all rows
  for r in range(positions.shape[0]):
    
    # Store prior (open) positions in array
    if r == 0:
      prior_pos = None
    else:
      prior_pos = holding[r-1,:-1]
    
    # Index of open positions
    open_positions_idx = np.nonzero(prior_pos)[0]

    # Capture revaluation of opening positions
    price_change = (prices[r,:] - prices[r-1,:]) 
    if r == 0:
      open_positions_reval = 0 
    else:
      open_positions_reval = np.sum(price_change[open_positions_idx] * holding[r-1,open_positions_idx])


    # Current prospective positions available to open (this will never be None)
    # -------------------------------------------------------------------------

    # TO DO - if a stock is no longer available to trade it should be excluded from the 
    # population available for selection. How to do this?
    if rndm:
      available_positions_idx = np.random.choice(np.arange(positions.shape[1]), size=max_positions, replace=False)
    else:
      available_positions_idx = np.nonzero(positions[r,:])[0]


    # If the current available open positions (ie. those with a buy signal) are the    
    # same stocks as held in the prior period, continue to hold that those stocks
    # -----------------------------------------------------------------------------
    hold_position_idx = np.intersect1d(open_positions_idx, available_positions_idx)


    # Index of positions that are available to sample.  Stocks with a buy signal, ex those 
    # that are already held and remain a buy per above, ie. in  'hold_position_idx'
    # ------------------------------------------------------------------------------------
    avail_position_to_sample_idx = np.setdiff1d(available_positions_idx, hold_position_idx)

    # Set the number of stocks to sample, minimum of max positions and stocks available to sample (this coud be nil) 
    adj_sample_size = min(max_positions - len(hold_position_idx), len(avail_position_to_sample_idx))

    # Randomly choose one of the available positions to enter
    if adj_sample_size > 0 and len(avail_position_to_sample_idx) > 0:
      sample_idx = np.random.choice(avail_position_to_sample_idx, size=adj_sample_size, replace=False)
    else:
      sample_idx = None
   
    # Join positions derived from hold and sample
    if sample_idx is not None:
      c = np.concatenate((sample_idx, hold_position_idx))
    else:
      c = hold_position_idx

    # Assign positions indicator to holding matrix
    holding[r,c] = 1

    # Create array for current versus prior holdings
    if r == 0:
      prior_holding = np.zeros(shape = positions.shape[1])
    else:
      prior_holding = holding[r-1,:-1]

    current_holding = holding[r,:-1]
    compare_holding = np.stack((prior_holding,current_holding))

    # Index of sales, prior holding is 1 and current 0 (purchases are in "sample_idx")
    sales_idx = np.where((compare_holding[0] != 0) & (compare_holding[1] == 0))[0]

    # Carry forward holding from prior period
    holding[r,hold_position_idx] = holding[r-1,hold_position_idx]

    open_cash = holding[r,-1]

    # Update cash balance for sales 
    if sales_idx is not None:
      proceed_sale = sum(prices[r,sales_idx] * holding[r-1,sales_idx])
    else:
      proceed_sale = 0
    holding[r,-1] = holding[r,-1] + proceed_sale

    # Size purchases
    cash_avail = holding[r,-1]
    if sample_idx is not None:
      holding[r,sample_idx] = np.floor(holding[r,sample_idx] * cash_avail / max_positions / prices[r,sample_idx])
    # Update cash balance for purchases
    if sample_idx is not None:
      cost_purch = np.sum(holding[r,sample_idx] * prices[r,sample_idx])
    else:
      cost_purch = 0
    holding[r,-1] = holding[r,-1] - cost_purch

    # Roll forward cash
    if r < positions.shape[0]-1:
      holding[r+1,-1] = holding[r,-1]

    # Assign valuation 
    valuation[r,:] = prices[r,:] * holding[r,:]

    if verbose:
      print('LOOP: ',r)
      print('Cash')
      print('1a. open cash: '.ljust(30), np.round_(open_cash,1))
      print('1b. sale proceeds: '.ljust(30), np.round_(proceed_sale,1))
      print('1c. cost purchased: '.ljust(30), np.round_(cost_purch,1))
      print('1d. closing cash: '.ljust(30), np.round_(holding[r,-1],1))
      print('\n')
      print('Holdings index')
      print('2a. open holding: '.ljust(30), open_positions_idx) # np.sort(open_positions_idx)
      print('2b. available: '.ljust(30), available_positions_idx)
      print('2c. retain: '.ljust(30), hold_position_idx)
      print('2d. sell: '.ljust(30), sales_idx)
      print('2e. available to purchase: '.ljust(30), avail_position_to_sample_idx)
      print('2f. purchased: '.ljust(30), sample_idx)
      print('2g. closing holding: '.ljust(30), c)
      print('\n')
      print('Valuation')
      print('3a. opening value: '.ljust(30), np.round_(np.sum(valuation[r-1,:]),1))
      print('3b. revaluation: '.ljust(30), [np.round_(open_positions_reval,1), price_change[open_positions_idx], holding[r-1,open_positions_idx]])
      print('3c. closing value: '.ljust(30), np.round_(np.sum(valuation[r,:]),1))
      print('\n')
  

  # Performance metrics
  # -------------------

  # Portfolio valuation
  portfolio_vltn = np.sum(valuation, axis=1)

  # Maximum drawdown percentage
  max_drawdown = np.max((np.maximum.accumulate(portfolio_vltn, axis=0) - portfolio_vltn) / np.maximum.accumulate(portfolio_vltn, axis=0))

  # Compound monthly growth rate
  cmgr = np.power(portfolio_vltn[-1] / portfolio_vltn[0], 1 / (len(portfolio_vltn) - 1)) - 1

  # Annual growth rate
  cagr = np.power(1 + cmgr, 12) - 1

  # Volatility
  volatility = np.std(np.ediff1d(np.log(portfolio_vltn))) * np.sqrt(12)
  
  return max_drawdown, cagr, volatility, portfolio_vltn









# Loop over backtest function to create list of attributes
# -------------------------------------------------------------------------------------------------
def monte_carlo_backtest1(
  prices, 
  positions, 
  seed_capital, 
  max_positions, 
  iter = 1000,
  rndm = False
  ):

  dd = []
  cagr = []
  vol = []
  ptf_val_ts = []
  for i in range(iter):
    result = monte_carlo_backtest(
      prices=prices, 
      positions=positions, 
      seed_capital=seed_capital, 
      max_positions=max_positions,
      rndm=rndm,
      verbose = False
      )
    
    dd.append(result[0])
    cagr.append(result[1])
    vol.append(result[2])
    ptf_val_ts.append(result[3])

  # Return dataframe
  df = pd.DataFrame({
    'max_drawdown': dd, 
    'cagr': cagr, 
    'volatility': vol,
    'portfolio_valuation_ts': ptf_val_ts
    })
  
  return df