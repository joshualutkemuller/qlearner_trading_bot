""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data  		  	   		 	 	 			  		 			 	 	 		 		 	

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904051695  # replace with your GT ID number

def study_group():
    """
    Returns
        A comma separated string of GT_Name of each member of your study group
        # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

    Return type
        str
    """
    return "jlutkemuller3"

def author():
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """
    return "jlutkemuller3"  # Change this to your user ID

def order_reader(order_file):

    """
    Read in an order file and return an ordered dataframe (default csv).

    """
    return pd.read_csv(order_file,parse_dates=['Date'],index_col='Date').sort_index()

def holdings_dataframe_builder(TRADES_DATAFRAME, starting_value):
    """
    Function to generate cumulative sum of shares
    held by the end of the day and starting with initial cash
    """

    return TRADES_DATAFRAME.cumsum()

def trader_dataframe_builder(order_df,prices, commission, impact):
    """
    Take parameters of order dataframe, prices dataframe,
    commission and impact to build a dataframe of trades over time
    """

    # initialize a dataframe of zeroes with initial columns from prices
    TRADES_DATAFRAME = pd.DataFrame(0, index=prices.index, columns = prices.columns)

    # iterate through each date and order row
    for date, order in order_df.iterrows():

        symbol_var = order['Symbol']
        sign_direction = 1 if order['Order'].upper() == 'BUY' else -1
        share_count_var = order['Shares']
        price_var = prices.loc[date, symbol_var]

        # Record trade for the particular date (allow for the potential for multiple trades in a single day for a certain symbol)
        TRADES_DATAFRAME.loc[date, symbol_var] += sign_direction * share_count_var

        # Calculate the cash change
        cost_of_trade = sign_direction * share_count_var * price_var * (1 + sign_direction * impact)

        # Update the `trades_df` with cost of the commission and the cost of the trade
        TRADES_DATAFRAME.loc[date, 'CASH'] -=  cost_of_trade
        TRADES_DATAFRAME.loc[date, 'CASH'] -= commission

    return TRADES_DATAFRAME


def compute_portvals(
    trades_df,
    symbol='JPM',
    start_val=1000000,
    commission=0.0,
    impact=0.0 		  	   		 	 	 			  		 			 	 	 		 		 	):
    """
    Computes the portfolio values.

    :param trades_df: Trades DataFrame
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your

    symbol = trades_df.columns[0]
    dates = trades_df.index
    prices = get_data([symbol], dates)
    prices = prices[[symbol]].copy()
    prices['CASH'] = 1.0

    # fill
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    # build trades df
    df_trades = pd.DataFrame(0.0, index=dates, columns=[symbol, 'CASH'])
    df_trades[symbol] = trades_df[symbol]

    # cash flows
    df_trades['CASH'] = - trades_df[symbol] * prices[symbol] * (1 + impact) - commission * (trades_df[symbol] != 0)
    # inject cash
    df_trades.loc[dates[0], 'CASH'] += start_val
    holdings = df_trades.cumsum()
    portvals = (holdings * prices).sum(axis=1)

    return portvals.to_frame(name='Portvals')


if __name__ == "__main__":

    pass