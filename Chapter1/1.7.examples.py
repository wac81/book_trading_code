import sys   
sys.setrecursionlimit(100000)
import pandas as pd
import numpy as np
import cvxpy as cp

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt import CLA
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from pypfopt import plotting
import os

stock_dir = '../data/stock_days_fq/'
saved_csv = 'stock.csv'
raw_df = None 
for stock in os.listdir(stock_dir)[:10]:
    df = pd.read_csv(stock_dir + stock)
    # if len(df) < 2200:
    #     continue
    
    close = df['close'].values
    if raw_df is None:
        raw_df = pd.DataFrame({'date':df['date'].values, stock:close})
    else:
        temp_df = pd.DataFrame({'date':df['date'].values, stock:close})

        raw_df = pd.merge(raw_df, temp_df, how='outer', on=['date'])

raw_df.sort_values("date",inplace=True)
raw_df.to_csv(saved_csv,index=False)


# Reading in the data; preparing expected returns and a risk model
df = pd.read_csv(saved_csv, parse_dates=True, index_col="date")
returns = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)
# mu = expected_returns.ema_historical_return(df)
S = risk_models.sample_cov(df)


# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
ef.portfolio_performance(verbose=True)
items = sorted(cleaned_weights.items(), key=lambda obj: obj[1], reverse=True)
for i in items[:100]:
    print(i)




from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))




# Now try with a nonconvex objective from  Kolm et al (2014)
def deviation_risk_parity(w, cov_matrix):
    diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
    return (diff ** 2).sum().sum()


ef = EfficientFrontier(mu, S)
weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
ef.portfolio_performance(verbose=True)
items = sorted(weights.items(), key=lambda obj: obj[1], reverse=True)
for i in items[:100]:
    print(i)

"""
Expected annual return: 22.9%
Annual volatility: 19.2%
Sharpe Ratio: 1.09
"""

# Hierarchical risk parity
hrp = HRPOpt(returns)
weights = hrp.optimize()
hrp.portfolio_performance(verbose=True)
items = sorted(weights.items(), key=lambda obj: obj[1], reverse=True)
for i in items[:100]:
    print(i)

import csv
with open('mycsvfile.csv','w') as f:
    w = csv.writer(f)
    for i in items:
        w.writerow(i)

# print(weights)
plotting.plot_dendrogram(hrp,filename='dendrogram.jpg')  # to plot dendrogram



#################################################################################################################################

