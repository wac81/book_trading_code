import pandas as pd
from numpy import sqrt,mean,log,diff
from scipy import stats
import numpy as np
import quandl
quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'
goog_table = quandl.get('WIKI/GOOG')
# 取google 2016年1月到8月每日股票收盘数据
close = goog_table['2016-01':'2016-08']['Adj. Close']
returns = goog_table['2016-01':'2016-08']["Adj. Close"]/goog_table['2016-01':'2016-08']["Adj. Close"].shift(1)



def historical_vol(close: list):
    r = diff(log(close))
    r_mean = mean(r)
    diff_square = [(r[i]-r_mean)**2 for i in range(0,len(r))]
    std = sqrt(sum(diff_square)*(1.0/(len(r)-1)))
    vol = std*sqrt(252)
    return vol

print(historical_vol(close))



def get_realized_vol(returns, time):
    returns_log = np.log(returns)
    data = pd.Series(returns_log, index=None)
    data.fillna(0, inplace = True)
    #window/time tells us how many days out vol you want. ~22 = 1 month out vol (~21 trading days in a month)
    #we do this so we can match up with the vix which is the 30 day out (~21 trading day) calculated vol
    volatility = data.rolling(window=time).std(ddof=0)*np.sqrt(252)
    return volatility

#假设每个月有22天
print(get_realized_vol(returns, 22))



def bsm_price(option_type, sigma, s, k, r, T, q):
    # calculate the bsm price of European call and put options
    sigma = float(sigma)
    d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'c':
        price = np.exp(-r*T) * (s * np.exp((r - q)*T) * stats.norm.cdf(d1) - k *  stats.norm.cdf(d2))
        return price
    elif option_type == 'p':
        price = np.exp(-r*T) * (k * stats.norm.cdf(-d2) - s * np.exp((r - q)*T) *  stats.norm.cdf(-d1))
        return price
    else:
        print('No such option type %s') %option_type
def implied_vol(option_type, option_price, s, k, r, T, q):
    # apply bisection method to get the implied volatility by solving the BSM function
    precision = 0.00001
    upper_vol = 500.0
    max_vol = 500.0
    min_vol = 0.0001
    lower_vol = 0.0001
    iteration = 0

    while 1:
        iteration +=1
        mid_vol = (upper_vol + lower_vol)/2.0
        price = bsm_price(option_type, mid_vol, s, k, r, T, q)
        if option_type == 'c':

            lower_price = bsm_price(option_type, lower_vol, s, k, r, T, q)
            if (lower_price - option_price) * (price - option_price) > 0:
                lower_vol = mid_vol
            else:
                upper_vol = mid_vol
            if abs(price - option_price) < precision: break 
            if mid_vol > max_vol - 5 :
                mid_vol = 0.000001
                break

        elif option_type == 'p':
            upper_price = bsm_price(option_type, upper_vol, s, k, r, T, q)

            if (upper_price - option_price) * (price - option_price) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
            if abs(price - option_price) < precision: break 
            if iteration > 50: break

    return mid_vol
mvol = implied_vol('c', 0.3, 3, 3, 0.032, 30.0/365, 0.01)
print(mvol)