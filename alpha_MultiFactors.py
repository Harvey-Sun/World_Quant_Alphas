# -*- coding:utf-8 -*-

from CloudQuant import MiniSimulator
import numpy as np
import pandas as pd

username = 'Harvey_Sun'
password = 'P948894dgmcsy'
Strategy_Name = 'WQ_alpha#MF_05_100'

INIT_CAP = 100000000
START_DATE = '20130101'
END_DATE = '20161231'
Fee_Rate = 0.001
program_path = 'C:/cStrategy/'
buy_number = 100
period = 5
s_or_b = False  # true表示选因子较小的股票，false表示选因子较大的股票


def correlation(x, y):
    upside = ((x - x.mean()) * (y - y.mean())).sum(skipna=False)
    dnside = x.std() * y.std()
    result = upside / dnside
    return result


def winsorize(x):
    sigma = x.std()
    m = x.mean()
    x[x > m + 3 * sigma] = x + 3 * sigma
    x[x < m - 3 * sigma] = x - 3 * sigma
    return x


def standardize(x):
    return (x - x.mean()) / x.std()


def initial(sdk):
    sdk.prepareData(['LZ_GPA_QUOTE_THIGH', 'LZ_GPA_QUOTE_TVOLUME',
                     'LZ_GPA_QUOTE_TLOW', 'LZ_GPA_QUOTE_TCLOSE', 'LZ_GPA_QUOTE_TVALUE',
                     'LZ_GPA_CMFTR_CUM_FACTOR', 'LZ_GPA_SLCIND_STOP_FLAG'])
    stock_pool = []
    sdk.setGlobal('stock_pool', stock_pool)
    step = period
    sdk.setGlobal('step', step)


def init_per_day(sdk):
    step = sdk.getGlobal('step')
    if step == period:
        stock_list = sdk.getStockList()
        high = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_THIGH')[-7:], columns=stock_list)
        low = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TLOW')[-7:], columns=stock_list)
        close = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TCLOSE')[-7:], columns=stock_list)
        volume = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TVOLUME')[-7:], columns=stock_list)
        value = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TVALUE')[-7:], columns=stock_list)
        adj_f = pd.DataFrame(sdk.getFieldData('LZ_GPA_CMFTR_CUM_FACTOR')[-7:], columns=stock_list)
        stop = pd.DataFrame(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-8:], columns=stock_list)
        volume.replace(0, np.nan, inplace=True)
        adj_high = high * adj_f
        adj_low = low * adj_f
        adj_close = close * adj_f
        adj_volume = volume / adj_f
        # 计算昨天的#83号factor
        temp = (adj_high - adj_low) / adj_close.rolling(window=5).mean()
        volume_rank = adj_volume.rank(axis=1, pct=True)
        vwap = value / adj_volume
        upside = temp.diff(2).rank(axis=1, pct=True).iloc[-1] * volume_rank.iloc[-1]
        dnside = (temp / (vwap - adj_close)).iloc[-1]
        factor83 = upside / dnside
        factor83[adj_close.isnull().any()] = np.nan
        factor83.replace([np.inf, -np.inf], 0, inplace=True)
        # 计算昨日的#41号factor
        factor41 = (((adj_high * adj_low) ** 0.5 - (value / adj_volume) * 10) / adj_close).iloc[-1]  # 计算昨天的alpha41
        factor41 = -1 * np.abs(factor41 - factor41.mean())
        # 处理因子并组合
        factor = standardize(winsorize(factor83)) + standardize(winsorize(factor41))
        # 选出排序靠前的股票
        factor[stop.notnull().any()] = np.nan  # 剔除停牌的股票
        factor.sort_values(ascending=s_or_b, inplace=True)
        stock_pool = factor.index[:buy_number]
        # 选出当天应该调仓的股票
        position = sdk.getPositions()
        position_dict = dict([i.code, i.optPosition] for i in position)
        stock_to_buy = set(stock_pool) - set(position_dict.keys())
        stock_to_sell = set(position_dict.keys()) - set(stock_pool)
        sdk.setGlobal('position_dict', position_dict)
        sdk.setGlobal('stock_to_buy', stock_to_buy)
        sdk.setGlobal('stock_to_sell', stock_to_sell)
    else:
        pass


def strategy(sdk):
    step = sdk.getGlobal('step')
    if step == period:
        step = 1
        today = sdk.getNowDate()
        sdk.sdklog(today, '====================')

        position_dict = sdk.getGlobal('position_dict')
        stock_to_buy = sdk.getGlobal('stock_to_buy')
        stock_to_sell = sdk.getGlobal('stock_to_sell')
        quotes = sdk.getQuotes(list(stock_to_buy | stock_to_sell))

        sell_orders = []
        for stock in stock_to_sell:
            if stock in quotes.keys():
                price = quotes[stock].open
                volume = position_dict[stock]
                order = [stock, price, volume, -1]
                sell_orders.append(order)
        if sell_orders:
            sdk.makeOrders(sell_orders)
            sdk.sdklog('===sell orders===')
            sdk.sdklog(np.array(sell_orders))
        available_cash = sdk.getAccountInfo().availableCash
        available_cash_one_stock = available_cash / len(stock_to_buy)
        buy_orders = []
        for stock in stock_to_buy:
            if stock in quotes.keys():
                price = quotes[stock].open
                volume = int(available_cash_one_stock / (price * 100)) * 100
                if volume > 0:
                    order = [stock, price, volume, 1]
                    buy_orders.append(order)
        if buy_orders:
            sdk.makeOrders(buy_orders)
            sdk.sdklog('===buy orders===')
            sdk.sdklog(np.array(buy_orders))
    else:
        step += 1
    sdk.setGlobal('step', step)


config = {
    'username': username,
    'password': password,
    'initCapital': INIT_CAP,
    'startDate': START_DATE,
    'endDate': END_DATE,
    'strategy': strategy,
    'initial': initial,
    'preparePerDay': init_per_day,
    'feeRate': Fee_Rate,
    'strategyName': Strategy_Name,
    'logfile': '%s.log' % Strategy_Name,
    'rootpath': program_path,
    'executeMode': 'D',
    'feeLimit': 5,
    'cycle': 10,
    'dealByVolume': True,
    'allowForTodayFactors': ['LZ_GPA_SLCIND_STOP_FLAG']
}

if __name__ == "__main__":
    # 在线运行所需代码
    import os

    config['strategyID'] = os.path.splitext(os.path.split(__file__)[1])[0]
    MiniSimulator(**config).run()