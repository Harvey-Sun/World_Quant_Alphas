# -*- coding:utf-8 -*-

from CloudQuant import MiniSimulator
import numpy as np
import pandas as pd

username = 'Harvey_Sun'
password = 'P948894dgmcsy'
Strategy_Name = 'WQ_alpha#41'

INIT_CAP = 100000000
START_DATE = '20130101'
END_DATE = '20161231'
Fee_Rate = 0.001
program_path = 'C:/cStrategy/'
buy_number = 100


def initial(sdk):
    sdk.prepareData(['LZ_GPA_QUOTE_THIGH', 'LZ_GPA_QUOTE_TLOW', 'LZ_GPA_QUOTE_TCLOSE',
                     'LZ_GPA_QUOTE_TVALUE', 'LZ_GPA_QUOTE_TVOLUME', 'LZ_GPA_SLCIND_STOP_FLAG'])
    stock_pool = []
    sdk.setGlobal('stock_pool', stock_pool)


def init_per_day(sdk):
    stock_list = sdk.getStockList()
    high = pd.Series(sdk.getFieldData('LZ_GPA_QUOTE_THIGH')[-1], index=stock_list)
    low = pd.Series(sdk.getFieldData('LZ_GPA_QUOTE_TLOW')[-1], index=stock_list)
    close = pd.Series(sdk.getFieldData('LZ_GPA_QUOTE_TCLOSE')[-1], index=stock_list)
    value = pd.Series(sdk.getFieldData('LZ_GPA_QUOTE_TVALUE')[-1], index=stock_list)
    volume = pd.Series(sdk.getFieldData('LZ_GPA_QUOTE_TVOLUME')[-1], index=stock_list)
    stop = pd.Series(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-1], index=stock_list)
    alpha_41 = ((high * low) ** 0.5 - (value / volume) * 10) / close  # 计算昨天的alpha41
    alpha_41 = pd.Series(np.where(stop.isnull(), alpha_41, np.nan), index=stock_list)  # 剔除今天停牌的股票
    alpha_41.sort_values(ascending=True, inplace=True)
    stock_pool = alpha_41.index[:buy_number]  # 选出alpha41因子较小的股票
    position = sdk.getPositions()
    position_dict = dict([i.code, i.optPosition] for i in position)
    stock_to_buy = set(stock_pool) - set(position_dict.keys())
    stock_to_sell = set(position_dict.keys()) - set(stock_pool)
    sdk.setGlobal('stock_to_buy', stock_to_buy)
    sdk.setGlobal('stock_to_sell', stock_to_sell)


def strategy(sdk):
    today = sdk.getNowDate()
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
    'cycle': 1,
    'dealByVolume': True,
    'allowForTodayFactors': ['LZ_GPA_SLCIND_STOP_FLAG']
}

if __name__ == "__main__":
    # 在线运行所需代码
    import os

    config['strategyID'] = os.path.splitext(os.path.split(__file__)[1])[0]
    MiniSimulator(**config).run()