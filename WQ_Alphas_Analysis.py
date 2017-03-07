# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:38:39 2017

@author: LZJF
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
# import statsmodels.api as sm
import os
mpl.style.use('ggplot')

os.chdir('C:\cStrategy\Factor')


def import_data(name):
    data = pd.read_csv('%s.csv' % name, index_col=0)
    data.index = pd.to_datetime([str(i) for i in data.index])
    return data


def pearson(x, y, skipna=False):
    upside = ((x - x.mean()) * (y - y.mean())).sum(skipna=skipna)
    downside = np.sqrt((np.square(x - x.mean()).sum()) *
                       (np.square(y - y.mean()).sum(skipna=skipna)))
    return upside / downside


def save_factor(factor, factor_id, no_zdt):
    if no_zdt:
        factor_name = 'alpha#%s_no_zdt_1D' % factor_id
    else:
        factor_name = 'alpha#%s_origin_1D' % factor_id
    factor.index.name = factor_name
    to_file = 'F:\Strategies\World_Quant_Alphas/#%s' % factor_id
    if not os.path.exists(to_file):
        os.mkdir(to_file)
    else:
        pass
    factor.to_csv(to_file + '/%s.csv' % factor_name)
    print '保存完成'


def rolling_stop(stop, window):
    temp = pd.DataFrame(index=stop.columns)
    for i in stop.index:
        one = stop[:i][-window:].notnull().any(axis=0)
        temp[i] = one
    return temp.T


def rolling_rank(data, window):
    temp = pd.DataFrame(index=data.columns)
    for i in data.index:
        one = data[:i][-window:].rank(axis=0, pct=True).iloc[-1]
        temp[i] = one
    return temp.T


def rolling_null(data, window):
    temp = pd.DataFrame(index=data.columns)
    for i in data.index:
        one = data[:i][-window:].isnull().any(axis=0)
        temp[i] = one
    return temp.T


def linear_decay(data, window):
    result = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    weight = np.arange(window) + 1.
    weight = weight / weight.sum()
    for i in range(window, data.shape[0]):
        t = data.index[i]
        result.ix[t, :] = data[i-window:i].T.dot(weight)
    return result


def ind_neutral(data, ind):
    temp = pd.DataFrame(index=data.columns)
    for i in data.index:
        temp[i] = data.ix[i].groupby(ind.ix[i]).apply(lambda x: x - x.mean())
    return temp.T


def factor_cal(data, factor_id, no_zdt=False, no_st=False, no_new=False, save=False):
    high = data['high']
    low = data['low']
    close = data['close']
    OPEN = data['OPEN']
    value = data['value']
    volume = data['volume']
    volume.replace(0, np.nan, inplace=True)
    cap = data['cap']
    stop = data['stop']
    st = data['st']
    ind = data['ind']
    fcf = data['fcf']
    asset_debt = data['asset_debt']
    price_adj_f = data['price_adj_f']

    # 将新股（上市小于63天）的所有数据设为na
    if no_new:
        high[rolling_null(close, 63)] = np.nan
        low[rolling_null(close, 63)] = np.nan
        close[rolling_null(close, 63)] = np.nan
        OPEN[rolling_null(close, 63)] = np.nan
        value[rolling_null(close, 63)] = np.nan
        volume[rolling_null(close, 63)] = np.nan
        cap[rolling_null(close, 63)] = np.nan
    
    adj_OPEN = OPEN * price_adj_f
    adj_close = close * price_adj_f
    adj_high = high * price_adj_f
    adj_low = low * price_adj_f
    adj_volume = volume / price_adj_f
    returns = adj_close.pct_change()
    vwap = value / adj_volume
    
    print '开始计算%s号因子' % factor_id
    if factor_id == '001':
        returns[returns < 0] = returns.rolling(window=20).std(skipna=False)
        ts_argmax = np.square(returns).rolling(window=5).apply(np.argmax) + 1
        ts_argmax[rolling_null(close, 5)] = np.nan
        ts_argmax[rolling_stop(stop, 20)] = np.nan
        factor = ts_argmax.rank(axis=1, pct=True) - 0.5
    elif factor_id == '002':
        temp1 = np.log(adj_volume).diff(2)
        temp1_rank = temp1.rank(axis=1, pct=True)
        temp2 = (close - OPEN) / OPEN
        temp2_rank = temp2.rank(axis=1, pct=True)
        factor = -1 * temp1_rank.rolling(window=6).corr(temp2_rank)
        factor[rolling_null(close, 6)] = np.nan
        factor[rolling_stop(stop, 6)] = np.nan
    elif factor_id == '003':
        open_rank = OPEN.rank(axis=1, pct=True)
        volume_rank = volume.rank(axis=1, pct=True)
        factor = -1 * open_rank.rolling(window=10).corr(volume_rank)
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '004':
        low_rank = adj_low.rank(axis=1, pct=True)
        factor = -1 * rolling_rank(low_rank, 9)
        factor[rolling_null(close, 9)] = np.nan
        factor[rolling_stop(stop, 9)] = np.nan
    elif factor_id == '005':
        be_half = (adj_OPEN - vwap.rolling(window=10).mean()).rank(axis=1, pct=True)
        af_half = (adj_close - vwap).rank(axis=1, pct=True)
        factor = -1 * be_half * af_half
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '006':
        factor = -1 * adj_OPEN.rolling(window=10).corr(adj_volume)
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '007':
        factor = -1 * rolling_rank(np.abs(adj_close.diff(7)), 60) * np.sign(adj_close.diff(7))
        factor[adj_volume.rolling(window=20).mean() > volume] = -1
        factor[rolling_null(close, 60)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '008':
        temp = OPEN.rolling(window=5).sum() * returns.rolling(window=5).sum()
        factor = -1 * (temp - temp.diff(10)).rank(axis=1, pct=True)
        # temp - temp.diff(10) 不就是10日前的temp吗？ SB啊！
        factor[rolling_null(close, 15)] = np.nan
        factor[rolling_stop(stop, 15)] = np.nan
    elif factor_id == '009':
        cond_1 = adj_close.diff().rolling(window=5).min() > 0
        cond_2 = adj_close.diff().rolling(window=5).max() < 0
        factor = -1 * adj_close.diff()
        factor[cond_1 | cond_2] = adj_close.diff()
        factor[rolling_null(close, 6)] = np.nan
        factor[rolling_stop(stop, 6)] = np.nan
    elif factor_id == '010':
        cond_1 = adj_close.diff().rolling(window=4).min() > 0
        cond_2 = adj_close.diff().rolling(window=4).max() < 0
        factor = -1 * adj_close.diff()
        factor[cond_1 | cond_2] = adj_close.diff()
        factor = factor.rank(axis=1, pct=True)
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '011':
        temp1 = (vwap - adj_close).rolling(window=3).max().rank(aixs=1, pct=True)
        temp2 = (vwap - adj_close).rolling(window=3).min().rank(axis=1, pct=True)
        temp3 = adj_volume.diff(3).rank(axis=1, pct=True)
        factor = (temp1 + temp2) * temp3
        factor[rolling_null(close, 3)] = np.nan
        factor[rolling_stop(stop, 3)] = np.nan
    elif factor_id == '012':
        factor = -1 * np.sign(adj_volume.diff()) * adj_close.diff()
        factor[rolling_null(close, 2)] = np.nan
        factor[rolling_stop(stop, 2)] = np.nan
    elif factor_id == '013':
        close_rank = adj_close.rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        factor = -1 * (close_rank.rolling(window=5).cov(volume_rank)).rank(axis=1, pct=True)
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '014':
        temp1 = adj_OPEN.rolling(window=10).corr(adj_volume)
        temp2 = returns.diff(3).rank(axis=1, pct=True)
        factor = -1 * temp2 * temp1
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '015':
        high_rank = adj_high.rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        temp1 = high_rank.rolling(window=3).corr(volume_rank)
        factor = -1 * temp1.rank(axis=1, pct=True).rolling(window=3).sum()
        factor[rolling_null(close, 6)] = np.nan
        factor[rolling_stop(stop, 6)] = np.nan
    elif factor_id == '016':
        high_rank = adj_high.rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        factor = -1 * high_rank.rolling(window=5).cov(volume_rank)
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '017':
        temp1 = rolling_rank(adj_close, 10).rank(axis=1, pct=True)
        temp2 = adj_close.diff(1).diff(1).rank(axis=1, pct=True)
        temp3 = rolling_rank(adj_volume / adj_volume.rolling(window=20).mean(), 5).rank(axis=1, pct=True)
        factor = -1 * temp1 * temp2 * temp3
        factor[rolling_null(close, 25)] = np.nan
        factor[rolling_stop(stop, 25)] = np.nan
    elif factor_id == '018':
        temp1 = adj_close.rolling(window=10).corr(adj_OPEN).replace([-np.inf, np.inf], np.nan)
        temp2 = np.abs(adj_close - adj_OPEN).rolling(window=5).std()
        factor = -1 * (temp2 + temp1 + adj_close - adj_OPEN).rank(axis=1, pct=True)
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '019':
        factor = -1 * np.sign(adj_close.diff(7)) * ((1 + returns.rolling(window=250).sum()).rank(axis=1, pct=True) + 1)
        factor[rolling_null(close, 250)] = np.nan
        factor[rolling_stop(stop, 8)] = np.nan
    elif factor_id == '020':
        temp1 = (adj_OPEN - adj_high.diff(1)).rank(axis=1, pct=True)
        temp2 = (adj_OPEN - adj_close.diff(1)).rank(axis=1, pct=True)
        temp3 = (adj_OPEN - adj_low.diff(1)).rank(axis=1, pct=True)
        factor = -1 * temp1 * temp2 * temp3
        factor[rolling_null(close, 2)] = np.nan
        factor[rolling_stop(stop, 2)] = np.nan
    elif factor_id == '021':
        cond_1 = adj_close.rolling(window=8).mean() + adj_close.rolling(window=8).std() < adj_close.rolling(window=2).mean()
        cond_2 = adj_close.rolling(window=8).mean() - adj_close.rolling(window=8).std() > adj_close.rolling(window=2).mean()
        cond_3 = adj_volume / adj_volume.rolling(window=20).mean() >= 1
        cond_4 = adj_volume / adj_volume.rolling(window=20).mean() < 1
        factor = pd.DataFrame(np.nan, index=adj_close.index, columns=adj_close.columns)
        factor[cond_1] = -1
        factor[cond_2] = 1
        factor[np.logical_and(np.logical_not(np.logical_or(cond_1, cond_2)), cond_3)] = 1
        factor[np.logical_and(np.logical_not(np.logical_or(cond_1, cond_2)), cond_4)] = -1
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '022':
        part_1 = adj_high.rolling(window=5).corr(adj_volume).diff(5)
        part_2 = adj_close.rolling(window=20).std().rank(axis=1, pct=True)
        factor = -1 * part_1 * part_2
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '023':
        factor = pd.DataFrame(np.nan, index=adj_high.index, columns=adj_high.columns)
        factor[adj_high > adj_high.rolling(window=20).mean()] = -1 * adj_high.diff(2)
        factor[adj_high <= adj_high.rolling(window=20).mean()] = 0
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '024':
        cond_1 = adj_close.rolling(window=100).mean().diff(100) / adj_close.shift(100) <= 0.05
        factor = -1 * adj_close.diff(3)
        factor[cond_1] = -1 * (adj_close - adj_close.rolling(window=100).min())
        factor[rolling_null(close, 100)] = np.nan
        factor[rolling_stop(stop, 3)] = np.nan
    elif factor_id == '025':
        part_1 = -1 * returns * adj_volume.rolling(window=20).mean()
        part_2 = vwap * (adj_high - adj_close)
        factor = (part_1 * part_2).rank(axis=1, pct=True)
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '026':
        volume_rank = rolling_rank(adj_volume, 5)
        high_rank = rolling_rank(adj_high, 5)
        factor = -1 * volume_rank.rolling(window=5).corr(high_rank).rolling(window=3).max()
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '027':
        factor = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        vwap_rank = vwap.rank(axis=1, pct=True)
        corr_ = volume_rank.rolling(window=6).corr(vwap_rank)
        corr_rank = corr_.rolling(window=2).mean().rank(axis=1, pct=True)
        cond_1 = corr_rank > 0.5
        cond_2 = corr_rank <= 0.5
        factor[cond_1] = -1
        factor[cond_2] = 1
        factor[rolling_null(close, 6)] = np.nan
        factor[rolling_stop(stop, 6)] = np.nan
    elif factor_id == '028':
        part_1 = adj_volume.rolling(window=20).mean().rolling(window=5).corr(adj_low)
        part_2 = (adj_high + adj_low) / 2 - adj_close
        factor = part_1 + part_2
        factor = (factor.T / factor.sum(axis=1)).T
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '029':
        temp1 = rolling_rank(-1 * returns.shift(6), 5)
        close_rank = (-1 * (adj_close - 1).diff(5).rank(axis=1, pct=True)).rank(axis=1, pct=True)
        temp2 = np.log(close_rank.rolling(window=2).min())
        temp3 = (temp2.T / temp2.sum(axis=1)).T.rank(axis=1, pct=True)
        factor = temp3.rolling(window=5).min() + temp1
        factor[rolling_null(close, 6)] = np.nan
        factor[rolling_stop(stop, 6)] = np.nan
    elif factor_id == '030':
        sign_1 = np.sign(adj_close.diff())
        sign_2 = np.sign(adj_close.diff().shift())
        sign_3 = np.sign(adj_close.diff().shift(2))
        sign_rank = (sign_1 + sign_2 + sign_3)
        factor = (1 - sign_rank) * adj_volume.rolling(window=5).sum() / adj_volume.rolling(window=20).sum()
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '031':
        close_rank_1 = adj_close.diff(10).rank(axis=1, pct=True)
        decay_linear_rank = linear_decay(-1 * close_rank_1, 10).rank(axis=1, pct=True)
        close_rank_2 = (-1 * adj_close.diff(3)).rank(axis=1, pct=True)
        corr_ = adj_volume.rolling(window=20).mean().rolling(window=12).corr(adj_low)
        corr_sign = np.sign((corr_.T / corr_.sum(axis=1)).T)
        factor = decay_linear_rank + close_rank_2 + corr_sign
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '032':
        temp1 = adj_close.rolling(window=7).mean() - adj_close
        part_1 = (temp1.T / temp1.sum(axis=1)).T
        temp2 = vwap.rolling(window=230).corr(adj_close.shift(5))
        part_2 = (temp2.T / temp2.sum(axis=1)).T
        factor = part_1 + 20 * part_2
        factor[rolling_null(close, 230)] = np.nan
        factor[rolling_stop(stop, 7)] = np.nan
    elif factor_id == '033':
        factor = (OPEN / close - 1).rank(axis=1, pct=True)
        factor[stop.notnull()] = np.nan
    elif factor_id == '034':
        temp1 = returns.rolling(window=2).std() / returns.rolling(window=5).std()
        close_rank = adj_close.diff().rank(axis=1, pct=True)
        factor = (1 - temp1.rank(axis=1, pct=True) + 1 - close_rank).rank(axis=1, pct=True)
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '035':
        volume_rank = rolling_rank(adj_volume, 32)
        xx_rank = rolling_rank(adj_close + adj_high - adj_low, 16)
        returns_rank = rolling_rank(returns, 32)
        factor = volume_rank * (1 - xx_rank) * (1 - returns_rank)
        factor[rolling_null(close, 32)] = np.nan
        factor[rolling_stop(stop, 32)] = np.nan
    elif factor_id == '036':
        part_1 = (adj_close - adj_OPEN).rolling(window=15).corr(adj_volume.shift(1)).rank(axis=1, pct=True)
        part_2 = (adj_OPEN - adj_close).rank(axis=1, pct=True)
        part_3 = rolling_rank(-1 * returns.shift(6), 5).rank(axis=1, pct=True)
        part_4 = vwap.rolling(window=6).corr(adj_volume.rolling(window=20).mean()).rank(axis=1, pct=True)
        part_5 = ((adj_close.rolling(window=200).mean() - adj_OPEN) * (adj_close - adj_OPEN)).rank(axis=1, pct=True)
        factor = 2.21 * part_1 + 0.7 * part_2 + 0.73 * part_3 + part_4 + 0.6 * part_5
        factor[rolling_null(close, 200)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '037':
        corr_ = (adj_OPEN - adj_close).shift().rolling(window=200).corr(adj_close)
        factor = corr_.rank(axis=1, pct=True) + (adj_OPEN - adj_close).rank(axis=1, pct=True)
        factor[rolling_null(close, 200)] = np.nan
        factor[stop.notnull()] = np.nan
    elif factor_id == '038':
        part_1 = rolling_rank(adj_close, 10).rank(axis=1, pct=True)
        part_2 = (close / OPEN).rank(axis=1, pct=True)
        factor = -1 * part_1 * part_2
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '039':
        temp1 = adj_volume / adj_volume.rolling(window=20).mean()
        decay_linear_rank = linear_decay(temp1, 9).rank(axis=1, pct=True)
        part_1 = (adj_close.diff(7) * (1 - decay_linear_rank)).rank(axis=1, pct=True)
        part_2 = returns.rolling(window=250).sum().rank(axis=1, pct=True)
        factor = -1 * part_1 * (1 + part_2)
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '040':
        factor = -1 * adj_high.rolling(window=10).std().rank(axis=1, pct=True) * adj_high.rolling(window=10).corr(adj_volume)
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '041':
        factor = ((adj_high * adj_low) ** 0.5 - vwap * 10) / adj_close
        factor = np.abs(factor.T - factor.mean(axis=1)).T
        factor[stop.notnull()] = np.nan
    elif factor_id == '042':
        factor = (vwap - adj_close).rank(axis=1, pct=True) / (vwap + adj_close).rank(axis=1, pct=True)
        factor[stop.notnull()] = np.nan
    elif factor_id == '043':
        part_1 = rolling_rank(adj_volume / adj_volume.rolling(window=20).mean(), 20)
        part_2 = rolling_rank(-1 * adj_close.diff(7), 8)
        factor = part_1 * part_2
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '044':
        factor = -1 * adj_high.rolling(window=5).corr(adj_volume.rank(axis=1, pct=True))
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '045':
        part_1 = adj_close.shift(5).rolling(window=20).mean().rank(axis=1, pct=True)
        part_2 = adj_close.rolling(window=2).corr(adj_volume)
        temp1 = adj_close.rolling(window=5).sum()
        temp2 = adj_close.rolling(window=20).sum()
        part_3 = temp1.rolling(window=2).corr(temp2).rank(axis=1, pct=True)
        factor = -1 * part_1 * part_2 * part_3
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '046':
        factor = -1 * adj_close.diff(1)
        temp1 = (adj_close.shift(20) - adj_close.shift(10)) / 10
        temp2 = (adj_close.shift(10) - adj_close) / 10
        cond_1 = temp1 - temp2 > 0.25
        cond_2 = temp1 - temp2 < 0
        factor[cond_2] = 1
        factor[cond_1] = -1
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '047':
        part_1 = (1 / adj_close).rank(axis=1, pct=True) * adj_volume / adj_volume.rolling(window=20).mean()
        part_2 = adj_high * (adj_high - adj_close).rank(axis=1, pct=True) / adj_high.rolling(window=5).mean()
        part_3 = vwap.diff(5).rank(axis=1, pct=True)
        factor = part_1 * part_2 * part_3
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '048':
        temp1 = adj_close.diff().rolling(window=250).corr(adj_close.shift().diff())
        temp2 = temp1 * adj_close.diff() / adj_close
        part_1 = ind_neutral(temp2, ind)
        part_2 = ((adj_close.diff() / adj_close.shift()) ** 2).rolling(window=250).sum()
        factor = part_1 / part_2
        factor[rolling_null(close, 250)] = np.nan
        factor[rolling_stop(stop, 2)] = np.nan
    elif factor_id == '049':
        factor = -1 * adj_close.diff()
        temp1 = (adj_close.shift(20) - adj_close.shift(10)) / 10
        temp2 = (adj_close.shift(10) - adj_close) / 10
        factor[temp1 - temp2 < 0.1] = 1
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '050':
        volume_rank = adj_volume.rank(axis=1, pct=True)
        vwap_rank = vwap.rank(axis=1, pct=True)
        corr_ = volume_rank.rolling(window=5).corr(vwap_rank)
        factor = -1 * rolling_rank(corr_.rank(axis=1, pct=True), 5)
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '051':
        factor = -1 * adj_close.diff()
        temp1 = (adj_close.shift(20) - adj_close.shift(10)) / 10
        temp2 = (adj_close.shift(10) - adj_close) / 10
        cond = temp1 - temp2 < -0.05
        factor[cond] = 1
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '052':
        part_1 = -1 * adj_low.rolling(window=5).min().diff(5)
        part_2 = (returns.rolling(window=240).sum() - returns.rolling(window=20).sum()) / 220
        factor = part_1 * part_2.rank(axis=1, pct=True) * rolling_rank(adj_volume, 5)
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '053':
        temp = ((close - low) - (high - close)) / (close - low)
        factor = -1 * temp.diff(9)
        factor[rolling_null(close, 9)] = np.nan
        factor[rolling_stop(stop, 9)] = np.nan
    elif factor_id == '054':
        factor = -1 * (low - close) * (OPEN ** 5) / ((low - high) * (close ** 5))
        factor[stop.notnull()] = np.nan
    elif factor_id == '055':
        numerator = adj_close - adj_low.rolling(window=12).min()
        denominator = adj_high.rolling(window=12).max() - adj_low.rolling(window=12).min()
        fraction = numerator / denominator
        fraction[rolling_stop(stop, 12)] = np.nan
        fraction_rank = fraction.rank(axis=1, pct=True)
        volume_rank = volume.replace(0, np.nan).rank(axis=1, pct=True)
        factor = -1 * fraction_rank.rolling(window=6).corr(volume_rank)
        factor[rolling_null(close, 12)] = np.nan
        factor[rolling_stop(stop, 12)] = np.nan
    elif factor_id == '056':
        temp1 = returns.rolling(window=10).sum()
        temp2 = returns.rolling(window=2).sum().rolling(window=3).sum()
        part_1 = (temp1 / temp2).rank(axis=1, pct=True)
        part_2 = (returns * cap).rank(axis=1, pct=True)
        factor = -1 * part_1 * part_2
        factor[rolling_null(close, 12)] = np.nan
        factor[rolling_stop(stop, 12)] = np.nan
    elif factor_id == '057':
        part_1 = adj_close - vwap
        temp = adj_close.rolling(window=30).apply(np.argmax).rank(axis=1, pct=True)
        part_2 = linear_decay(temp, 2)
        factor = -1 * part_1 / part_2
        factor[rolling_null(close, 30)] = np.nan
        factor[rolling_stop(stop, 30)] = np.nan
    elif factor_id == '058':
        temp1 = ind_neutral(vwap, ind)
        temp2 = temp1.rolling(window=3).corr(adj_volume)
        temp3 = linear_decay(temp2, 7)
        factor = -1 * rolling_rank(temp3, 5)
        factor[rolling_null(close, 7)] = np.nan
        factor[rolling_stop(stop, 7)] = np.nan
    elif factor_id == '059':
        temp1 = vwap * 0.728317 + vwap * (1 - 0.728317)
        temp2 = ind_neutral(temp1, ind)
        temp3 = temp2.rolling(window=4).corr(adj_volume)
        factor = -1 * rolling_rank(linear_decay(temp3, 16), 8)
        factor[rolling_null(close, 16)] = np.nan
        factor[rolling_stop(stop, 16)] = np.nan
    elif factor_id == '060':
        temp1 = adj_volume * ((adj_close - adj_low) - (adj_high - adj_close)) / (adj_high - adj_low)
        temp2 = temp1.rank(axis=1, pct=True)
        part_1 = (temp2.T / temp2.sum(axis=1)).T
        temp3 = adj_close.rolling(window=10).apply(np.argmax).rank(axis=1, pct=True)
        part_2 = (temp3.T / temp3.sum(axis=1)).T
        factor = -1 * (2 * part_1 - part_2)
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '061':
        adv180 = adj_volume.rolling(window=180).mean()
        part_1 = (vwap - vwap.rolling(window=16).min()).rank(axis=1, pct=True)
        part_2 = vwap.rolling(window=17).corr(adv180).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor[rolling_null(close, 180)] = np.nan
        factor[rolling_stop(stop, 16)] = np.nan
    elif factor_id == '062':
        adv20 = adj_volume.rolling(window=20).mean()
        part_1 = vwap.rolling(window=9).corr(adv20.rolling(window=22).sum()).rank(axis=1, pct=True)
        temp1 = 2 * adj_OPEN.rank(axis=1, pct=True)
        temp2 = ((adj_high + adj_low) / 2).rank(axis=1, pct=True) + adj_high.rank(axis=1, pct=True)
        temp3 = temp1
        temp3[temp1 < temp2] = 1
        temp3[temp1 >= temp2] = 0
        part_2 = temp3.rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor = -1 * factor
        factor[rolling_null(close, 22)] = np.nan
        factor[rolling_stop(stop, 22)] = np.nan
    elif factor_id == '063':
        temp1 = ind_neutral(adj_close, ind).diff(2)
        part_1 = linear_decay(temp1, 8).rank(axis=1, pct=True)
        temp2 = vwap * 0.318108 + adj_OPEN * (1 - 0.318108)
        temp3 = adj_volume.rolling(window=180).mean().rolling(window=37).sum()
        temp4 = temp2.rolling(window=13).corr(temp3)
        part_2 = linear_decay(temp4, 12).rank(axis=1, pct=True)
        factor = -1 * (part_1 - part_2)
        factor[rolling_null(close, 180)] = np.nan
        factor[rolling_stop(stop, 13)] = np.nan
    elif factor_id == '064':
        adv120 = adj_volume.rolling(window=120).mean()
        temp1 = (adj_OPEN * 0.178404 + adj_low * (1 - 0.178404)).rolling(window=12).sum()
        temp2 = adv120.rolling(window=12).sum()
        part_1 = temp1.rolling(window=16).corr(temp2).rank(axis=1, pct=True)
        temp3 = (adj_high + adj_low) / 2 * 0.178404 + vwap * (1 - 0.178404)
        part_2 = temp3.diff(3).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor = -1 * factor
        factor[rolling_null(close, 120)] = np.nan
        factor[rolling_stop(stop, 16)] = np.nan
    elif factor_id == '065':
        adv60 = adj_volume.rolling(window=60).mean()
        temp1 = adj_OPEN * 0.00817205 + vwap * (1 - 0.00817205)
        temp2 = adv60.rolling(window=8).sum()
        part_1 = temp1.rolling(window=6).corr(temp2).rank(axis=1, pct=True)
        part_2 = (adj_OPEN - adj_OPEN.rolling(window=13).min()).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor = -1 * factor
        factor[rolling_null(close, 60)] = np.nan
        factor[rolling_stop(stop, 13)] = np.nan
    elif factor_id == '066':
        part_1 = linear_decay(vwap.diff(3), 7).rank(axis=1, pct=True)
        temp1 = (adj_low - vwap) / (adj_OPEN - (high + low) / 2)
        temp2 = linear_decay(temp1, 11)
        part_2 = rolling_rank(temp2, 6)
        factor = -1 * (part_1 + part_2)
        factor[rolling_null(close, 11)] = np.nan
        factor[rolling_stop(stop, 11)] = np.nan
    elif factor_id == '067':
        part_1 = (adj_high - adj_high.rolling(window=2).min()).rank(axis=1, pct=True)
        temp1 = ind_neutral(vwap, ind)
        temp2 = ind_neutral(adj_volume.rolling(window=20).mean(), ind)
        part_2 = temp1.rolling(window=6).corr(temp2).rank(axis=1, pct=True)
        factor = -1 * (part_1 ** part_2)
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '068':
        adv15 = adj_volume.rolling(window=15).mean()
        high_rank = adj_high.rank(axis=1, pct=True)
        adv15_rank = adv15.rank(axis=1, pct=True)
        part_1 = rolling_rank(high_rank.rolling(window=8).corr(adv15_rank), 13)
        temp = adj_close * 0.518371 + adj_low * (1 - 0.518371)
        part_2 = temp.diff(1).rank(axis=1, pct=True)
        factor = part_1.sort(axis=1)
        factor[part_1.sort(axis=1) < part_2.sort(axis=1)] = 1
        factor[part_1.sort(axis=1) >= part_2.sort(axis=1)] = 0
        factor = -1 * factor
        factor[rolling_null(close, 15)] = np.nan
        factor[rolling_stop(stop, 15)] = np.nan
    elif factor_id == '069':
        temp1 = ind_neutral(vwap, ind).diff(2)
        part_1 = temp1.rolling(window=4).max().rank(axis=1, pct=True)
        temp2 = adj_close * 0.490655 + vwap * (1 - 0.490655)
        temp3 = adj_volume.rolling(window=20).mean()
        part_2 = rolling_rank(temp2.rolling(window=4).corr(temp3), 9)
        factor = -1 * (part_1 ** part_2)
        factor[rolling_null(close, 20)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '070':
        part_1 = vwap.diff(1).rank(axis=1, pct=True)
        temp1 = ind_neutral(adj_close, ind)
        temp2 = adj_volume.rolling(window=50).mean()
        part_2 = rolling_rank(temp1.rolling(window=17).corr(temp2), 17)
        factor = -1 * (part_1 ** part_2)
        factor[rolling_null(close, 50)] = np.nan
        factor[rolling_stop(stop, 17)] = np.nan
    elif factor_id == '071':
        adv180 = adj_volume.rolling(window=180).mean()
        adv180_rank = rolling_rank(adv180, 12)
        close_rank = rolling_rank(adj_close, 3)
        corr_ = close_rank.rolling(window=18).corr(adv180_rank)
        part_1 = rolling_rank(linear_decay(corr_, 4), 15)
        temp = (((adj_low + adj_OPEN) - (vwap + vwap)).rank(axis=1, pct=True)) ** 2
        part_2 = rolling_rank(linear_decay(temp, 16), 4)
        factor = part_1
        factor[part_1 < part_2] = part_2
        factor[rolling_null(close, 180)] = np.nan
        factor[rolling_stop(stop, 18)] = np.nan
    elif factor_id == '072':
        adv40 = adj_volume.rolling(window=40).mean()
        corr_1 = ((adj_high + adj_low) / 2).rolling(window=8).corr(adv40)
        part_1 = linear_decay(corr_1, 10).rank(axis=1, pct=True)
        corr_2 = (rolling_rank(vwap, 3).rolling(window=6).corr(rolling_rank(volume, 18)))
        part_2 = linear_decay(corr_2, 2).rank(axis=1, pct=True)
        factor = part_1 / part_2
        factor[rolling_null(close, 40)] = np.nan
        factor[rolling_stop(stop, 18)] = np.nan
    elif factor_id == '073':
        part_1 = linear_decay(vwap.diff(4), 2).rank(axis=1, pct=True)
        temp1 = adj_OPEN * 0.147155 + adj_low * (1 - 0.147155)
        part_2 = rolling_rank(linear_decay(-1 * temp1.diff(2) / temp1, 3), 16)
        temp2 = adj_OPEN * 0.147155 + adj_low * (1 - 0.147155)
        factor = part_1
        factor[part_1 < part_2] = part_2
        factor = -1 * factor
        factor[rolling_null(close, 16)] = np.nan
        factor[rolling_stop(stop, 16)] = np.nan
    elif factor_id == '074':
        adv30 = adj_volume.rolling(window=30).mean()
        sum_adv30 = adv30.rolling(window=37).sum()
        corr_1 = adj_close.rolling(window=15).corr(sum_adv30)
        part_1 = corr_1.rank(axis=1, pct=True)
        temp1 = (adj_high * 0.0261661 + vwap * (1 - 0.0261661)).rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        corr_2 = temp1.rolling(window=11).corr(volume_rank)
        part_2 = corr_2.rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor = -1 * factor
        factor[rolling_null(close, 67)] = np.nan
        factor[rolling_stop(stop, 30)] = np.nan
    elif factor_id == '075':
        part_1 = vwap.rolling(window=4).corr(adj_volume).rank(axis=1, pct=True)
        low_rank = adj_low.rank(axis=1, pct=True)
        adv50_rank = adj_volume.rolling(window=50).mean().rank(axis=1, pct=True)
        part_2 = low_rank.rolling(window=12).corr(adv50_rank).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor[rolling_null(close, 50)] = np.nan
        factor[rolling_stop(stop, 12)] = np.nan
    elif factor_id == '076':
        adv81 = adj_volume.rolling(window=81).mean()
        part_1 = linear_decay(vwap.diff(1), 11).rank(axis=1, pct=True)
        low_ind = ind_neutral(adj_low, ind)
        corr_ = low_ind.rolling(window=8).corr(adv81)
        part_2 = rolling_rank(linear_decay(rolling_rank(corr_, 19), 17), 19)
        factor = part_1
        factor[part_1 < part_2] = part_2
        factor = -1 * factor
        factor[rolling_null(close, 81)] = np.nan
        factor[rolling_stop(stop, 19)] = np.nan
    elif factor_id == '077':
        adv40 = adj_volume.rolling(window=40).mean()
        temp1 = (adj_high + adj_low) / 2 + adj_high - (vwap +adj_high)
        part_1 = linear_decay(temp1, 20).rank(axis=1, pct=True)
        temp2 = (adj_high + adj_low) / 2
        corr_ = temp2.rolling(window=3).corr(adv40)
        part_2 = linear_decay(corr_, 5).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 > part_2] = part_2
        factor[rolling_null(close, 43)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '078':
        temp1 = (adj_low * 0.352233 + vwap * (1 - 0.352233)).rolling(window=19).sum()
        temp2 = adj_volume.rolling(window=40).mean().rolling(window=19).sum()
        part_1 = temp1.rolling(window=6).corr(temp2).rank(axis=1, pct=True)
        vwap_rank = vwap.rank(axis=1, pct=True)
        volume_rank = volume.rank(axis=1, pct=True)
        part_2 = vwap_rank.rolling(window=5).corr(volume_rank).rank(axis=1, pct=True)
        factor = part_1 ** part_2
        factor[rolling_null(close, 59)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '079':
        adv150 = adj_volume.rolling(window=150).mean()
        temp = adj_close * 0.60733 + adj_OPEN * (1 - 0.60733)
        part_1 = ind_neutral(temp, ind).diff(1).rank(axis=1, pct=True)
        vwap_rank = rolling_rank(vwap, 3)
        adv150_rank = rolling_rank(adv150, 9)
        part_2 = vwap_rank.rolling(window=3).corr(adv150_rank).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor[rolling_null(close, 150)] = np.nan
        factor[rolling_stop(stop, 9)] = np.nan
    elif factor_id == '080':
        temp1 = ind_neutral(adj_OPEN * 0.868128 + adj_high * (1 - 0.868128), ind)
        part_1 = np.sign(temp1.diff(4)).rank(axis=1, pct=True)
        adv10 = adj_volume.rolling(window=10).mean()
        part_2 = rolling_rank(adj_high.rolling(window=5).corr(adv10), 5)
        factor = -1 * (part_1 ** part_2)
        factor[rolling_null(close, 10)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '081':
        adv10 = adj_volume.rolling(window=10).mean()
        sum_adv10 = adv10.rolling(window=49).sum()
        corr_ = vwap.rolling(window=8).corr(sum_adv10)
        corr_rank = corr_.rank(axis=1, pct=True)
        part_1 = np.log(corr_rank.rolling(window=14).apply(np.prod)).rank(axis=1, pct=True)
        vwap_rank = vwap.rank(axis=1, pct=True)
        volume_rank = volume.rank(axis=1, pct=True)
        part_2 = vwap_rank.rolling(window=5).corr(volume_rank).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor = -1 * factor
        factor[rolling_null(close, 59)] = np.nan
        factor[rolling_stop(stop, 14)] = np.nan
    elif factor_id == '082':
        part_1 = linear_decay(adj_OPEN.diff(), 14).rank(axis=1, pct=True)
        volume_ind = ind_neutral(adj_volume, ind)
        corr_ = volume_ind.rolling(window=17).corr(adj_OPEN)
        part_2 = rolling_rank(linear_decay(corr_, 6), 13)
        factor = part_1
        factor[part_1 > part_2] = part_2
        factor = -1 * factor
        factor[rolling_null(close, 17)] = np.nan
        factor[rolling_stop(stop, 17)] = np.nan
    elif factor_id == '083':
        temp1 = (adj_high - adj_low) / adj_close.rolling(window=5).mean()
        part_1 = temp1.shift(2).rank(axis=1, pct=True) * adj_volume.rank(axis=1, pct=True)
        part_2 = temp1 / (vwap - adj_close)
        factor = part_1 / part_2
        factor[rolling_null(close, 5)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '084':
        part_1 = rolling_rank(vwap - vwap.rolling(window=15).max(), 20)
        part_2 = adj_close.diff(4)
        factor = part_1 ** part_2
        factor[rolling_null(close, 35)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '085':
        temp1 = adj_high * 0.876703 + adj_close * (1 - 0.876703)
        adv30 = adj_volume.rolling(window=30).mean()
        corr_1 = temp1.rolling(window=9).corr(adv30)
        part_1 = corr_1.rank(axis=1, pct=True)
        hl_rank = rolling_rank((adj_high + adj_low) / 2, 3)
        volume_rank = rolling_rank(volume, 10)
        corr_2 = hl_rank.rolling(window=7).corr(volume_rank)
        part_2 = corr_2.rank(axis=1, pct=True)
        factor = part_1 ** part_2
        factor[rolling_null(close, 30)] = np.nan
        factor[rolling_stop(stop, 30)] = np.nan
    elif factor_id == '086':
        adv20 = adj_volume.rolling(window=20).mean()
        sum_adv20 = adv20.rolling(window=14).sum()
        part_1 = rolling_rank(adj_close.rolling(window=6).corr(sum_adv20), 20)
        part_2 = (adj_close - vwap).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] =0
        factor = -1 * factor
        factor[rolling_null(close, 60)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '087':
        adv81 = adj_volume.rolling(window=81).mean()
        temp1 = adj_close * 0.369701 + vwap * (1 - 0.369701)
        part_1 = linear_decay(temp1.diff(1), 2).rank(axis=1, pct=True)
        adv81_ind = ind_neutral(adv81, ind)
        corr_ = np.abs(adv81_ind.rolling(window=13).corr(adj_close))
        part_2 = rolling_rank(linear_decay(corr_, 4), 14)
        factor = part_1
        factor[part_1 < part_2] = part_2
        factor = -1 * factor
        factor[rolling_null(close, 108)] = np.nan
        factor[rolling_stop(stop, 14)] = np.nan
    elif factor_id == '088':
        temp1 = (adj_OPEN.rank(axis=1, pct=True) + adj_low.rank(axis=1, pct=True) - 
                 adj_high.rank(axis=1, pct=True) - adj_close.rank(axis=1, pct=True))
        part_1 = linear_decay(temp1, 8).rank(axis=1, pct=True)
        close_rank = rolling_rank(adj_close, 8)
        adv60 = adj_volume.rolling(window=60).mean()
        adv60_rank = rolling_rank(adv60, 20)
        corr_ = close_rank.rolling(window=8).corr(adv60_rank)
        part_2 = rolling_rank(linear_decay(corr_, 6), 2)
        factor = part_1
        factor[part_1 > part_2] = part_2
        factor[rolling_null(close, 88)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '089':
        adv10 = adj_volume.rolling(window=10).mean()
        corr_ = adj_low.rolling(window=6).corr(adv10)
        part_1 = rolling_rank(linear_decay(corr_, 5), 3)
        vwap_ind = ind_neutral(vwap, ind)
        part_2 = rolling_rank(linear_decay(vwap_ind.diff(3), 10), 15)
        factor = part_1 - part_2
        factor[rolling_null(close, 28)] = np.nan
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '090':
        part_1 = (adj_close - adj_close.rolling(window=4).max()).rank(axis=1, pct=True)
        adv_ind = ind_neutral(adj_volume.rolling(window=40).mean(), ind)
        corr_ = adv_ind.rolling(window=5).corr(adj_low)
        part_2 = rolling_rank(corr_, 3)
        factor = -1 * (part_1 ** part_2)
        factor[rolling_null(close, 48)] = np.nan
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '091':
        close_ind = ind_neutral(adj_close, ind)
        corr_1 = close_ind.rolling(window=9).corr(adj_volume)
        part_1 = rolling_rank(linear_decay(linear_decay(corr_1, 16), 3), 4)
        corr_2 = (value / adj_volume).rolling(window=4).corr(adj_volume.rolling(window=30).mean())
        part_2 = linear_decay(corr_2, 2).rank(axis=1, pct=True)
        factor = -1 * (part_1 - part_2)
        factor[rolling_null(close, 32)] = np.nan
        factor[rolling_stop(stop, 32)] = np.nan
    elif factor_id == '092':
        temp1 = (adj_high + adj_low) / 2 + adj_close
        temp2 = adj_low + adj_OPEN
        temp3 = temp1
        temp3[temp1 < temp2] = 1
        temp3[temp1 >= temp2] = 0
        part_1 = rolling_rank(linear_decay(temp3, 14), 18)
        low_rank = adj_low.rank(axis=1, pct=True)
        adv30_rank = adj_volume.rolling(window=30).mean().rank(axis=1, pct=True)
        corr_ = low_rank.rolling(window=7).corr(adv30_rank)
        part_2 = rolling_rank(linear_decay(corr_, 6), 6)
        factor = part_1
        factor[part_1 > part_2] = part_2
        factor[rolling_null(close, 43)] = np.nan
        factor[rolling_stop(stop, 18)] = np.nan
    elif factor_id == '093':
        ind_vwap = ind_neutral(vwap, ind)
        corr_ = ind_vwap.rolling(window=17).corr(adj_volume.rolling(window=81).mean())
        part_1 = rolling_rank(linear_decay(corr_, 19), 7)
        temp1 = (adj_close * 0.52434 + vwap * (1 - 0.524434)).diff(2)
        part_2 = linear_decay(temp1, 16).rank(axis=1, pct=True)
        factor = part_1 / part_2
        factor[rolling_null(close, 43)] = np.nan
        factor[rolling_stop(stop, 19)] = np.nan
    elif factor_id == '094':
        part_1 = (vwap - vwap.rolling(window=11).min()).rank(axis=1, pct=True)
        vwap_rank = rolling_rank(vwap, 19)
        adv_rank = rolling_rank(adj_volume.rolling(window=60).mean(), 4)
        corr_ = vwap_rank.rolling(window=18).corr(adv_rank)
        part_2 = rolling_rank(corr_, 2)
        factor = -1 * (part_1 ** part_2)
        factor[rolling_null(close, 82)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '095':
        adv40 = adj_volume.rolling(window=40).mean()
        part_1 = (adj_OPEN - adj_OPEN.rolling(window=12).min()).rank(axis=1, pct=True)
        temp1 = ((adj_high + adj_low) / 2).rolling(window=19).sum()
        temp2 = adv40.rolling(window=19).sum()
        corr_ = temp1.rolling(window=12).corr(temp2)
        part_2 = rolling_rank(corr_.rank(axis=1, pct=True) ** 5, 11)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] = 0
        factor[rolling_null(close, 71)] = np.nan
        factor[rolling_stop(stop, 19)] = np.nan
    elif factor_id == '096':
        vwap_rank = vwap.rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        corr_1 = vwap_rank.rolling(window=3).corr(volume_rank)
        part_1 = rolling_rank(linear_decay(corr_1, 4), 8)
        adv60 = adj_volume.rolling(window=60).mean()
        close_rank = rolling_rank(adj_close,7)
        adv60_rank = rolling_rank(adv60, 4)
        corr_2 = close_rank.rolling(window=3).corr(adv60_rank)
        part_2 = rolling_rank(linear_decay(corr_2.rolling(window=12).apply(np.argmax), 14), 13)
        part_1[part_1 < part_2] = part_2
        factor = -1 * part_1
        factor[rolling_null(close, 80)] = np.nan
        factor[rolling_stop(stop, 12)] = np.nan
    elif factor_id == '097':
        temp = adj_low * 0.721001 + vwap * (1 - 0.721001)
        indneu_diff = ind_neutral(temp, ind).diff(3)
        part_1 = linear_decay(indneu_diff, 20).rank(axis=1, pct=True)
        low_rank = rolling_rank(adj_low, 7)
        adv_rank = rolling_rank(adj_volume.rolling(window=60).mean(), 17)
        corr_ = low_rank.rolling(4).corr(adv_rank)
        corr_rank = rolling_rank(corr_, 18)
        part_2 = rolling_rank(linear_decay(corr_rank, 15), 6)
        factor = -1 * (part_1 - part_2)
        factor[rolling_null(close, 103)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '098':
        temp1 = vwap.rolling(window=4).corr(adj_volume.rolling(window=5).mean())
        part_1 = linear_decay(temp1, 7).rank(axis=1, pct=True)
        open_rank = adj_OPEN.rank(axis=1, pct=True)
        adv_rank = adj_volume.rolling(window=15).mean().rank(axis=1, pct=True)
        corr_ = open_rank.rolling(window=20).corr(adv_rank)
        ts_argmin = corr_.rolling(window=8).apply(np.argmin)
        part_2 = linear_decay(rolling_rank(ts_argmin, 6), 8).rank(axis=1, pct=True)
        factor = part_1 - part_2
        factor[rolling_null(close, 42)] = np.nan
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '099':
        adv60 = adj_volume.rolling(window=60).mean()
        temp1 = ((adj_high + adj_low) / 2).rolling(window=19).sum()
        temp2 = adv60.rolling(window=19).sum()
        part_1 = temp1.rolling(window=8).corr(temp2).rank(axis=1, pct=True)
        part_2 = adj_low.rolling(window=6).corr(adj_volume).rank(axis=1, pct=True)
        factor = part_1
        factor[part_1 < part_2] = 1
        factor[part_1 >= part_2] =0
        factor = -1 * factor
        factor[rolling_null(close, 87)] = np.nan
        factor[rolling_stop(stop, 19)] = np.nan
    elif factor_id == '100':
        temp1 = adj_volume * ((adj_close - adj_low) - (adj_high - adj_close)) / (adj_high - adj_low)
        temp2 = ind_neutral(temp1.rank(axis=1, pct=True), ind)
        part_1 = (temp2.T / temp2.sum(axis=1)).T
        temp3 = adj_volume.rolling(window=20).mean()
        corr_ = adj_close.rolling(window=5).corr(temp3)
        close_rank = adj_close.rolling(30).apply(np.argmin).rank(axis=1, pct=True)
        temp4 = ind_neutral(corr_ - close_rank, ind)
        part_2 = (temp4.T / temp4.sum(axis=1)).T
        factor = -1 * (1.5 * part_1 - part_2) * (adj_volume / adj_volume.rolling(window=20).mean())
        factor[rolling_null(close, 30)] = np.nan
        factor[rolling_stop(stop, 30)] = np.nan
    elif factor_id == '101':
        factor = (close - OPEN) / ((high - low) + 0.001)
        factor[stop.notnull()] = np.nan
    elif factor_id == 'vol':
        returns = adj_close.pct_change()
        factor = pd.DataFrame()
        for t in returns.index:
            not_stop = stop[:t][-21:].isnull().all(axis=0)
            vol = adj_close[:t][-21:].pct_change()[1:].std(skipna=False)
            vol = pd.Series(np.where(not_stop, vol, np.nan), index=not_stop.index)
            vol.index.name = t
            factor[t] = vol
        factor = factor.T
    elif factor_id == 'PeterLynch':
        fcf[fcf < 0] = np.nan
        asset_debt[asset_debt < 0] = np.nan
        factor = (close / fcf).rank(axis=1, pct=True) * asset_debt.rank(axis=1, pct=True)
    else:
        print 'factor id is wrong'
    # 将涨跌停股票的因子设为na
    if no_zdt:
        increase_stop = np.round(close * 1.1, 2).shift(1)
        decrease_stop = np.round(close * 0.9, 2).shift(1)
        bool_temp = np.logical_or(close == increase_stop, decrease_stop == close)
        factor = np.where(bool_temp, np.nan, factor)
        factor = pd.DataFrame(factor, columns=close.columns, index=close.index)
    # 将st股票的因子值设为na
    if no_st:
        factor[st.notnull()] = np.nan
    print '计算完成'
    if save:
        print '保存因子'
        save_factor(factor, factor_id, no_zdt)
    return factor


def factor_summary(factor, period=1):
    percentiles = np.arange(0.1, 1, 0.1)
    summary = pd.DataFrame()
    for i in range(0, len(factor.index), period):
        t = factor.index[i]
        summary[t] = factor.ix[t].dropna().describe(percentiles)
    return summary.T


def ic(factor, return_mode, period, data, by_day=False):
    close = data['close']
    OPEN = data['OPEN']
    price_adj_f = data['price_adj_f']
    adj_close = close * price_adj_f
    adj_OPEN = OPEN * price_adj_f
    if return_mode == 'close-close':
        returns = adj_close.pct_change(period).shift(-period).stack()
    elif return_mode == 'close-open':
        returns = (adj_close.shift(-(period-1)) / adj_OPEN - 1).shift(-period).stack()
    elif return_mode == 'open-open':
        returns = adj_OPEN.pct_change(period).shift(-(period + 1)).stack()
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    returns.name = 'returns'
    factor = factor.stack()
    factor.name = 'factor'
    temp = pd.concat([returns, factor], axis=1).dropna(axis=0, how='any')
    if by_day:
        ic_list = []
        days = temp.index.levels[0]
        for t in days:
            ic_list.append(temp.ix[t].corr().iloc[0][1])
        ic_series = pd.Series(ic_list, index=days)
        return ic_series
    else:
        return temp.corr().iloc[0][1]


def group_analysis(factor, data, return_mode, period=1, bins=10, cut_mode='quantile'):
    close = data['close']
    OPEN = data['OPEN']
    price_adj_f = data['price_adj_f']
    adj_close = close * price_adj_f
    adj_OPEN = OPEN * price_adj_f
    if return_mode == 'close-close':
        returns = adj_close.pct_change()
    # elif return_mode == 'close-open':
    #    returns = (adj_close.shift / adj_OPEN - 1)
    elif return_mode == 'open-open':
        returns = adj_OPEN.pct_change()
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    days = len(factor.index)
    group_mean = pd.DataFrame()
    group_return = pd.DataFrame()
    stock_num = pd.DataFrame()
    for i in range(0, days, period):
        t = factor.index[i]
        temp = factor.ix[t].dropna()
        if cut_mode == 'quantile':
            temp.sort_values(ascending=True, inplace=True)
            temp[:] = range(len(temp))
            # 根据实际数值大小分组，可能某一个数值的股票数量太多，两个bin的边界相同
            # 这里把因子排序，并用一个从1开始的序列替代原数值，在分组较多时，可能不准确
            groups = pd.qcut(temp, bins, labels=False)
        elif cut_mode == 'interval':
            groups = pd.cut(temp, bins, labels=False)
        group_mean[t] = factor.ix[t].dropna().groupby(groups).mean()
        stock_num[t] = factor.ix[t].dropna().groupby(groups).count()
        if return_mode == 'close-close':
            period_returns = returns.ix[t:].ix[1: period+1].groupby(groups, axis=1).mean()
        elif return_mode == 'open-open':
            period_returns = returns.ix[t:].ix[2: period+2].groupby(groups, axis=1).mean()
        group_return = pd.concat([group_return, period_returns])
    a_return = returns.mean(axis=1)
    a_mean = factor.mean(axis=1)
    # group_mean.index = group_mean.index + 1
    # group_return.index = group_return.index + 1
    group_mean.index.name = 'Group'
    group_mean = group_mean.T
    group_mean['Factor Average'] = a_mean
    group_return.columns.name = 'Group'
    group_return['Simple Average'] = a_return
    stock_num.index.name = 'Group'
    stock_num = stock_num.T
    return {'group_mean': group_mean, 'group_return': group_return, 'stock_num': stock_num}


def _analysis(factor, data, return_mode, stock_number, ascending=True, period=1):
    close = data['close']
    OPEN = data['OPEN']
    price_adj_f = data['price_adj_f']
    adj_close = close * price_adj_f
    adj_OPEN = OPEN * price_adj_f
    if return_mode == 'close-close':
        returns = adj_close.pct_change(period).shift(-period)
    elif return_mode == 'close-open':
        returns = (adj_close.shift(-(period-1)) / adj_OPEN - 1).shift(-period)
    elif return_mode == 'open-open':
        returns = adj_OPEN.pct_change(period).shift(-(period + 1))
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    days = len(factor.index)
    _mean = []
    _return = []
    _t = []
    stocks = pd.DataFrame()
    factors = pd.DataFrame()
    for i in range(0, days, period):
        t = factor.index[i]
        _t.append(t)
        temp = factor.ix[t].dropna()
        temp.sort_values(ascending=ascending, inplace=True)
        stock_to_hold = temp[:stock_number].index
        stocks[t] = stock_to_hold
        factors[t] = temp[stock_to_hold]
        _return.append(returns.ix[t][stock_to_hold].mean())
        _mean.append(temp[stock_to_hold].mean())
    _mean = pd.Series(_mean, index=_t, name='factor_mean')
    _return = pd.Series(_return, index=_t, name='mean_return')
    return {'factor_mean': _mean, 'mean_return': _return,
            'stocks': stocks.T, 'factors': factors.T}


high = import_data('LZ_GPA_QUOTE_THIGH')
low = import_data('LZ_GPA_QUOTE_TLOW')
close = import_data('LZ_GPA_QUOTE_TCLOSE')
OPEN = import_data('LZ_GPA_QUOTE_TOPEN')
value = import_data('LZ_GPA_QUOTE_TVALUE')
volume = import_data('LZ_GPA_QUOTE_TVOLUME')
cap = import_data('LZ_GPA_VAL_A_TCAP')
stop = import_data('LZ_GPA_SLCIND_STOP_FLAG')
price_adj_f = import_data('LZ_GPA_CMFTR_CUM_FACTOR')
st = import_data('LZ_GPA_SLCIND_ST_FLAG')
ind = import_data('LZ_GPA_INDU_SW')

fcf = import_data('LZ_GPA_FIN_IND_FCFFPS')
asset_debt = import_data('LZ_GPA_FIN_IND_DEBTTOASSETS')

data = {'high': high,
        'low': low,
        'close': close,
        'OPEN': OPEN,
        'value': value,
        'volume': volume,
        'cap': cap,
        'stop': stop,
        'st': st,
        'ind': ind,
        'fcf': fcf,
        'asset_debt': asset_debt,
        'price_adj_f': price_adj_f}


factor = factor_cal(data, '019', no_zdt=False, no_st=False, no_new=True, save=True)
save_factor(factor, '015', no_zdt=True)

factor_006 = factor_cal(data, '006', no_zdt=False)
factors = pd.concat([factor.stack(), factor_006.stack()], axis=1)
days = factors.index.levels[0]
for d in days:
    print factors.ix[d].corr().iloc[0][1]


factor_ = factor_cal()
factor_.to_csv('F:\Factors\Volatility/LowVol_nostop_adj.csv')
factor_ = factor_.rolling(window=12).mean()
temp_ = factor_.stack()
temp_.plot(kind='hist', bins=20)
temp_[temp_>0.1]

summary = factor_summary(factor['2013-01-01':].dropna(axis=0, how='all'))
summary.ix[:, ['mean', 'std', '10%', '90%']].plot(figsize=(10, 10))

ic_series = ic(factor, 'close-close', 20, data, by_day=True)
title = '60 days moving average of IC (absolute value)'
np.abs(ic_series).rolling(window=60).mean().plot(figsize=(14, 7), title=title)


temp = np.abs(factor_.T - factor_.mean(axis=1)).T
factor_name = 'alpha#101_nozdt_abs_1D'
temp.index.name = factor_name
temp.to_csv('F:\Strategies\World_Quant_Alphas\#%s\%s.csv' %
            ('101', factor_name))

stt = factor['2013-01-07':].dropna(axis=0, how='all')
stt = np.abs(stt.T - stt.mean(axis=1)).T
stt.mean(axis=1).mean()

a = group_analysis(stt, data, 'close-close', period=50, bins=10, cut_mode='quantile')
a['group_return'].mean().plot(kind='bar')
a['group_mean'].mean().plot()
(a['group_return'] + 1).cumprod().plot(figsize=(14, 7))
a['group_mean']['Factor Average'].plot(figsize=(14, 7))
a['group_return'].columns.name

a = _analysis(stt, data, 'close-close', stock_number=10, ascending=False, period=10)
a['mean_return'].mean()
a['factor_mean'].min()
(a['mean_return'] + 1).cumprod().plot(figsize=(14, 7))
a['stocks'].to_csv('F:\Strategies\World_Quant_Alphas/10_stocks.csv')
a['factors'].head()


# ============因子相关性分析============================

factor_list = ['002', '013', '015', '016', '041', '044', '055', '083']
factors = pd.DataFrame()
for i in factor_list:
    factor = factor_cal(data, i, no_zdt=False, no_st=False, no_new=False, save=False)
    factor = factor.stack()
    factor.name = i
    factors = pd.concat([factors, factor], axis=1)

factors.corr().to_csv('correlation.csv')
