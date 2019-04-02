import pandas as pd
import numpy as np
import time
import os
import math
from datetime import datetime
import csv

def record_csv(file, asset, tensor3, price):
    try:
        with open(file) as f:
            numline = 10
    except:
        numline = 0
    entry = [str(datetime.now().strftime("%Y-%b-%d %H:%M")), asset, tensor3[0][0], tensor3[0][1], tensor3[0][2], tensor3[0][3], tensor3[0][4],price]
    with open(file, 'a', newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        if numline == 0:
            wr.writerow(['time', 'asset', 'very_bear', 'bear', 'neutral', 'bull', 'very_bull', 'price'])
        wr.writerow(entry)
        fp.close()

def record_trade(file, asset, action, price, qty, total, orderId):
    try:
        with open(file) as f:
            numline = 10
    except:
        numline = 0
    entry = [str(datetime.now().strftime("%Y-%b-%d %H:%M")), asset, action, price, qty, total, orderId]
    with open(file, 'a', newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        if numline == 0:
            wr.writerow(['time', 'asset', 'action', 'price', 'qty', 'total', 'orderId'])
        wr.writerow(entry)
        fp.close()

#Drop NAN values
def dropNAN(df, column):
    df = df[np.isfinite(df[column])]
    return df

def dropAllNAN(df):
    for i in df.columns:
        df = df[np.isfinite(df[i])]
    return df

def find_support(df, timeframe, dec_places):
    conditions = [
    (df['Close'] < df['Close'].shift(1)) & (df['Close'] < df['Close'].shift(-1)) &
    (df['Close'] < df['Close'].shift(2)) & (df['Close'] < df['Close'].shift(-2)) &
    (df['Close'] < df['Close'].shift(3)) & (df['Close'] < df['Close'].shift(-3))]
    choices = [round(df['Close'], dec_places)]
    df['Support'] = np.select(conditions, choices, default=0)
    return df

def macd_trigger(df, macd_trigger):
    conditions = [
    (df[macd_trigger].shift(1).rolling(5).mean() < 0) &
    (df[macd_trigger] > 0)]
    df['higher'] = np.select(conditions, [1], default=0)

    conditions2 = [
    (df[macd_trigger].shift(1).rolling(5).mean() > 0) &
    (df[macd_trigger] < 0)]
    df['lower'] = np.select(conditions2, [-1], default=0)

    df['MACD Trigger'] = df['lower'] + df['higher']
    df.drop(columns = ['lower', 'higher'])

    return df


# cluster list into groups --used for calculating support
def cluster(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][0]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

# Returns support levels. COUNT is amount of times a level must be tested to be considered support. larger data = larger numbers
def support_price(df, count, decimal_places):
    support_levels = []
    lows = df['Low'].tolist()
    maxgap = (sum(lows)/len(lows)) * 0.002
    grouped = cluster(lows, maxgap=maxgap)
    for i in grouped:
        if len(i) > count:
            support_level = round(sum(i)/len(i), decimal_places)
            support_levels.append(support_level)
    support_price = min(support_levels)
    return support_price

# same thing as support price but in pandas DF
def support_prices(df, timeframe, count, decimal_places):
    for index, row in df.iterrows():
        if index > timeframe:
            support_levels = []
            lows = df.iloc[index-timeframe:index]['Low'].tolist()
            maxgap = (sum(lows)/len(lows)) * 0.002
            grouped = cluster(lows, maxgap=maxgap)
            for i in grouped:
                if len(i) > count:
                    support_level = round(sum(i)/len(i), decimal_places)
                    support_levels.append(support_level)
            if support_levels:
                support_price = min(support_levels)
            else:
                for i in grouped:
                    if len(i) > 5:
                        support_level = round(sum(i)/len(i), decimal_places)
                        support_levels.append(support_level)
                if support_levels:
                    support_price = min(support_levels)
                else:
                    support_price = min(lows)

        else:
            support_price = df.iloc[index]['Low']
        df.at[index, 'Support Level'] = support_price 
    return df

def supportlvl(df, timeframe, count, decimal_places):
    for index, row in df.iterrows():
        if index > timeframe:
            support_levels = []
            lows = df['Low'].iloc[index-timeframe:index].tolist()
            maxgap = (sum(lows)/len(lows)) * 0.002
            grouped = cluster(lows, maxgap=maxgap)
            for i in grouped:
                if len(i) > count:
                    support_level = round(sum(i)/len(i), decimal_places)
                    support_levels.append(support_level)
            if support_levels:
                support_price = min(support_levels)
            else:
                for i in grouped:
                    if len(i) > 5:
                        support_level = round(sum(i)/len(i), decimal_places)
                        support_levels.append(support_level)
                if support_levels:
                    support_price = min(support_levels)
                else:
                    support_price = min(lows)

        else:
            support_price = df['Low'].iloc[index]
        df.at[index, 'Support'] = support_price 
    return df

def reslvl(df, timeframe, count, decimal_places):
    for index, row in df.iterrows():
        if index > timeframe:
            res_levels = []
            highs = df.iloc[index-timeframe:index]['High'].tolist()
            maxgap = (sum(highs)/len(highs)) * 0.002
            grouped = cluster(highs, maxgap=maxgap)
            for i in grouped:
                if len(i) > count:
                    res_level = round(sum(i)/len(i), decimal_places)
                    res_levels.append(res_level)
            if res_level:
                res_price = max(res_levels)
            else:
                for i in grouped:
                    if len(i) > 5:
                        res_level = round(sum(i)/len(i), decimal_places)
                        res_levels.append(res_level)
                if res_levels:
                    res_price = min(res_levels)
                else:
                    res_price = min(highs)

        else:
            res_price = df.iloc[index]['High']
        df.at[index, 'Resistence'] = res_price 
    return df

def ratio(df, var, ma):
    label = str(var) + '/' + str(ma) + ' Ratio'
    df[label] = round((df[var] / df[ma]), 2)
    return df


def percent_above_ma(df, timeframe):
    name = 'Percent Above MA ' + str(timeframe)
    total_up = df['Above MA'].rolling(timeframe).sum()
    df[name] = round((total_up/timeframe)*100)
    return df

def above_ma(df, ma):
    conditions = [(df['Close'] > df[ma])]
    df['Above MA'] = np.select(conditions, [1], default=0)
    return df

def VWAP(df, timeframe):
    df['VWAP'] = df['Base Volume'].rolling(timeframe).sum()/df['Volume'].rolling(timeframe).sum()
    return df

def distance_from_MA(df, ma):
    distance_MA = pd.Series(round(((df['Close'] - df[ma]) / df['Close']) * 100), name = 'Distance From ' + str(ma))
    df = df.join(distance_MA)
    return df

#get range for given timeframe
def get_range(df, timeframe):
    low = pd.Series(df['Low'].shift(1).rolling(timeframe).min(), name = str(timeframe) + ' Period Low')
    high = pd.Series(df['High'].shift(1).rolling(timeframe).max(), name = str(timeframe) + ' Period High')
    range_difference = pd.Series(round((((high-low) / low) * 100), 2), name = str(timeframe) + ' Period High-Low Range')
    df = df.join(high)
    df = df.join(low)
    df = df.join(range_difference)
    return df

def standard_deviation(df, data_column, n):  
    df = df.join(pd.Series(df[data_column].rolling(n).std(), name = str(n) + ' Period Std Dev'))
    return df  

#Log return
def log_return(df, n):
    periodReturn = pd.Series((np.log(df['Close']/df['Close'].shift(n))), name = str(n) + ' Period Log Return')
    periodReturn = round(periodReturn * 100, 2)
    df = df.join(periodReturn)
    return df

#Standard return
def std_return(df, n):
    periodReturn = pd.Series(round((((df['Close']-df['Close'].shift(n))/df['Close'].shift(n)) * 100), 2), name = str(n) + ' Period Return')
    df = df.join(periodReturn)
    return df

#Moving Average  
def MA(df, variable, n):  
    MA = pd.Series(df[variable].rolling(n).mean(), name = str(variable) + ' ' + str(n) + ' MA')
    df = df.join(MA)  
    return df

#Median
def Median(df, variable, n):  
    med = pd.Series(df[variable].rolling(n).median(), name = str(variable) + ' ' + str(n) + ' Median')
    df = df.join(med)  
    return df

#Median Mean
def med_mean(df, high, low, timeframe):
    df = median_balance_point(df, high, low)
    mean = pd.Series(round(df['Med Balance Point'].rolling(timeframe).mean(), 2), name = 'Med Mean')
    df = df.join(mean)
    return df

#Exponential Moving Average  
def EMA(df, variable, n):  
    EMA = pd.Series(pd.ewma(df[variable], span = n, min_periods = n - 1), name = str(variable) + ' ' + str(n) + ' EMA')  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    df = df.join(ATR)  
    return df

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['High'])  
    R2 = pd.Series(PP + df['High'] - df['Low'])  
    S2 = pd.Series(PP - df['High'] + df['Low'])  
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df

#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
    df = df.join(SOk)  
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)  
def STO(df,  nK, nD, nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df  
# Stochastic Oscillator, SMA smoothing, nS = slowing (1 if no slowing)  
def STO(df, nK, nD,  nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.rolling(window=nD, center=False).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.rolling(window=nS, center=False).mean()  
    SOd = SOd.rolling(window=nS, center=False).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df  
#Trix  
def TRIX(df, n):  
    EX1 = pd.ewma(df['Close'], span = n, min_periods = n - 1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)  
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)  
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))  
    df = df.join(Trix)  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast=12, n_slow=26):  
    EMAfast = pd.Series(pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(round(EMAfast - EMAslow, 8), name = 'MACD')  
    MACDsign = pd.Series(round(pd.ewma(MACD, span = 9, min_periods = 8)), name = 'MACD Signal')  
    MACDdiff = pd.Series(round(MACD - MACDsign), name = 'MACD Difference')
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

#Mass Index  
def MassI(df):  
    Range = df['High'] - df['Low']  
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)  
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)  
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df

#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['Close'].diff(r1 - 1)  
    N = df['Close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(r2 - 1)  
    N = df['Close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['Close'].diff(r3 - 1)  
    N = df['Close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['Close'].diff(r4 - 1)  
    N = df['Close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df = df.join(RSI)  
    return df

#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['Close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))  
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))  
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))  
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

#Chaikin Oscillator  
def Chaikin(df):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')  
    df = df.join(Chaikin)  
    return df

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['Volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:  
            OBV.append(df.get_value(i + 1, 'Volume'))  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:  
            OBV.append(0)  
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:  
            OBV.append(-df.get_value(i + 1, 'Volume'))  
        i = i + 1  
    OBV = pd.Series(OBV)  
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))  
    df = df.join(CCI)  
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['Close'].diff(int(n * 11 / 10) - 1)  
    N = df['Close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(int(n * 14 / 10) - 1)  
    N = df['Close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N  
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df
