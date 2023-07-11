from django.shortcuts import render
from trading_bot_django import settings
import asyncio
import datetime
from tasks.harmonics import *
from tasks.news import *
from sklearn.preprocessing import LabelEncoder
import numpy as np
import datetime
import time
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from apscheduler.schedulers.background import BackgroundScheduler
import os
from django.shortcuts import render, redirect, get_object_or_404
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta
from collections import Counter


token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '615ae0df-2198-4162-9b23-34a4285baa35'

label_encoder = LabelEncoder()

rsi_period = 14
smi_period = 14
atr_period = 14
atr_multiplier = 2.0
bollinger_period = 20
bollinger_std = 2
ema_period = 20
adx_period = 14
n_fast = 3
n_slow = 6
n_signal = 4
global symbols,timeframes

symbols=['AUDMXNm', 'AUDNOKm', 'AUDNZDm', 'AUDPLNm', 'AUDSEKm', 'AUDSGDm',
    'AUDTRYm', 'AUDUSDm', 'AUDUSX', 'AUDZARm', 'AUS200m', 'AUXAUD', 'AUXTHB', 'AUXUSD', 'AUXZAR',
    'AVGOm', 'BABAm', 'BACm', 'BATUSDm', 'BAm', 'BCHUSDm', 'BIIBm', 'BMYm', 'BNBUSDm', 'BTCAUDm',
    'BTCCNHm', 'BTCJPYm', 'BTCKRWm', 'BTCTHBm', 'BTCUSDm', 'BTCXAGm', 'BTCXAUm', 'BTCZARm',
    'CADCHFm', 'CADCZKm', 'CADJPYm', 'CADMXNm', 'CADNOKm', 'CADPLNm', 'CADTRYm', 'CHFDKKm',
    'CHFHUFm', 'CHFJPYm', 'CHFMXNm', 'CHFNOKm', 'CHFPLNm', 'CHFSEKm', 'CHFSGDm', 'CHFTRYm',
    'CHFZARm', 'CHTRm', 'CMCSAm', 'CMEm', 'COSTm', 'CSCOm', 'CSXm', 'CVSm', 'CZKPLNm', 'Cm',
    'DE30m', 'DKKCZKm', 'DKKHUFm', 'DKKJPYm', 'DKKPLNm', 'DKKSGDm', 'DKKZARm', 'DOTUSDm', 'DXYm',
    'EAm', 'EBAYm', 'ENJUSDm', 'EQIXm', 'ETHUSDm', 'EURAUDm', 'EURAUX', 'EURCADm', 'EURCHFm',
    'EURCZKm', 'EURDKKm', 'EURGBPm', 'EURGBX', 'EURHKDm', 'EURHKX', 'EURHUFm', 'EURJPX', 'EURJPYm',
    'EURMXNm', 'EURNOKm', 'EURNZDm', 'EURPLNm', 'EURSEKm', 'EURSGDm', 'EURTRYm', 'EURUSDm', 'EURUSX',
    'EURZARm', 'EUXAUD', 'EUXEUR', 'EUXGBP', 'EUXTHB', 'EUXUSD', 'EUXZAR', 'FBm', 'FILUSDm', 'FR40m',
    'Fm', 'GBPAUDm', 'GBPAUX', 'GBPCADm', 'GBPCHFm', 'GBPCZKm', 'GBPDKKm', 'GBPHKX', 'GBPHUFm',
    'GBPILSm', 'GBPJPX', 'GBPJPYm', 'GBPMXNm', 'GBPNOKm', 'GBPNZDm', 'GBPPLNm', 'GBPSEKm', 'GBPSGDm',
    'GBPTRYm', 'GBPUSDm', 'GBPUSX', 'GBPZARm', 'GBXAUD', 'GBXGBP', 'GBXTHB', 'GBXUSD', 'GBXZAR',
    'GILDm', 'GOOGLm', 'HDm', 'HK50m', 'HKDJPYm', 'HKXHKD', 'HKXTHB', 'HKXZAR', 'HUFJPYm', 'IBMm',
    'IN50m', 'INTCm', 'INTUm', 'ISRGm', 'JNJm', 'JP225m', 'JPMm', 'JPXJPY', 'KOm', 'LINm', 'LLYm',
    'LMTm', 'LTCUSDm', 'MAm', 'MCDm', 'MDLZm', 'METAm', 'MMMm', 'MOm', 'MRKm', 'MSFTm', 'MSm', 'MXNJPYm',
    'NADUSD', 'NFLXm', 'NKEm', 'NOKDKKm', 'NOKJPYm', 'NOKSEKm', 'NVDAm', 'NZDCADm', 'NZDCHFm',
    'NZDCZKm', 'NZDDKKm', 'NZDHUFm', 'NZDJPYm', 'NZDMXNm', 'NZDNOKm', 'NZDPLNm', 'NZDSEKm', 'NZDSGDm',
    'NZDTRYm', 'NZDUSDm', 'NZDZARm', 'ORCLm', 'PEPm', 'PFEm', 'PGm', 'PLNDKKm', 'PLNHUFm', 'PLNJPYm',
    'PLNSEKm', 'PMm', 'PYPLm', 'REGNm', 'SBUXm', 'SEKDKKm', 'SEKJPYm', 'SEKPLNm', 'SGDHKDm',
    'SGDJPYm', 'SNXUSDm', 'SOLUSDm', 'STOXX50m', 'THBJPX', 'TMOm', 'TMUSm', 'TRXUSD', 'TRYDKKm',
    'TRYJPYm', 'TRYZARm', 'TSLAm', 'Tm', 'UK100m', 'UKOILm', 'UNHm', 'UNIUSDm', 'UPSm', 'US30_x10m',
    'US30m', 'US500_x100m', 'US500m', 'USDAED', 'USDAEDm', 'USDAMD', 'USDAMDm', 'USDARS', 'USDARSm',
    'USDAZN', 'USDAZNm', 'USDBDT', 'USDBDTm', 'USDBGN', 'USDBGNm', 'USDBHD', 'USDBHDm', 'USDBND',
    'USDBNDm', 'USDBRL', 'USDBRLm', 'USDBYN', 'USDBYR', 'USDCADm', 'USDCHFm', 'USDCLP', 'USDCLPm',
    'USDCNHm', 'USDCNY', 'USDCNYm', 'USDCOP', 'USDCOPm', 'USDCRC', 'USDCZKm', 'USDDKKm', 'USDDZD',
    'USDDZDm', 'USDEGP', 'USDEGPm', 'USDGEL', 'USDGELm', 'USDGHS', 'USDGHSm', 'USDHKDm', 'USDHKX',
    'USDHRK', 'USDHRKm', 'USDHUF', 'USDHUFm', 'USDIDR', 'USDIDRm', 'USDILSm', 'USDINR', 'USDINRm',
    'USDIRR', 'USDISK', 'USDISKm', 'USDJOD', 'USDJODm', 'USDJPX', 'USDJPYm', 'USDKES', 'USDKESm',
    'USDKGS', 'USDKGSm', 'USDKHR', 'USDKRW', 'USDKRWm', 'USDKWD', 'USDKWDm', 'USDKZT', 'USDKZTm',
    'USDLAK', 'USDLBP', 'USDLBPm', 'USDLKR', 'USDLKRm', 'USDMAD', 'USDMADm', 'USDMMK', 'USDMXNm',
    'USDMYR', 'USDMYRm', 'USDNGN', 'USDNGNm', 'USDNOKm', 'USDNPR', 'USDNPRm', 'USDOMR', 'USDOMRm',
    'USDPAB', 'USDPEN', 'USDPHP', 'USDPHPm', 'USDPKR', 'USDPKRm', 'USDPLNm', 'USDPYG', 'USDQAR',
    'USDQARm', 'USDROL', 'USDRON', 'USDRONm', 'USDRUB', 'USDRUBm', 'USDRUR', 'USDRURm', 'USDRWF',
    'USDSAR', 'USDSARm', 'USDSCR', 'USDSEKm', 'USDSGDm', 'USDSYP', 'USDSYPm', 'USDTHBm', 'USDTJS',
    'USDTJSm', 'USDTMT', 'USDTMTm', 'USDTND', 'USDTNDm', 'USDTRYm', 'USDTUSD', 'USDTWD', 'USDTWDm',
    'USDTZS', 'USDUAH', 'USDUAHm', 'USDUGX', 'USDUGXm', 'USDUYU', 'USDUZS', 'USDUZSm', 'USDVND',
    'USDVNDm', 'USDVUV', 'USDVUVm', 'USDXAF', 'USDXOF', 'USDXOFm', 'USDZARm', 'USDZMW', 'USOILm',
    'USTEC_x100m', 'USTECm', 'USXJPY', 'USXRUB', 'USXTHB', 'USXUSD', 'USXZAR', 'VRTXm', 'VZm', 'Vm',
    'WFCm', 'WMTm', 'XAGAUDm', 'XAGEURm', 'XAGGBPm', 'XAGJPYm', 'XAGUSDm', 'XAUAUDm', 'XAUEURm',
    'XAUGBPm','XAUUSDm','XNGUSDm','XOMm','XPDUSDm','XPTUSDm','XRPUSDm','XTZUSDm','ZARJPX','ZARJPYm'
    ]


timeframes  = ['1m','5m','15m','30m','1h','4h','1d','1w']

async def get_candles(timeframe,symbol):
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']
    timeframe=timeframe
    symbol=symbol
    if initial_state not in deployed_states:
        # wait until account is deployed and connected to broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
    try:
        # Fetch historical price data
        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
        # Convert candles to DataFrame
        print(type(candles))
        return candles
    except Exception as e:
        return f"Error retrieving candle data: {e}"


async def get_candles_m(timeframe,symbol):
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']
    timeframe=timeframe
    symbol=symbol
    if initial_state not in deployed_states:
        # wait until account is deployed and connected to broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
    try:
        # Create an empty dataframe to store the candlestick data
        df = pd.DataFrame()

        # retrieve last 10K 1m candles
        pages = 15
        print(f'Downloading {pages}K latest candles for {symbol}')
        started_at = datetime.now().timestamp()
        start_time = None
        candles = None
        for i in range(pages):
            # the API to retrieve historical market data is currently available for G1 only
            candles = await account.get_historical_candles(symbol, '1m', start_time)
            print(f'Downloaded {len(candles) if candles else 0} historical candles for {symbol}')
            
            
            if candles:
                start_time = candles[0]['time']
                start_time.replace(minute=start_time.minute - 1)
                print(f'First candle time is {start_time}')
                
                # Create a new dataframe for each iteration and add it to the main dataframe
                new_df = pd.DataFrame(candles)
                df = pd.concat([df, new_df], ignore_index=True)
                print(f'Candles added to dataframe')
                print(df)
        return df

    except Exception as e:
        return f"Error retrieving candle data: {e}"


def extract(candles):
    data=pd.DataFrame(candles)
    # Extract necessary columns (e.g., 'Timestamp', 'Open', 'High', 'Low', 'Close')
    date = data['time']
    open_price = data['open']
    high_price = data['high']
    low_price = data['low']
    close_price = data['close']
    n_fast = 3
    n_slow = 6
    n_signal = 4


    # Calculate the simple moving average (SMA)
    def calculate_sma(data, period):
        sma = data.rolling(window=period, min_periods=period).mean()
        return sma

    # Calculate the exponential moving average (EMA)
    def calculate_ema(data, period):
        ema = data.ewm(span=period, adjust=False).mean()
        return ema

    # Calculate the relative strength index (RSI)
    def calculate_rsi(data, period):
        delta = data.diff()
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = abs(losses.rolling(window=period).mean())
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Calculate the MACD line, signal line, and histogram
    def calculate_macd(data, n_fast, n_slow, n_signal):
        closing_prices = data['close']
        ema_fast = calculate_ema(closing_prices, n_fast)
        ema_slow = calculate_ema(closing_prices, n_slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, n_signal)
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line

    # Calculate the required indicators
    sma_period = 20
    sma = calculate_sma(close_price, sma_period)

    ema_period = 20
    ema = calculate_ema(close_price, ema_period)

    rsi_period = 14
    rsi = calculate_rsi(close_price, rsi_period)

    macd_line, signal_line = calculate_macd(data, n_fast, n_slow, n_signal)

    # Find trend lines using MACD
    uptrend = macd_line > signal_line
    downtrend = macd_line < signal_line

    # Find trend line breakouts
    breakout_up = (close_price.shift(1) > sma.shift(1)) & (close_price < sma)
    breakout_down = (close_price.shift(1) < sma.shift(1)) & (close_price > sma)

    # Calculate the number of pips covered from breakout to subsequent swing high/low
    #pip_movement = high_price - low_price.shift(1)
    sma_period = 20
    sma = calculate_sma(close_price, sma_period)

    ema_period = 20
    ema = calculate_ema(close_price, ema_period)

    rsi_period = 14
    rsi = calculate_rsi(close_price, rsi_period)

    macd_line, signal_line = calculate_macd(data, n_fast, n_slow, n_signal)

    # Find trend lines using moving averages
    #uptrend = close_price > sma
    #downtrend = close_price < sma

    # Define the number of candlesticks to observe
    num_candlesticks = 50
    df = pd.DataFrame(columns=['Trend','Stime','Etime', 'Duration', 'Pip Movement'])

    # Initialize data frames to store trends, time periods, and pip movements
    trend_df = pd.DataFrame(columns=['Trend'])
    time_df = pd.DataFrame(columns=['Time Period'])
    pips_df = pd.DataFrame(columns=['Pip Movement'])


    # Check for uptrend and downtrend over the specified number of candlesticks
    for i in range(num_candlesticks, len(date)-1):
        '''if all(uptrend[i - num_candlesticks : i + 1]):
            pip_movement = close_price[i] - open_price[i - num_candlesticks]
            trend_df = trend_df.append({'Trend': 'Uptrend'}, ignore_index=True)
            time_df = time_df.append({'Time Period': f"{date[i - num_candlesticks + 1]} to {date[i + 1]}"}, ignore_index=True)
            pips_df = pips_df.append({'Pip Movement': pip_movement}, ignore_index=True)
        elif all(downtrend[i - num_candlesticks : i + 1]):
            pip_movement = open_price[i - num_candlesticks] - close_price[i]
            trend_df = trend_df.append({'Trend': 'Downtrend'}, ignore_index=True)
            time_df = time_df.append({'Time Period': f"{date[i - num_candlesticks + 1]} to {date[i + 1]}"}, ignore_index=True)
            pips_df = pips_df.append({'Pip Movement': pip_movement}, ignore_index=True)'''
        if all(uptrend[i - num_candlesticks: i + 1]):
            pip_movement = close_price[i] - open_price[i - num_candlesticks]
            df = df.append({'Trend': 'Uptrend', 'Stime': f"{date[i - num_candlesticks]}",'Etime': f"{date[i]}",  'Duration': num_candlesticks, 'Pip Movement': pip_movement}, ignore_index=True)
        elif all(downtrend[i - num_candlesticks: i + 1]):
            downtrend_indices = np.where(downtrend[:i])[0]
            if len(downtrend_indices) > 0:
                duration = i - downtrend_indices[-1]
            else:
                duration = i
            pip_movement = open_price[i - num_candlesticks] - close_price[i]
            df = df.append({'Trend': 'Downtrend', 'Stime': f"{date[i - num_candlesticks]}",'Etime': f"{date[i]}", 'Duration': duration, 'Pip Movement': pip_movement}, ignore_index=True)
    return df

def knn_train(df,symbol):
    # Step 1: Data Preparation
    # Assuming you have a CSV file named 'data.csv' with columns 'feature1', 'feature2', 'trend', 'time', 'pips', 'duration'

    # Load the data from CSV file
    data = df
    print('deffe   =',df)
    if not df.empty:
        # Extract features and target variables
        timestamps = data['Stime'].values
        y_trend = data['Trend'].values
        times= data['Etime'].values
        y_pips = data['Pip Movement'].values
        y_duration = data['Duration'].values
        #datetime_objects = [datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z") for timestamp in timestamps]
        #datetime_objects = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in timestamps]
        datetime_objects = [np.datetime64(timestamp).astype(datetime) for timestamp in timestamps]
        # Extract relevant features from datetime objects
        years = [dt.year for dt in datetime_objects]
        months = [dt.month for dt in datetime_objects]
        days = [dt.day for dt in datetime_objects]
        hours = [dt.hour for dt in datetime_objects]
        minutes = [dt.minute for dt in datetime_objects]
        seconds = [dt.second for dt in datetime_objects]

        # Assign extracted features to X for training and testing
        X = np.column_stack((years, months, days, hours, minutes, seconds))
        print(X)

        #datetime_objects = [datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z") for timestamp in times]
        datetime_objects = [np.datetime64(timestamp).astype(datetime) for timestamp in times]
        #datetime_objects = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in times]
        data['symbol'] = [f'{symbol}'] * len(data['Etime'])
        xy_symbol=data['symbol']
        y_symbol= label_encoder.fit_transform(xy_symbol)
        # Extract relevant features from datetime objects
        years = [dt.year for dt in datetime_objects]
        months = [dt.month for dt in datetime_objects]
        days = [dt.day for dt in datetime_objects]
        hours = [dt.hour for dt in datetime_objects]
        minutes = [dt.minute for dt in datetime_objects]
        seconds = [dt.second for dt in datetime_objects]
        y_time= np.column_stack((years, months, days, hours, minutes, seconds))
        # Perform data scaling if required
        scaler = MinMaxScaler()
        print(X.shape)
        X =    scaler.fit_transform(X)

        # Split the data into train and test sets
        X_train, X_test, y_train_trend, y_test_trend, y_train_time, y_test_time, y_train_pips, y_test_pips, y_train_duration, y_test_duration, y_train_symbol, y_test_symbol = train_test_split(
            X, y_trend, y_time, y_pips, y_duration, y_symbol, test_size=0.2, random_state=42)

        # Step 2: Model Training
        # Instantiate the KNN classifier and regressor
        knn_trend = KNeighborsClassifier(n_neighbors=1)
        knn_time = KNeighborsClassifier(n_neighbors=1)
        knn_pips = KNeighborsRegressor(n_neighbors=1)
        knn_duration = KNeighborsRegressor(n_neighbors=1)
        knn_symbol = KNeighborsRegressor(n_neighbors=1)

        # Train the models
        knn_trend.fit(X_train, y_train_trend)
        knn_time.fit(X_train, y_train_time)
        knn_pips.fit(X_train, y_train_pips)
        knn_duration.fit(X_train, y_train_duration)
        knn_symbol.fit(X_train, y_train_symbol)
        predictions_trend = knn_trend.predict(X_test)
        predictions_time = knn_time.predict(X_test)
        predictions_pips = knn_pips.predict(X_test)
        predictions_duration = knn_duration.predict(X_test)
        predictions_symbol =knn_symbol.predict(X_test)
        predictions_symbols=label_encoder.inverse_transform(predictions_symbol.astype(int))
        mse_time = mean_squared_error(y_test_time, predictions_time)
        # Compute evaluation metrics
        accuracy_trend = accuracy_score(y_test_trend, predictions_trend)
        mean_accuracy = np.mean([accuracy_score(y_test_time[:, i], predictions_time[:, i]) for i in range(y_test_time.shape[1])])
        mse_pips = mean_squared_error(y_test_pips, predictions_pips)
        mae_pips = mean_absolute_error(y_test_pips, predictions_pips)

        print("MSE (Pips): {:.2f}".format(mse_pips))
        print("MAE (Pips): {:.2f}".format(mae_pips))
        mse_duration = mean_squared_error(y_test_duration, predictions_duration)

        print("Accuracy (Trend): {:.2f}%".format(accuracy_trend * 100))
        print("MSE (Time): {:.2f}".format(mse_time))

        print("Mean Accuracy (Time): {:.2f}%".format(mean_accuracy * 100))

        print("MSE (Duration): {:.2f}".format(mse_duration))
        return knn_trend, knn_time, knn_pips, knn_duration,knn_symbol
    else:
        print('df is empty')


def run_tasks():
    symbols = settings.symbols
    timeframes = settings.timeframes
    for symbol in symbols:
        for timeframe in timeframes:
            candles=asyncio.run(get_candles_m(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    continue
            else:
                df=extract(candles)
                settings.knn_trend, settings.knn_time, settings.knn_pips, settings.knn_duration,settings.knn_symbol=knn_train(df,symbol)
        time.sleep(120)
    time.sleep(60)


def knn(start_time,stop_time,symbol,timeframe):
    start_time=start_time
    from datetime import timedelta

    def faketime(start_time,stop_time):
        format_string = "%Y-%m-%d %H:%M:%S"
        # Create a list to store the generated timestamps
        data_points = []

        # Calculate the stop time by adding 30 minutes to the start time
        stop_time = stop_time#start_time + timedelta(minutes=30)

        # Generate timestamps at regular intervals
        current_datetime = start_time
        while current_datetime <= stop_time:
            data_points.append(current_datetime.strftime(format_string))
            current_datetime += timedelta(minutes=1)

        # Create a DataFrame with the data points
        df = pd.DataFrame(data_points, columns=["FTimestamp"])

        # Return the DataFrame
        return df["FTimestamp"]




    X_t=faketime(start_time,stop_time)

    #datetime_objects = [datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z") for timestamp in X_t]
    datetime_objects = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in X_t]
    # Extract relevant features from datetime objects
    years = [dt.year for dt in datetime_objects]
    months = [dt.month for dt in datetime_objects]
    days = [dt.day for dt in datetime_objects]
    hours = [dt.hour for dt in datetime_objects]
    minutes = [dt.minute for dt in datetime_objects]
    seconds = [dt.second for dt in datetime_objects]
    X_f= np.column_stack((years, months, days, hours, minutes, seconds))
    # Step 10: Model Evaluation
    # Perform predictions on the test set
    candles=asyncio.run(get_candles_m(timeframe,symbol))
    df=extract(candles)
    settings.knn_trend, settings.knn_time, settings.knn_pips, settings.knn_duration,settings.knn_symbol=knn_train(df,symbol)
    predictions_trend = settings.knn_trend.predict(X_f)
    predictions_time = settings.knn_time.predict(X_f)
    predictions_pips = settings.knn_pips.predict(X_f)
    predictions_duration = settings.knn_duration.predict(X_f)
    predictions_symbol =settings.knn_symbol.predict(X_f)
    predictions_symbols=label_encoder.inverse_transform(predictions_symbol.astype(int))
    return predictions_trend, predictions_time, predictions_pips,predictions_duration,predictions_symbols
    
ysymbols=[
    'EURUSDm',
    'USDJPYm',
    'GBPUSDm',
    'XAUUSDm',
    'XAGUSDm',
    'AUDUSDm',
    'NZDUSDm'
]
def index(request):
    List=[]
    for symbol in ysymbols:
        headline,link,relatedTickers,thumbnail_url,sentiment,sentiment_label=news_analysis(symbol)
        if sentiment==0:
            pass
        else:
            List.append({
                'Headline': f'{headline}',
                'link': f'{link}',
                'relatedTickers': f'{relatedTickers}',
                'thumbnail_url': f'{thumbnail_url}',
                'Sentiment_Score': f'{sentiment} %',
                'Sentiment_Label': f'{sentiment_label}',
                'Symbol':f'{symbol}',
                })
    results=[]
    timeframes=['1h',]
    word_list = ['no', 'hold']
    for symbol in ysymbols:
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    answer='Error accessing the forex market'
            else:
                
                dt=cal_double_tops(candles)
                abcd=cal_identify_abcd_pattern(candles)
                bp=cal_identify_bat_pattern(candles)
                bup=cal_identify_butterfly_pattern(candles)
                cp=cal_identify_crab_pattern(candles)
                cyp=cal_identify_cypher_pattern(candles)
                db=cal_identify_double_bottom(candles)
                gp=cal_identify_gartley_pattern(candles)
                sp=cal_identify_shark_pattern(candles)
                tdp=cal_identify_three_drive_pattern(candles)
                hns=cal_identify_head_shoulders_pattern(candles)
                chaup=cal_channel_up(candles)
                chadn=cal_channel_down(candles)
                jcp=identify_candle_pattern(candles)
                jcp=identify_candle_pattern(candles)
                asc=analyze_ascending_triangle(candles)
                dsc=analyze_descending_triangle(candles)
                flag=flag_pattern(candles)
                tripplet=triple_top(candles)
                trippleb=triple_bottom(candles)
                rect=rectangle(candles)
                penn=pennant(candles)
                symmetrical=symmetrical_triangle(candles)
                wed=wedge(candles)
                diam=diamond(candles)
                ihns=inverse_head_and_shoulders(candles)
                cnh=cup_and_handle(candles)
                if flag is not None:
                    if any(word.lower() in flag.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'flag Pattern', 'Decision': flag,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if tripplet is not None:
                    if any(word.lower() in tripplet.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Tripple Top', 'Decision': tripplet,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if trippleb is not None:
                    if any(word.lower() in trippleb.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Tripple  Bottom', 'Decision': trippleb,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if rect is not None:
                    if any(word.lower() in rect.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Rectangle', 'Decision': rect,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if penn is not None:
                    if any(word.lower() in penn.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Pennant', 'Decision': penn,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if symmetrical is not None:
                    if any(word.lower() in symmtrical.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Symmetrical Triangle', 'Decision': symmetrical,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if wed is not None:
                    if any(word.lower() in wed.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Wedge', 'Decision': wed,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if diam is not None:
                    if any(word.lower() in diam.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Diamond', 'Decision': diam,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if ihns is not None:
                    if any(word.lower() in ihns.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Inverse Head and Shoulders', 'Decision': ihns,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if cnh is not None:
                    if any(word.lower() in cnh.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Cup and Handle', 'Decision': cnh,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if asc is not None:
                    if any(word.lower() in asc.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Ascending Triangle', 'Decision': asc,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if dsc is not None:
                    if any(word.lower() in dsc.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Descending Triangle', 'Decision': dsc,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if jcp is not None:
                    if any(word.lower() in jcp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Japanese Candle Pattern', 'Decision': jcp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if chaup is not None:
                    if any(word.lower() in chaup.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Channel Up', 'Decision': chaup,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if chadn is not None:
                    if any(word.lower() in chadn.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Channel Down', 'Decision': chadn,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if hns is not None:
                    if any(word.lower() in hns.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Head and Shoulders', 'Decision': hns,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if dt is not None:
                    if any(word.lower() in dt.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Double Tops', 'Decision': dt,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if abcd is not None:
                    if any(word.lower() in abcd.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'ABCD Pattern', 'Decision': abcd,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if bp is not None:
                    if any(word.lower() in bp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Bat Pattern', 'Decision': bp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if bup is not None:
                    if any(word.lower() in bup.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Butterfly Pattern', 'Decision': bup,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if cp is not None:
                    if any(word.lower() in cp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Crab Pattern', 'Decision': cp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if cyp is not None:
                    if any(word.lower() in cyp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Cypher Pattern', 'Decision': cyp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if db is not None:
                    if any(word.lower() in db.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Double Bottoms', 'Decision': db,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if gp is not None:
                    if any(word.lower() in gp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Gartley Pattern', 'Decision': gp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if sp is not None:
                    if any(word.lower() in sp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Shark Pattern', 'Decision': sp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if tdp is not None:
                    if any(word.lower() in tdp.lower() for word in word_list):
                        pass
                    else:
                        results.append({'Pattern': 'Three Drive Pattern', 'Decision': tdp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})

                
                answer="Market Analysied Successfully"#f'{concatenated_string}  </br> For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        table_data = results
    context={'List':List,'table_data':table_data,'answer':answer}
    return render(request,'home.html',context)



def predict(request):
    time= ['30m','1h','2h','3h','4h','5h']
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        length = request.POST.get('length')
        stop_tiM=str(length)
        print(type(timeframe))
        #candles = pd.read_csv('2000.csv')
        print('Start')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                output_string='Error accessing the forex market'
        else:
            # Access the last candle
            last_candle = candles[-1]
            # Extract the timestamp of the last candle
            start_time = last_candle['time']

            
            if stop_tiM == '30m':
                stop_time=start_time + timedelta(minutes=30)
            elif stop_tiM=='1h':
                stop_time=start_time + timedelta(hours=1)
            elif stop_tiM=='2h':
                stop_time=start_time + timedelta(hours=2)
            elif stop_tiM=='3h':
                stop_time=start_time + timedelta(hours=3)
            elif stop_tiM=='4h':
                stop_time=start_time + timedelta(hours=4)
            elif stop_tiM=='5h':
                stop_time=start_time + timedelta(hours=5)
            
            #df=extract(candles)
            print('started')
            predictions_trend, predictions_time, predictions_pips,predictions_duration,predictions_symbols=knn(start_time,stop_time,symbol,timeframe)


            output_string = ""
            pips=None
            format_string = "%Y-%m-%d %H:%M:%S"
            for prediction in zip(predictions_symbols, predictions_trend, predictions_time, predictions_pips, predictions_duration):
                sym_symbol, sym_trend, sym_time, sym_pips, sym_duration = prediction
                if sym_symbol == symbol:
                    if sym_trend=='Downtrend' and pips!=sym_pips:
                        stop_time=start_time + timedelta(hours=5)
                        output_string += f"Trend: {sym_trend}<br>Time: {sym_time}<br>Pips: {sym_pips}<br>Duration: {sym_duration}<br><br><br><br>"
                        pips=sym_pips
                    if sym_trend=='Uptrend' and pips!=sym_pips:
                        '''
                        if timeframe=='1m':
                            start_time=sym_time - timedelta(minutes=70)
                        if timeframe=='5m':
                            sym_times=str(sym_time)
                            start_time=sym_times.strftime(format_string) - timedelta(minutes=350)
                        if timeframe=='15m':
                            start_time=sym_time - timedelta(hours=17)
                        if timeframe=='30m':
                            start_time=sym_time - timedelta(hours=35)
                        if timeframe=='1h':
                            start_time=sym_time - timedelta(hours=70)
                        if timeframe=='4h':
                            start_time=sym_time - timedelta(hours=280)
                        if timeframe=='1d':
                            start_time=sym_time - timedelta(hours=1680)
                        if timeframe=='1w':
                            start_time=sym_time - timedelta(hours=11760)
                        '''
                        output_string += f"Trend: {sym_trend}<br>Ending Time: {sym_time}<br>Duration(Minimum Expected): {sym_duration} candles, So <u>Start time</u> is {sym_duration} candles before endtime<br><br><br><br>"
                        pips=sym_pips

        pairs = symbols
        timeframees = time

        answer=''
        indicator='TradeGPT-X2, Your Trend Predictor'
        return render(request, 'pd.html', {'indicator': indicator, 'AI': 'head', 'pairs': pairs, 'timeframes': timeframes, 'answer': answer, 'output_string': output_string})

    pairs = symbols
    time = time
    timeframees=timeframes
    answer='No prediction Run yet'
    indicator='TradeGPT, Your AI Trend Predictor'
    return render(request,'index.html',{'time':time,'indicator':indicator,'AI':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



#
#
#
#INDICATOR



def calculate_rsi(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    rsi=ta.rsi(df['close'], length=14)
    if rsi[-1] < 30:
        return f"RSI value is {rsi[-1]}, Buy"
    if rsi[-1] > 70:
        return f"RSI value is {rsi[-1]}, Sell"
    else:
        return f"RSI value is {rsi[-1]}, neautral"


def calculate_adx(candles, period=14):
    high = np.array([candle['high'] for candle in candles])
    low = np.array([candle['low'] for candle in candles])
    close = np.array([candle['close'] for candle in candles])
    # Calculate True Range (TR)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])

    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[(up_move <= 0) | (up_move <= down_move)] = 0.0
    down_move[(down_move <= 0) | (down_move <= up_move)] = 0.0

    # Calculate True Range (TR) EMA
    tr_ema = np.convolve(tr, np.ones(period), mode='valid') / period

    # Calculate Directional Movement Index (DMI)
    plus_di = (np.convolve(up_move, np.ones(period), mode='valid') / tr_ema) * 100
    minus_di = (np.convolve(down_move, np.ones(period), mode='valid') / tr_ema) * 100

    # Calculate DX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    # Pad DX array to match the length of input arrays
    dx = np.concatenate((np.full(period-1, np.nan), dx))

    # Calculate ADX
    adx = np.convolve(dx, np.ones(period), mode='valid') / period

    # Pad ADX array to match the length of input arrays
    adx = np.concatenate((np.full(period-1, np.nan), adx))
    if adx[-1]>25:
        return f"ADX value is {adx[-1]}, Strong Trend Now check MA,SMA,MACD OR EMA TO Check weather to Buy or Sell"
    else:
        return f"ADX value is {adx[-1]}, No or weak trend(Ignore)"
    


def calculate_volatility(candles):
    # Extracting high and low prices from the candlestick data
    highs = np.array([candle['high'] for candle in candles])
    lows = np.array([candle['low'] for candle in candles])
    
    # Calculating the range (difference) between high and low prices
    ranges = highs - lows
        
    # Calculating volatility as the standard deviation of the price ranges
    volatility = np.std(ranges)
    
    return volatility


def calculate_ema(data, n):
    # Calculate the smoothing factor
    alpha = 2 / (n + 1)

    # Calculate the initial EMA as the first data point
    ema = np.array([data[0]])

    # Calculate the subsequent EMAs
    for i in range(1, len(data)):
        ema = np.append(ema, alpha * data[i] + (1 - alpha) * ema[-1])
    
    if np.any(data == 200):
        # Handle the case when data is equal to 200
        pass
    if np.any(data == 50):
        # Handle the case when data is equal to 50
        pass
    
    return ema

def calculate_macd(candles, n_fast, n_slow, n_signal):
    # Extract closing prices from the candlestick data\
    candlestick_data=candles
    closing_prices= np.array([candle['close'] for candle in candlestick_data])

    # Calculate the short-term exponential moving average (EMA)
    ema_fast = calculate_ema(closing_prices, n_fast)

    # Calculate the long-term exponential moving average (EMA)
    ema_slow = calculate_ema(closing_prices, n_slow)

    # Calculate the MACD line
    macd_line = ema_fast - ema_slow

    # Calculate the signal line (n-period EMA of MACD line)
    signal_line = calculate_ema(macd_line, n_signal)

    # Calculate the MACD histogram
    macd_histogram = macd_line - signal_line

    if macd_line[-1]>0 and macd_line[-1]>signal_line[-1]:
        return "Buy based on MACD"
    elif macd_line[-1]<0 and macd_line[-1]<signal_line[-1]:
        return "Sell based on MACD"
    else:
        return "No signal based on MACD"
    
def calculate_smi(candles, period=14, smoothing_period=3, double_smoothing_period=3):
    highest_highs = np.array([candle['high'] for candle in candles])
    lowest_lows = np.array([candle['low'] for candle in candles])
    close_prices = np.array([candle['close'] for candle in candles])


    hh_diff = np.max(highest_highs[:period]) - np.min(lowest_lows[:period])

    smi = np.zeros(len(close_prices))

    for i in range(period - 1, len(close_prices)):
        sum1 = np.sum((close_prices[j] - lowest_lows[j]) / hh_diff for j in range(i - period + 1, i + 1))
        sum2 = np.sum((highest_highs[j] - lowest_lows[j]) / hh_diff for j in range(i - period + 1, i + 1))

        smi[i] = 100 * (sum1 / sum2)

    smi_smoothed = np.zeros(len(smi) - smoothing_period)
    for i in range(len(smi_smoothed)):
        smi_smoothed[i] = np.sum(smi[j] for j in range(i, i + smoothing_period)) / smoothing_period

    smi_double_smoothed = np.zeros(len(smi_smoothed) - double_smoothing_period)
    for i in range(len(smi_double_smoothed)):
        smi_double_smoothed[i] = np.sum(smi_smoothed[j] for j in range(i, i + double_smoothing_period)) / double_smoothing_period

    smi= np.pad(smi_double_smoothed, (0, len(close_prices) - len(smi_double_smoothed)))
    if smi[-1]>0:
        return "Buy based on SMI"
    elif smi[-1]<0:
        return "Sell based on SMI"
    else:
        return "No signal based on SMI"

def calculate_ichimoku_cloud(high_prices, low_prices, conversion_period=9, base_period=26, leading_span_b_period=52, displacement=26):
    # Tenkan-sen (Conversion Line)
    conversion_line = (np.max(high_prices[-conversion_period:]) + np.min(low_prices[-conversion_period:])) / 2

    # Kijun-sen (Base Line)
    base_line = (np.max(high_prices[-base_period:]) + np.min(low_prices[-base_period:])) / 2

    # Senkou Span A (Leading Span A)
    leading_span_a = (conversion_line + base_line) / 2

    # Senkou Span B (Leading Span B)
    leading_span_b = (np.max(high_prices[-leading_span_b_period:]) + np.min(low_prices[-leading_span_b_period:])) / 2

    # Shift the Leading Span A and Leading Span B values forward
    leading_span_a = np.roll(leading_span_a, displacement)
    leading_span_b = np.roll(leading_span_b, displacement)

    return conversion_line, base_line, leading_span_a, leading_span_b

def calculate_ichimoku(candles):
    high_prices = np.array([candle['high'] for candle in candles])
    low_prices = np.array([candle['low'] for candle in candles])
    close_prices = np.array([candle['close'] for candle in candles])
    conversion_line, base_line, leading_span_a, leading_span_b = calculate_ichimoku_cloud(high_prices, low_prices)

    current_close = close_prices[-1]
    previous_close = close_prices[-2]

    # Check for trend direction
    if current_close > leading_span_a and current_close > leading_span_b:
        trend_direction = "Strong Uptrend"
    elif current_close < leading_span_a and current_close < leading_span_b:
        trend_direction = "Strong Downtrend"
    elif current_close > leading_span_a or current_close > leading_span_b:
        trend_direction = "Weak Uptrend"
    else:
        trend_direction = "Weak Downtrend"

    # Trading signals based on trend direction
    if trend_direction == "Strong Uptrend":
        if previous_close < conversion_line and previous_close < base_line:
            return "Buy Signal,Strong Uptrend"

    if trend_direction == "Strong Downtrend":
        if previous_close > conversion_line and previous_close > base_line:
            return "Sell Signal,Strong Downtrend"

    return "No Signal"
def calculate_bb(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    df['bollinger_middle'] = ta.sma(df['close'], length=bollinger_period)
    df['bollinger_std'] = ta.stdev(df['close'], length=bollinger_period)
    df['bollinger_upper'] = df['bollinger_middle'] + bollinger_std * df['bollinger_std']
    df['bollinger_lower'] = df['bollinger_middle'] - bollinger_std * df['bollinger_std']
    
    upper_band = df['bollinger_upper'].values
    lower_band = df['bollinger_lower'].values
    close_prices = df['close'].values
    if close_prices[-1] > upper_band[-1]:
        return f"Buy, Safe place a buy order"
    elif close_prices[-1] < lower_band[-1]:
        return f"Sell, Safe place a sell order"
    else:  
        return f"Hold, Stay away from that market"


def rsi(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                rsi='Error accessing the forex market'
        else:
            rsi=calculate_rsi(candles) 
        
        # Fetch the available  pairs from the database
        pairs = symbols
        timeframees = timeframes
        indicator='RSI (Relative Strength Index)'
        answer=f'{rsi} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        return render(request,'index.html',{'indicator':indicator,'rsi':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='RSI (Relative Strength Index)'
    return render(request,'index.html',{'indicator':indicator,'rsi':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


def macd(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                macd='Error accessing the forex market'
        else:
            macd=calculate_macd(candles,n_fast, n_slow, n_signal)
        
        # Fetch  the available pairs from the database
        pairs = symbols
        timeframees = timeframes
        answer=f'{macd} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        indicator='MACD (Moving Average Convergence Divergence)'
        return render(request,'index.html',{'indicator':indicator,'macd':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='MACD (Moving Average Convergence Divergence)'
    return render(request,'index.html',{'indicator':indicator,'macd':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



def smi(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                smi='Error accessing the forex market'
        else:
            smi=calculate_smi(candles)
        # Fetch the available pairs  from the database
        pairs = symbols
        timeframees = timeframes
        answer=f'{smi} for Symbol :  {symbol} and Timeframe:  {timeframe}'

        indicator='Stochastic Momentum Index'
        return render(request,'index.html',{'indicator':indicator,'smi':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Stochastic Momentum Index'
    return render(request,'index.html',{'indicator':indicator,'smi':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


def bb(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                bb='Error accessing the forex market'
        else:
            bb=calculate_bb(candles)
        # Fetch the available pairs from  the database
        pairs = symbols
        timeframees = timeframes
        indicator='Bollinger Bands'
        answer=f'{bb} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        return render(request,'index.html',{'indicator':indicator,'bb':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Bollinger Bands'
    return render(request,'index.html',{'indicator':indicator,'bb':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})




def adx(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                adx='Error accessing the forex market'
        else:
            adx=calculate_adx(candles)
        # Fetch the available pairs from the  database
        pairs = symbols
        timeframees = timeframes
        answer=f'{adx} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        indicator='Average Directional Index'
        return render(request,'index.html',{'indicator':indicator,'adx':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Average Directional Index'
    return render(request,'index.html',{'indicator':indicator,'adx':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



def ichimuko(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                ichimuko='Error accessing the forex market'
        else:
            ichimuko=calculate_ichimoku(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{ichimuko} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        indicator='Ichimuko Cloud'
        return render(request,'index.html',{'indicator':indicator,'ichimuko':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Ichimuko Cloud'
    return render(request,'index.html',{'indicator':indicator,'ichimuko':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})






# Momentum
def cal_momentum(candles, period=5):
    
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    data=df['close']
    momentum_value=data[-1] - data[-period]
    if momentum_value > 0:
        return "Buy"
    else:
        return "Sell"



def momentum(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                mmm='Error accessing the forex market'
        else:
            mmm=cal_momentum(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{mmm} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        indicator='Momentum'
        return render(request,'index.html',{'indicator':indicator,'mmm':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Momentum'
    return render(request,'index.html',{'indicator':indicator,'mmm':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



# Parabolic SAR
def parabolic_sar(candles, acceleration_factor_step=0.02, acceleration_factor_max=0.2):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']

    af = acceleration_factor_step
    ep = high[0]
    sar = low[0]
    trend = 1  # 1 for bullish, -1 for bearish

    for i in range(2, len(high)):
        if trend == 1:
            if low[i] < sar:
                trend = -1
                sar = ep
                af = acceleration_factor_step
        else:
            if high[i] > sar:
                trend = 1
                sar = ep
                af = acceleration_factor_step
        
        if trend == 1:
            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration_factor_step, acceleration_factor_max)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration_factor_step, acceleration_factor_max)
        
        sar += af * (ep - sar)
        
    # Parabolic SAR
    if low[-1] > sar:
        return "Buy based on Parabolic SAR"
    else:
        return "Sell based on Parabolic SAR"


def pbr(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                pbr='Error accessing the forex market'
        else:
            pbr=parabolic_sar(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{pbr} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        indicator='Parabolic SAR'
        return render(request,'index.html',{'indicator':indicator,'pbr':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Parabolic SAR'
    return render(request,'index.html',{'indicator':indicator,'pbr':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


# Simple Moving Average (SMA)
def simple_moving_average(candles, period=5):
    
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    data=df['close']
    sma=np.mean(data[-period:])
    # Simple Moving Average (SMA)
    if data[-1] > sma:
        return "Buy based on Simple Moving Average (SMA)"
    else:
        return "Sell based on Simple Moving Average (SMA)"



def sma(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                sma='Error accessing the forex market'
        else:
            sma=simple_moving_average(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{sma} for Symbol :  {symbol} and Timeframe:  {timeframe}'
        indicator='Simple Moving Average (SMA)'
        return render(request,'index.html',{'indicator':indicator,'sma':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Simple Moving Average (SMA)'
    return render(request,'index.html',{'indicator':indicator,'sma':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



# Exponential Moving Average (EMA)
def exponential_moving_average(candles, period=5):
    
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    data=df['close']
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(data, weights, mode='full')[:len(data)]
    ema=ema[-1]
    # Exponential Moving Average (EMA)
    if data[-1] > ema:
        return "Buy based on  Exponential Moving Average (EMA)"
    else:
        return "Sell based on  Exponential Moving Average (EMA)"


def ema(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                ema='Error accessing the forex market'
        else:
            ema=exponential_moving_average(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{ema} for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Exponential Moving Average (EMA)'
        return render(request,'index.html',{'indicator':indicator,'ema':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Exponential Moving Average (EMA)'
    return render(request,'index.html',{'indicator':indicator,'ema':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



def donchian_channels(candles, period=5):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values

    dc_high = np.max(high[-period:])
    dc_low = np.min(low[-period:])
    # Trading decisions
    if close[-1] > dc_high:
        return "Sell based on Donchian Channels"
    elif close[-1] < dc_low:
        return "Buy based on Donchian Channels"
    else:
        return "Neither, Hold"


def dcc(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                dcc='Error accessing the forex market'
        else:
            dcc=donchian_channels(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{dcc}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Donchian Channels'
        return render(request,'index.html',{'indicator':indicator,'dcc':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Donchian Channels'
    return render(request,'index.html',{'indicator':indicator,'dcc':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


# Moving Average (MA)
def moving_average(candles, period=5):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    data=df['close']
    ma= np.mean(data[-period:])
    # Moving Average (MA)
    if data[-1] > ma:
        return "Buy Opportunity based on Moving Average"
    else:
        return "Sell Opportunity based on Moving Average"


def moa(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                moa='Error accessing the forex market'
        else:
            moa=moving_average(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{moa}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Moving Average (MA)'
        return render(request,'index.html',{'indicator':indicator,'moa':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Moving Average (MA)'
    return render(request,'index.html',{'indicator':indicator,'moa':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



def chaikins_volatility(candles, period=5):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']
    clv = ((close - low) - (high - close)) / (high - low)
    cv = np.mean(np.abs(clv[-period:]))
    if cv > 0.05:
        return f" Good strong trend,Now check MA,SMA,MACD OR EMA TO CHECK WEATHER TO Buy or sell based on Chaikin's Volatility:  {cv}"



def chv(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                chv='Error accessing the forex market'
        else:
            chv=chaikins_volatility(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{chv}   for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Chaikins Volatility'
        return render(request,'index.html',{'indicator':indicator,'chv':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Chaikins Volatility'
    return render(request,'index.html',{'indicator':indicator,'chv':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


def standard_deviation(candles, period=5):
    
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    close=df['close'].values
    sd= np.std(close[-period:])
    if close[-1] > (close[-2] + 2 * sd):
        return "Sell based on Standard Deviation"
    elif close[-1] < (close[-2] - 2 * sd):
        return "Buy based on Standard Deviation"
    else:
        return 'Neither,Hold'


def std(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                std='Error accessing the forex market'
        else:
            std=standard_deviation(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{std}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Standard Deviation'
        return render(request,'index.html',{'indicator':indicator,'std':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Standard Deviation'
    return render(request,'index.html',{'indicator':indicator,'std':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})




def average_true_range(candles, period=5):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr = np.mean(tr[-period:])
    if close[-1] > (close[-2] + atr):
        return "Sell based on Average True Range"
    elif close[-1] < (close[-2] - atr):
        return "Buy based on Average True Range"
    else:
        return 'Neither,Hold'


def atr(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                atr='Error accessing the forex market'
        else:
            atr=average_true_range(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{atr}   for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Average True Range (ATR)'
        return render(request,'index.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Average True Range (ATR)'
    return render(request,'index.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


def calculate_adl(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values
    volume=df['tickVolume'].values
    adl = ((close - low) - (high - close)) / (high - low)
    adl = adl * volume
    adl = np.cumsum(adl)
    
    if adl[-1] > adl[-2]:
        return "Buy according to ADL"
    elif adl[-1] < adl[-2]:
        return "Sell according to ADL"

    else:
        return 'Neither,Hold'


def adl(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                adl='Error accessing the forex market'
        else:
            adl=calculate_adl(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{adl}   for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Accumulation/Distribution Line (ADL)'
        return render(request,'index.html',{'indicator':indicator,'adl':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Accumulation/Distribution Line (ADL)'
    return render(request,'index.html',{'indicator':indicator,'adl':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



def calculate_obv(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    close=df['close'].values
    volume=df['tickVolume'].values

    obv = np.zeros_like(close)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    if obv[-1] > obv[-2]:
        return "Buy according to OBV"
    elif obv[-1] < obv[-2]:
        return "Sell according to OBV"
    else:
        return 'Neither,Hold'


def obv(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                obv='Error accessing the forex market'
        else:
            obv=calculate_obv(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{obv}   for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='On-Balance Volume (OBV)'
        return render(request,'index.html',{'indicator':indicator,'obv':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='On-Balance Volume (OBV)'
    return render(request,'index.html',{'indicator':indicator,'obv':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


def calculate_cmf(candles, period=20):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values
    volume=df['tickVolume'].values

    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    adl = np.cumsum(money_flow_volume)

    cmf = np.zeros_like(close)
    cmf[period-1:] = np.sum(adl[1:period] - adl[:period-1], axis=0) / np.sum(volume[:period], axis=0)

    if cmf[-1] > 0.3:
        return "Buy according to CMF"
    elif cmf[-1] < -0.3:
        return "Sell according to CMF"
    else:
        return 'Neither,Hold'


def cmf(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                cmf='Error accessing the forex market'
        else:
            cmf=calculate_cmf(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{cmf}   for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Chaikin Money Flow (CMF)'
        return render(request,'index.html',{'indicator':indicator,'cmf':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Chaikin Money Flow (CMF)'
    return render(request,'index.html',{'indicator':indicator,'cmf':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})





def stochastic_oscillator(candles, period=14):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values
    highest_high = np.max(high[-period:])
    lowest_low = np.min(low[-period:])
    current_close = close[-1]
    
    K = (current_close - lowest_low) / (highest_high - lowest_low) * 100
    D = np.mean(K)

    if K > D:
        return "Buy according to Stochastic Oscillator"
    else:
        return "Sell according to Stochastic Oscillator"
    

def sto(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                so='Error accessing the forex market'
        else:
            so=stochastic_oscillator(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{so}   for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Stochastic Oscillator'
        return render(request,'index.html',{'indicator':indicator,'sto':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Stochastic Oscillator'
    return render(request,'index.html',{'indicator':indicator,'so':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})




def cci_c(candles, period=20):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values
    typical_price = (high + low + close) / 3
    typical_price_sma = np.mean(typical_price[-period:])
    mean_deviation = np.mean(np.abs(typical_price[-period:] - typical_price_sma))
    
    constant = 0.015
    cci_value = (typical_price[-1] - typical_price_sma) / (constant * mean_deviation)
    
    if cci_value > 100:
        return "Buy according to CCI"
    elif cci_value < -100:
        return "Sell according to CCI"
    else:
        return 'Neither,Hold'


def cci(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                cci='Error accessing the forex market'
        else:
            cci=cci_c(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{cci}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Commodity Channel Index (CCI)'
        return render(request,'index.html',{'indicator':indicator,'cci':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Commodity Channel Index (CCI)'
    return render(request,'index.html',{'indicator':indicator,'cci':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})




def williams_percent_r(candles, period=14):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high'].values
    low=df['low'].values
    close=df['close'].values
    highest_high = np.max(high[-period:])
    lowest_low = np.min(low[-period:])
    
    R = (highest_high - close[-1]) / (highest_high - lowest_low) * -100
    
    if R > -20:
        return "Buy"
    elif R < -80:
        return "Sell"
    else:
        return "Hold"


def wpr(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                wpr='Error accessing the forex market'
        else:
            wpr=williams_percent_r(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{wpr}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Williams %R'
        return render(request,'index.html',{'indicator':indicator,'wpr':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Williams %R'
    return render(request,'index.html',{'indicator':indicator,'wpr':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    


def rate_of_change(candles, period=14):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    close=df['close'].values
    roc = (close[-1] - close[-period]) / close[-period] * 100
    
    if roc > 0:
        return "Buy according to rate of change"
    else:
        return "Sell according to rate of change"
    

def roc(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                roc='Error accessing the forex market'
        else:
            roc=rate_of_change(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{roc}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Rate of Change (ROC)'
        return render(request,'index.html',{'indicator':indicator,'roc':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Rate of Change (ROC)'
    return render(request,'index.html',{'indicator':indicator,'roc':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})






# Trendlines
def calculate_trendlines(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']
    trendline_slope = 0.5  # Example slope value
    trendline_intercept = 100  # Example intercept value
    trendline = trendline_slope * np.arange(len(close)) + trendline_intercept
    if close[-1] > trendline[-1]:
        return "Buy based on Trendlines"
    else:
        return "Sell based on Trendlines"




def trendlines(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                trendlines='Error accessing the forex market'
        else:
            trendlines=calculate_trendlines(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{trendlines}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Trendlines'
        return render(request,'index.html',{'indicator':indicator,'trendlines':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Trendlines'
    return render(request,'index.html',{'indicator':indicator,'trendlines':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

# Gann Fan
def calculate_gannfan(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']
    previous_high = high[-2]
    previous_low = low[-2]
    fan_angle = 1  # Example angle value
    fan_line = previous_low + fan_angle * (previous_high - previous_low)
    if close[-1] > fan_line:
        return "Buy based on Gann Fan"
    else:
        return "Sell based on Gann Fan"





def gan_fan(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                gan_fan='Error accessing the forex market'
        else:
            gan_fan=calculate_gannfan(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{gan_fan}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Gann Fan'
        return render(request,'index.html',{'indicator':indicator,'gan_fan':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Gann Fan'
    return render(request,'index.html',{'indicator':indicator,'gan_fan':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})

    
# Support and Resistance Lines
def cal_support_resistance(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']
    support_line = np.min(low)
    resistance_line = np.max(high)
    if close[-1] > resistance_line:
        return "Buy based on Support and Resistance Lines"
    elif close[-1] < support_line:
        return "Sell based on Support and Resistance Lines"
    else:
        return "Hold based on Support and Resistance Lines"







def suppt_resistance(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                pivot_points='Error accessing the forex market'
        else:
            pivot_points=cal_support_resistance(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{pivot_points}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Support and Resistance Lines'
        return render(request,'index.html',{'indicator':indicator,'suppt_resistance':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Support and Resistance Lines'
    return render(request,'index.html',{'indicator':indicator,'suppt_resistance':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})



# Pivot Points
def calculate_pivot_point(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']
    pivot_point = (high[-1] + low[-1] + close[-1]) / 3
    if close[-1] > pivot_point:
        return "Buy based on Pivot Points"
    else:
        return "Sell based on Pivot Points"





def pivot_points(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                pivot_points='Error accessing the forex market'
        else:
            pivot_points=calculate_pivot_point(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{pivot_points}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Pivot Points'
        return render(request,'index.html',{'indicator':indicator,'pivot_points':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Pivot Points'
    return render(request,'index.html',{'indicator':indicator,'pivot_points':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})





# Fibonacci Retracement
def calculate_fibonachi(candles):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    high=df['high']
    low=df['low']
    close=df['close']
    previous_high = high[-2]
    previous_low = low[-2]
    fibonacci_38_2 = previous_high - 0.382 * (previous_high - previous_low)
    if close[-1] > fibonacci_38_2:
        return "Buy based on Fibonacci Retracement"
    else:
        return "Sell based on Fibonacci Retracement"


def fibonachi(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                fibonachi='Error accessing the forex market'
        else:
            fibonachi=calculate_fibonachi(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{fibonachi}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Fibonacci Retracement'
        return render(request,'index.html',{'indicator':indicator,'fibonachi':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Fibonacci Retracement'
    return render(request,'index.html',{'indicator':indicator,'fibonachi':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


import pandas as pd
import numpy as np

def candlepatterndt(candles):
    # Candlestick pattern detection functions
    def check_doji(open_prices, close, high, low):
        body_size = np.abs(close - open_prices)
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        
        is_doji = body_size <= 0.01 * np.mean(body_size)
        
        return is_doji

    def check_hammer(open_prices, close, high, low):
        body_size = np.abs(close - open_prices)
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        
        is_hammer = (body_size <= 0.01 * np.mean(body_size)) & \
                    (lower_shadow >= 2 * body_size) & \
                    (upper_shadow <= 0.01 * np.mean(body_size))
        
        return is_hammer

    def check_shooting_star(open_prices, close, high, low):
        body_size = np.abs(close - open_prices)
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        
        is_shooting_star = (body_size <= 0.01 * np.mean(body_size)) & \
                        (upper_shadow >= 2 * body_size) & \
                        (lower_shadow <= 0.01 * np.mean(body_size))
        
        return is_shooting_star

    def check_engulfing_pattern(open_prices, close, high, low):
        prev_open_prices = open_prices.shift(1)
        previous_close = close.shift(1)
        
        is_bullish_engulfing = (previous_close < prev_open_prices) & \
                            (close > open_prices) & \
                            (open_prices < previous_close) & \
                            (close > prev_open_prices)
        
        is_bearish_engulfing = (previous_close > prev_open_prices) & \
                            (close < open_prices) & \
                            (open_prices > previous_close) & \
                            (close < prev_open_prices)
        
        return is_bullish_engulfing, is_bearish_engulfing

    def check_harami_pattern(open_prices, close, high, low):
        prev_open_prices = open_prices.shift(1)
        previous_close = close.shift(1)
        
        is_bullish_harami = (previous_close > prev_open_prices) & \
                            (close < open_prices) & \
                            (open_prices > previous_close) & \
                            (close < prev_open_prices)
        
        is_bearish_harami = (previous_close < prev_open_prices) & \
                            (close > open_prices) & \
                            (open_prices < previous_close) & \
                            (close > prev_open_prices)
        
        return is_bullish_harami, is_bearish_harami

    def check_morning_star(open_prices, close, high, low):
        prev_open_prices = open_prices.shift(1)
        previous_close = close.shift(1)
        prev_low = low.shift(1)
        
        is_morning_star = (previous_close < prev_open_prices) & \
                        (close > open_prices) & \
                        (open_prices < previous_close) & \
                        (close > prev_open_prices) & \
                        (previous_close > prev_open_prices) & \
                        (open_prices > prev_low)
        
        return is_morning_star

    def check_evening_star(open_prices, close, high, low):
        prev_open_prices = open_prices.shift(1)
        previous_close = close.shift(1)
        prev_high = high.shift(1)
        
        is_evening_star = (previous_close > prev_open_prices) & \
                        (close < open_prices) & \
                        (open_prices > previous_close) & \
                        (close < prev_open_prices) & \
                        (previous_close < prev_open_prices) & \
                        (open_prices < prev_high)
        
        return is_evening_star


    # Trend analysis function (simple moving average)
    def check_trend(prices, window=20):
        moving_average = prices.rolling(window).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        if current_price > moving_average:
            return "Uptrend"
        elif current_price < moving_average:
            return "Downtrend"
        else:
            return "Sideways"

    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    
    # Extract OHLC data
    open_prices = pd.Series(df['open'].astype(float))
    high_prices = pd.Series(df['high'].astype(float))
    low_prices = pd.Series(df['low'].astype(float))
    close_prices = pd.Series(df['close'].astype(float))

    # Check each candlestick pattern
    is_doji = check_doji(open_prices, close_prices, high_prices, low_prices)
    is_hammer = check_hammer(open_prices, close_prices, high_prices, low_prices)
    is_shooting_star = check_shooting_star(open_prices, close_prices, high_prices, low_prices)
    bullish_engulfing, bearish_engulfing = check_engulfing_pattern(open_prices, close_prices, high_prices, low_prices)
    bullish_harami, bearish_harami = check_harami_pattern(open_prices, close_prices, high_prices, low_prices)
    is_morning_star = check_morning_star(open_prices, close_prices, high_prices, low_prices)
    is_evening_star = check_evening_star(open_prices, close_prices, high_prices, low_prices)

    # Perform trend analysis
    prices = close_prices  # Using close prices for trend analysis
    trend = check_trend(prices)
    List = []
    
    # Make trading decisions based on patterns and trend
    if any(is_doji) and trend == "Uptrend":
        ds = "Doji pattern detected in an uptrend. Consider selling."
        List.append(ds)
    if any(is_doji) and trend == "Downtrend":
        db = "Doji pattern detected in a downtrend. Consider buying."
        List.append(db)

    if any(is_hammer) and trend == "Uptrend":
        hs = "Hammer pattern detected in an uptrend. Consider selling."
        List.append(hs)

    if any(is_hammer) and trend == "Downtrend":
        hb = "Hammer pattern detected in a downtrend. Consider buying."
        List.append(hb)

    if any(is_shooting_star) and trend == "Uptrend":
        sss = "Shooting Star pattern detected in an uptrend. Consider selling."
        List.append(sss)

    if any(is_shooting_star) and trend == "Downtrend":
        ssb = "Shooting Star pattern detected in a downtrend. Consider buying."
        List.append(ssb)

    if any(bullish_engulfing) and trend == "Uptrend":
        bues = "Bullish Engulfing pattern detected in an uptrend. Consider selling."
        List.append(bues)

    if any(bearish_engulfing) and trend == "Downtrend":
        beb = "Bearish Engulfing pattern detected in a downtrend. Consider buying."
        List.append(beb)

    if any(bullish_harami) and trend == "Uptrend":
        buhs = "Bullish Harami pattern detected in an uptrend. Consider selling."
        List.append(buhs)

    if any(bearish_harami) and trend == "Downtrend":
        behb = "Bearish Harami pattern detected in a downtrend. Consider buying."
        List.append(behb)

    if any(is_morning_star) and trend == "Uptrend":
        mss = "Morning Star pattern detected in an uptrend. Consider selling."
        List.append(mss)

    if any(is_evening_star) and trend == "Downtrend":
        evb = "Evening Star pattern detected in a downtrend. Consider buying."
        List.append(evb)
        
    if trend != "Downtrend" and trend != "Uptrend":
        trd = "Choppy Market identified. Stay away from the market."
        List.append(trd)
        
    concatenated_string = "\n".join(List)
    
    if concatenated_string != "":
        return concatenated_string
    else:
        return "No pattern identified"




def candlepattern(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                candlepatter='Error accessing the forex market'
        else:
            candlepatter=candlepatterndt(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer= f'{candlepatter}  for Symbol : {symbol} and Timeframe: {timeframe}' 
        indicator='Japanese Candlestick Pattern'
        return render(request,'index.html',{'indicator':indicator,'candlepatter':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
    else:
        # Fetch the available pairs from the database
        pairs = symbols
        timeframees = timeframes
        answer='No analysis ran yet'
        indicator='Japanese Candlestick Pattern'
        return render(request,'index.html',{'indicator':indicator,'candlepatter':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})


def calculate_mfi(candles, period=14):
    dataframe = pd.DataFrame(candles)
    dataframe.set_index('time', inplace=True)
    print(dataframe)
    #dataframe['close'] = dataframe['close'].astype(float)
    # Calculate typical price
    dataframe['Typical Price'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

    # Calculate raw money flow
    dataframe['Raw Money Flow'] = dataframe['Typical Price'] * dataframe['tickVolume'].values

    # Calculate money flow ratio
    dataframe['Positive Money Flow'] = np.where(dataframe['Typical Price'] > dataframe['Typical Price'].shift(1), dataframe['Raw Money Flow'], 0)
    dataframe['Negative Money Flow'] = np.where(dataframe['Typical Price'] < dataframe['Typical Price'].shift(1), dataframe['Raw Money Flow'], 0)

    # Calculate money flow index
    dataframe['Positive Money Flow Sum'] = dataframe['Positive Money Flow'].rolling(window=period).sum()
    dataframe['Negative Money Flow Sum'] = dataframe['Negative Money Flow'].rolling(window=period).sum()
    dataframe['Money Flow Index'] = 100 - (100 / (1 + (dataframe['Positive Money Flow Sum'] / dataframe['Negative Money Flow Sum'])))

    # Determine buy or sell opportunities
    signals = []
    for i in range(period, len(dataframe)):
        if dataframe['Money Flow Index'].iloc[i] > 80:
            signals.append('Sell')
        elif dataframe['Money Flow Index'].iloc[i] < 20:
            signals.append('Buy')
        else:
            signals.append('Hold')

    most_common_signal = Counter(signals).most_common(1)[0][0]

    return most_common_signal

def mfi(request):
    if request.method == 'POST':
        timeframe = request.POST.get('timeframe')
        symbol = request.POST.get('pair')
        candles=asyncio.run(get_candles(timeframe,symbol))
        word1="Error"
        if type(candles)== str:
            if word1.lower() in candles.lower():
                mfi='Error accessing the forex market'
        else:
            mfi=calculate_mfi(candles)
        
        pairs = symbols
        timeframees = timeframes
        
        answer=f'{mfi}  for Symbol : {symbol} and Timeframe: {timeframe}'
        indicator='Money Flow Index'
        return render(request,'index.html',{'indicator':indicator,'pivot_points':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Money Flow Index'
    return render(request,'index.html',{'indicator':indicator,'pivot_points':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
async def calculate_margin(symbol,volume,order_type):
    try:
        api = MetaApi(token)
        account = await api.metatrader_account_api.get_account(accountId)
        initial_state = account.state
        deployed_states = ['DEPLOYING', 'DEPLOYED']
        if initial_state not in deployed_states:
            # wait until account is deployed and connected to broker
            print('Deploying account')
            await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
        # connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()
        
        # wait until terminal state synchronized to the local state
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()
        # calculate margin required for trade
        prices = await connection.get_symbol_price(symbol)
        print(prices)
        current_price = prices['ask']
        print(current_price)
        volume=float(volume)
        print(type(symbol),type(order_type),symbol,order_type,type(volume),volume)
        first_margin= await connection.calculate_margin({
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'openPrice':  current_price
        })
        print(first_margin)
        first_margin=float(first_margin['margin'])
        print(first_margin)
        return first_margin
    except Exception as e:
        return f"Error : {e}"

def margin(request):
    m_order_type=['SELL', 'BUY',]
    m_volume=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        volume = request.POST.get('volume')
        order_types =str(request.POST.get('order_type'))
        word2='SELL'
        if word2.lower() in order_types.lower():
            order_type='ORDER_TYPE_SELL'
        else:
            order_type='ORDER_TYPE_BUY'
        try:
            mfi= asyncio.run(calculate_margin(symbol,volume,order_type))
        except Exception as e:
            mfi=f"Error calculating margin: {e}"
            return mfi
        pairs = symbols
        timeframees = timeframes
        
        answer=f'${mfi} margin  for Symbol : {symbol}, Lot size: {volume} and Order Type: {order_types}'
        indicator='Margin Calculation'
        return render(request,'index.html',{'indicator':indicator,'margin':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})
    # Fetch the available pairs from the database
    pairs = symbols
    timeframees = timeframes
    answer='No analysis ran yet'
    indicator='Margin Calculation'

    return render(request,'index.html',{'m_volume':m_volume,'m_order_type':m_order_type,'indicator':indicator,'margin':'head', 'pairs':pairs,'timeframes':timeframees,'answer':answer})




def trend(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                ichimuko=calculate_ichimoku(candles)
                ma=moving_average(candles)
                ema=exponential_moving_average(candles)
                sma=simple_moving_average(candles)
                macd=calculate_macd(candles,n_fast, n_slow, n_signal)
                parabolic=parabolic_sar(candles)
                adx=calculate_adx(candles)
                smi=calculate_smi(candles)
                List.append(ichimuko)
                List.append(ma)
                List.append(ema)
                List.append(sma)
                List.append(macd)
                List.append(parabolic)
                List.append(adx)
                List.append(smi)

                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Trend Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Trend Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})


def volatility(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                atr=average_true_range(candles)

                List.append(atr)
                bb=calculate_bb(candles)
                List.append(bb)
                std=standard_deviation(candles)
                List.append(std)
                chv=chaikins_volatility(candles)
                List.append(chv)
                dc=donchian_channels(candles)
                List.append(dc)

                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Volatility Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Volatility Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

def momentum(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                rsi=calculate_rsi(candles)
                List.append(rsi)
                sto=stochastic_oscillator(candles)
                List.append(sto)
                cci=cci_c(candles)
                List.append(cci)
                money=cal_momentum(candles)
                List.append(money)
                roc=rate_of_change(candles)
                List.append(roc)


                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Momentum Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Momentum Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})


def volumn(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                obv=calculate_obv(candles)
                List.append(obv)
                cmf=calculate_cmf(candles)
                List.append(cmf)
                adl=calculate_adl(candles)
                List.append(adl)

                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Volumn Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Volume Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})


def oscillator(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                rsi=calculate_rsi(candles)
                List.append(rsi)
                sto=stochastic_oscillator(candles)
                List.append(sto)
                macd=calculate_macd(candles,n_fast, n_slow, n_signal)
                cci=cci_c(candles)
                List.append(macd)
                List.append(cci)
                wil=williams_percent_r(candles)
                List.append(wil)
                money=calculate_mfi(candles)
                List.append(money)
                roc=rate_of_change(candles)
                List.append(roc)

                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Osicillator Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Osicillator Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})


def ranging(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                pp=calculate_pivot_point(candles)
                List.append(pp)
                fib=calculate_fibonachi(candles)
                List.append(fib)
                gf=calculate_gannfan(candles)
                List.append(gf)
                sppt=cal_support_resistance(candles)
                List.append(sppt)
                trend=calculate_trendlines(candles)
                List.append(trend)

                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Ranging Market Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Ranging Market  Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})


def general(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        print(symbol)
        List=[]
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    concatenated_string='Error accessing the forex market'
            else:
                pp=calculate_pivot_point(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Pivot Points', 'Results': f'{pp}'}
                List.append(new_item)
                fib=calculate_fibonachi(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'fibonachi Retracement', 'Results': f'{fib}'}
                List.append(new_item)
                gf=calculate_gannfan(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Gann_Fan', 'Results': f'{gf}'}
                List.append(new_item)
                sppt=cal_support_resistance(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Support Resistance', 'Results': f'{sppt}'}
                List.append(new_item)
                trend=calculate_trendlines(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Trendlines', 'Results': f'{trend}'}
                List.append(new_item)
                rsi=calculate_rsi(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Relative Strength Index', 'Results': f'{rsi}'}
                List.append(new_item)
                sto=stochastic_oscillator(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Stochastic Oscillator', 'Results': f'{sto}'}
                List.append(new_item)
                macd=calculate_macd(candles,n_fast, n_slow, n_signal)
                cci=cci_c(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Moving Average Convergence Divergence', 'Results': f'{macd}'}
                List.append(new_item)
                new_item = {'time': f'{timeframe}', 'indicator': 'Commodity Channel Index', 'Results': f'{cci}'}
                List.append(new_item)
                wil=williams_percent_r(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Williams_%R', 'Results': f'{wil}'}
                List.append(new_item)
                money=calculate_mfi(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                roc=rate_of_change(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                bv=calculate_obv(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                cmf=calculate_cmf(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                adl=calculate_adl(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                sto=stochastic_oscillator(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                money=cal_momentum(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                roc=rate_of_change(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                bb=calculate_bb(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                std=standard_deviation(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                chv=chaikins_volatility(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                dc=donchian_channels(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                ichimuko=calculate_ichimoku(candles)
                new_item = {'time': f'{timeframe}', 'indicator': '', 'Results': f'{pp}'}
                List.append(new_item)
                ma=moving_average(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Moving Average', 'Results': f'{ma}'}
                List.append(new_item)
                ema=exponential_moving_average(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Exponential_Moving_Average', 'Results': f'{ema}'}
                List.append(new_item)
                sma=simple_moving_average(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Simple Moving Average', 'Results': f'{sma}'}
                List.append(new_item)
                parabolic=parabolic_sar(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Parabolic Sar', 'Results': f'{parabolic}'}
                List.append(new_item)
                adx=calculate_adx(candles)
                new_item = {'time': f'{timeframe}', 'indicator': 'Average Directional Index(ADX)', 'Results': f'{adx}'}
                List.append(new_item)
                smi=calculate_smi(candles)
                new_item = {'time': f'{timeframe}', 'indicator': f'Simple Moving Average', 'Results': f'{smi}'}
                List.append(new_item)

                concatenated_string = "\n".join(List)

            
        pairs = symbols
        answer=f'{concatenated_string}  \n For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
        indicator='All Ranging Market Indicators'
        return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All Ranging Market  Indicators'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})


import csv
def detect_shapes(request):
    if request.method == 'POST':
        symbol = request.POST.get('pair')
        results = []
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            word1="Error"
            if type(candles)== str:
                if word1.lower() in candles.lower():
                    answer='Error accessing the forex market'
            else:
                
                dt=cal_double_tops(candles)
                abcd=cal_identify_abcd_pattern(candles)
                bp=cal_identify_bat_pattern(candles)
                bup=cal_identify_butterfly_pattern(candles)
                cp=cal_identify_crab_pattern(candles)
                cyp=cal_identify_cypher_pattern(candles)
                db=cal_identify_double_bottom(candles)
                gp=cal_identify_gartley_pattern(candles)
                sp=cal_identify_shark_pattern(candles)
                tdp=cal_identify_three_drive_pattern(candles)
                hns=cal_identify_head_shoulders_pattern(candles)
                chaup=cal_channel_up(candles)
                chadn=cal_channel_down(candles)
                jcp=identify_candle_pattern(candles)
                asc=analyze_ascending_triangle(candles)
                dsc=analyze_descending_triangle(candles)
                flag=flag_pattern(candles)
                tripplet=triple_top(candles)
                trippleb=triple_bottom(candles)
                rectangle=rectangle(candles)
                pennant=pennant(candles)
                symmetrical=symmetrical_triangle(candles)
                wedge=wedge(candles)
                diam=diamond(candles)
                ihns=inverse_head_and_shoulders(candles)
                cnh=cup_and_handle(candles)
                if flag is not None:
                    results.append({'Pattern': 'flag Pattern', 'Decision': flag,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if tripplet is not None:
                    results.append({'Pattern': 'Tripple Top', 'Decision': tripplet,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if trippleb is not None:
                    results.append({'Pattern': 'Tripple  Bottom', 'Decision': trippleb,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if rectangle is not None:
                    results.append({'Pattern': 'Rectangle', 'Decision': rectangle,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if pennant is not None:
                    results.append({'Pattern': 'Pennant', 'Decision': pennant,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if symmetrical is not None:
                    results.append({'Pattern': 'Symmetrical Triangle', 'Decision': symmetrical,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if wedge is not None:
                    results.append({'Pattern': 'Wedge', 'Decision': wedge,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if diam is not None:
                    results.append({'Pattern': 'Diamond', 'Decision': diam,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if ihns is not None:
                    results.append({'Pattern': 'Inverse Head and Shoulders', 'Decision': ihns,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if cnh is not None:
                    results.append({'Pattern': 'Cup and Handle', 'Decision': cnh,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if asc is not None:
                    results.append({'Pattern': 'Ascending Triangle', 'Decision': asc,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if dsc is not None:
                    results.append({'Pattern': 'Descending Triangle', 'Decision': dsc,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if jcp is not None:
                    results.append({'Pattern': 'Japanese Candle Pattern', 'Decision': jcp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if chaup is not None:
                    results.append({'Pattern': 'Channel Up', 'Decision': chaup,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if chadn is not None:
                    results.append({'Pattern': 'Channel Down', 'Decision': chadn,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if hns is not None:
                    results.append({'Pattern': 'Head and Shoulders', 'Decision': hns,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if dt is not None:
                    results.append({'Pattern': 'Double Tops', 'Decision': dt,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if abcd is not None:
                    results.append({'Pattern': 'ABCD Pattern', 'Decision': abcd,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if bp is not None:
                    results.append({'Pattern': 'Bat Pattern', 'Decision': bp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if bup is not None:
                    results.append({'Pattern': 'Butterfly Pattern', 'Decision': bup,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if cp is not None:
                    results.append({'Pattern': 'Crab Pattern', 'Decision': cp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if cyp is not None:
                    results.append({'Pattern': 'Cypher Pattern', 'Decision': cyp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if db is not None:
                    results.append({'Pattern': 'Double Bottoms', 'Decision': db,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if gp is not None:
                    results.append({'Pattern': 'Gartley Pattern', 'Decision': gp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if sp is not None:
                    results.append({'Pattern': 'Shark Pattern', 'Decision': sp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})
                if tdp is not None:
                    results.append({'Pattern': 'Three Drive Pattern', 'Decision': tdp,'Timeframe':f'{timeframe}','Symbol':f'{symbol}'})

                
                answer="Market Analysied Successfully"#f'{concatenated_string}  </br> For Symbol :<b><u> {symbol}</u></b> \n Timeframe: <b><u> {timeframe}</u></b>'
            for result in results:
                result['Symbol'] = symbol
        table_data = results
        pairs = symbols
        
        indicator='All  Shape(Harmonics) Scan'
        return render(request,'table.html',{'table_data':table_data,'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})

    # Fetch the available pairs from the database
    pairs = symbols
    answer='No analysis ran yet'
    indicator='All  Shape(Harmonics) Scan'
    return render(request,'table.html',{'indicator':indicator,'atr':'head', 'pairs':pairs,'answer':answer})







