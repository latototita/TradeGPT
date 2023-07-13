import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd

# Initialize MetaApi client
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or 'cf6ff7fe-5929-4349-872a-841cac56f7dc'
timeframe = '1m'

# Define parameters
symbol_list = ['XAUUSDm','XAGUSDm','EURJPYm','USDJPYm','EURUSDm','AUDUSDm','USDCHFm','GBPTRYm','USDTRYm','USDCADm','NZDUSDm','GBPCHFm','EURCHFm','EURGBPm','EURAUDm','CHFJPYm','AUDJPYm','AUDNZDm','GBPJPYm','GBPUSDm',]
rsi_period = 14

import pandas as pd
import numpy as np
import pandas_ta as ta

def check_signals(candles, rsi_period):
    # Convert candles to DataFrame
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)

    # Apply RSI indicator
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)

    # Apply ATR indicator
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Check the most recent crossing
    last_crossing_index = None
    last_crossing_threshold = None

    cross_above_70 = np.where(df['rsi'] > 67)[0]
    cross_below_30 = np.where(df['rsi'] < 33)[0]

    if len(cross_above_70) > 0:
        last_crossing_index = cross_above_70[-1]
        last_crossing_threshold = 70

    if len(cross_below_30) > 0 and (last_crossing_index is None or cross_below_30[-1] > last_crossing_index):
        last_crossing_index = cross_below_30[-1]
        last_crossing_threshold = 30

    if last_crossing_index is not None:
        last_crossing_value = df['rsi'][last_crossing_index]
        current_rsi_value = df['rsi'][-1]
        if abs(current_rsi_value - last_crossing_threshold) <= 10:
            if last_crossing_threshold == 70:
                trend_direction = "Downwards"
                sell_signal = current_rsi_value > 70 and last_crossing_threshold == 70
                buy_signal = not sell_signal
                if sell_signal:
                    print("Sell signal, crossed 70, now in a downtrend")
            else:
                trend_direction = "Upwards"
                buy_signal = current_rsi_value < 30 and last_crossing_threshold == 30
                sell_signal = not buy_signal
                if buy_signal:
                    print("Buy signal, crossed 30, now in an uptrend")

            print(f"The RSI crossed {last_crossing_threshold} at index {last_crossing_index} with a value of {last_crossing_value}.")
            print(f"The current trend direction is {trend_direction}.")

            return buy_signal, sell_signal
        else:
            return False, False
    else:
        return False, False


def check_signals_biggerFF(candles, rsi_period):
    # Convert candles to DataFrame
    dt = pd.DataFrame(candles)
    dt.set_index('time', inplace=True)
    dt['close'] = dt['close'].astype(float)

    # Apply RSI indicator
    dt['rsi'] = ta.rsi(dt['close'], length=rsi_period)

    # Apply ATR indicator
    dt['atr'] = ta.atr(dt['high'], dt['low'], dt['close'], length=14)

    # Check the most recent crossing
    last_crossing_index = None
    last_crossing_threshold = None

    cross_above_70 = np.where(dt['rsi'] > 65)[0]
    cross_below_30 = np.where(dt['rsi'] < 35)[0]

    if len(cross_above_70) > 0:
        last_crossing_index = cross_above_70[-1]
        last_crossing_threshold = 70

    if len(cross_below_30) > 0 and (last_crossing_index is None or cross_below_30[-1] > last_crossing_index):
        last_crossing_index = cross_below_30[-1]
        last_crossing_threshold = 30

    if last_crossing_index is not None:
        last_crossing_value = dt['rsi'][last_crossing_index]
        current_rsi_value = dt['rsi'][-1]
        if abs(current_rsi_value - last_crossing_threshold) <= 10:
            if last_crossing_threshold == 70:
                trend_direction = "Downwards"
                sell_signal = current_rsi_value > 65 and last_crossing_threshold == 70
                buy_signal=not sell_signal
                if sell_signal:
                    print("Bigger Sell signal, crossed 70, now in a downtrend")

            else:
                trend_direction = "Upwards"
                buy_signal = current_rsi_value < 36 and last_crossing_threshold == 30
                sell_signal = not buy_signal
                if buy_signal:
                    print(" Bigger Buy signal, crossed 30, now in an uptrend")

            return buy_signal, sell_signal
        else:
            return False, False
    else:
        return False, False


overbought_threshold=78
oversold_threshold=23
rsi_period=14
sma_period=20
def calculate_rsi_sma(data, period=14):
    delta = data.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi + 10
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
    return K,D

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
    return cci_value
def check_market_condition(candles):
    dt = pd.DataFrame(candles)
    dt.set_index('time', inplace=True)
    dt['close'] = dt['close'].astype(float)

    # Calculate RSI based on SMA method
    dt['rsi'] = calculate_rsi_sma(dt['close'], period=rsi_period)

    def determine_market_direction(dt):
        # Apply other indicators or calculations as needed
        dt['sma'] = dt['close'].rolling(window=sma_period).mean()

        if dt['close'].iloc[-1] > dt['sma'].iloc[-1]:
            return "Buy"
        elif dt['close'].iloc[-1] < dt['sma'].iloc[-1]:
            return "Sell"
        else:
            return "Neutral"
    dt['slowk'], dt['slowd'] = stochastic_oscillator(candles)
    dt['cci'] = cci_c(candles)
    market_direction = determine_market_direction(dt)
    print(dt['rsi'].iloc[-1])
    buy_signal = market_direction == "Sell" and dt['rsi'].iloc[-1] < oversold_threshold and dt['slowk'].iloc[-1] < 20 and dt['cci'].iloc[-1] < -100
    sell_signal = market_direction == "Buy" and dt['rsi'].iloc[-1] > overbought_threshold and dt['slowk'].iloc[-1] > 80 and dt['cci'].iloc[-1] > 100

    return buy_signal, sell_signal


def check_signals_biggers(candles, rsi_period):
    # Convert candles to DataFrame
    dt = pd.DataFrame(candles)
    dt.set_index('time', inplace=True)
    dt['close'] = dt['close'].astype(float)

    # Apply RSI indicator
    dt['rsi'] = ta.rsi(dt['close'], length=rsi_period)

    # Apply other indicators or calculations as needed
    dt['sma'] = ta.sma(dt['close'], timeperiod=20)
    dt['ema'] = ta.ema(dt['close'], timeperiod=20)

    # Check the most recent crossing
    last_crossing_index = None
    last_crossing_threshold = None

    cross_above_70 = np.where(dt['rsi'] > 70)[0]
    cross_below_30 = np.where(dt['rsi'] < 30)[0]

    if len(cross_above_70) > 0:
        last_crossing_index = cross_above_70[-1]
        last_crossing_threshold = 70

    if len(cross_below_30) > 0:
        if last_crossing_index is None or cross_below_30[-1] > last_crossing_index:
            last_crossing_index = cross_below_30[-1]
            last_crossing_threshold = 30

    if last_crossing_index is not None:
        last_crossing_value = dt['rsi'][last_crossing_index]
        current_rsi_value = dt['rsi'].iloc[-1]

        # Additional conditions for trend confirmation
        confirm_trend_condition = (dt['close'].iloc[-1] > dt['sma'].iloc[-1]) and (dt['sma'].iloc[-1] > dt['ema'].iloc[-1])

        if abs(current_rsi_value - last_crossing_value) <= 10 and confirm_trend_condition:
            trend_direction = "Downwards" if last_crossing_threshold == 70 else "Upwards"
            sell_signal = current_rsi_value >= 60 and abs(last_crossing_value - 70) <= 10
            buy_signal = current_rsi_value <= 40 and abs(last_crossing_value - 30) <= 10

            if sell_signal:
                print("Bigger Sell signal: RSI crossed above 70, indicating a downtrend.")
            elif buy_signal:
                print("Bigger Buy signal: RSI crossed below 30, indicating an uptrend.")

            return buy_signal, sell_signal

    return False, False


def detect_trend_reversals(candles, rsi_period=14, macd_fast_period=12, macd_slow_period=26, macd_signal_period=9,
                           stoch_period=14, rsi_threshold=70, macd_threshold=0, stoch_threshold=20,
                           rsi_lookback=3):
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    def calculate_ema(data, n):
        alpha = 2 / (n + 1)
        ema = np.array([data[0]])
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
        closing_prices = np.array([candle['close'] for candle in candles])
        ema_fast = calculate_ema(closing_prices, n_fast)
        ema_slow = calculate_ema(closing_prices, n_slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, n_signal)
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    # Calculate RSI
    rsi= ta.rsi(df['close'], length=rsi_period)

    # Calculate MACD
    macd_line, signal_line, macd_histogram = calculate_macd(candles, macd_fast_period, macd_slow_period, macd_signal_period)


    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    highest_high = np.max(high[-stoch_period:])
    lowest_low = np.min(low[-stoch_period :])
    current_close = close[-1]

    slowk = (current_close - lowest_low) / (highest_high - lowest_low) * 100
    slowd = np.mean(slowk)
    # Detect potential trend reversal for buying opportunity
    def detect_buy_reversal():
        if rsi[-1] <= rsi_threshold and macd_histogram[-1] > macd_threshold and slowk[-1] < stoch_threshold \
                and slowd[-1] < stoch_threshold:
            for i in range(1, rsi_lookback + 1):
                if rsi[-i] <= rsi[-i - 1]:
                    return False
            return True
        return False

    # Detect potential trend reversal for selling opportunity
    def detect_sell_reversal():
        if rsi[-1] >= rsi_threshold and macd_histogram[-1] < macd_threshold and slowk[-1] > stoch_threshold \
                and slowd[-1] > stoch_threshold:
            for i in range(1, rsi_lookback + 1):
                if rsi[-i] >= rsi[-i - 1]:
                    return False
            return True
        return False

    # Detect potential trend reversal for buying and selling opportunities
    buy_reversal = detect_buy_reversal()
    sell_reversal = detect_sell_reversal()

    return buy_reversal, sell_reversal



def run_trading_bot(timeframez,btimeframe):
    async def main():
        # Connect to the MetaTrader account

        api = MetaApi(token)
        account = await api.metatrader_account_api.get_account(accountId)
        initial_state = account.state
        deployed_states = ['DEPLOYING', 'DEPLOYED']
        if initial_state not in deployed_states:
            # Wait until the account is deployed and connected to the broker
            print('Deploying account')
            await account.deploy()
        print('Waiting for API server to connect to the broker (may take a few minutes)')
        await account.wait_connected()

        # Connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()

        # Wait until terminal state synchronized to the local state
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()
        
        # Check for open trades
        trades = await connection.get_positions()#connection.get_orders()
        if trades:
            print("There are open trades. Skipping analysis.")
            await asyncio.sleep(1200)
        else:
            for symbol in symbol_list:
                print(symbol)
                trades = await connection.get_positions()
                if trades:
                    print("There are open trades. Skipping analysis.")
                    await asyncio.sleep(1200)
                    continue
                try:
                    # Fetch historical price data
                    biggerCandles = await account.get_historical_candles(symbol=symbol, timeframe=btimeframe, start_time=None, limit=1000)
                except Exception as e:
                    print(f"Error retrieving candle data: {e}")

                #buy_signal, sell_signal = check_signals(candles, rsi_period)
                #Bbuy_signal, Bsell_signal = detect_trend_reversals(biggerCandles)
                
                Bbuy_signal, Bsell_signal = check_market_condition(biggerCandles)#check_signals_biggers(biggerCandles, rsi_period)
                print(Bbuy_signal,Bsell_signal)
                print(timeframez,btimeframe)
                if Bbuy_signal==True:
                    # Execute trading orders
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframez, start_time=None, limit=1000)
                        
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                            
                    atr_multiplier = 15.0  # Multiplier to determine the distance from the current price based on ATR
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df['close'] = df['close'].astype(float)
                    # Apply ATR indicator
                    
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    # Determine the ATR multiplier based on the atr_value
                    if df['atr'][-1]>1<=2:
                        atr_multiplierx = 4
                        atr_multiplier = 5
                    elif df['atr'][-1] >2<=4:
                        atr_multiplierx =2
                        atr_multiplier = 5
                    elif df['atr'][-1] >4:
                        atr_multiplierx =1
                        atr_multiplier = 6
                    else:
                        atr_multiplierx =6
                        atr_multiplier = 6
                    take_profit = current_price + (atr_multiplierx* df['atr'][-1])
                    stop_loss = current_price - (atr_multiplier * df['atr'][-1])
                    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
                    print('atr',df['atr'][-1])
                    
                    try:
                        # Calculate margin required for trade
                        first_margin = await connection.calculate_margin({
                            'symbol': symbol,
                            'type': 'ORDER_TYPE_BUY',
                            'volume': 0.01,
                            'openPrice':  current_price
                        })
                        first_margins = float(first_margin['margin'])
                        if first_margins < ((10/100) * 10):
                            result = await connection.create_market_buy_order(
                                symbol,
                                0.01,
                                stop_loss,
                                take_profit,
                                {'trailingStopLoss': {
                                    'distance': {
                                        'distance': 1,
                                        'units': 'RELATIVE_BALANCE_PERCENTAGE'
                                    }
                                }
                            })
                            print('Trade successful, result code is ' + result['stringCode'])
                            continue
                        
                    except Exception as err:
                        print('Trade failed with error:')
                        print(api.format_error(err))
                    
                if Bsell_signal==True:

                    # Execute trading orders
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframez, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                            
                     # Multiplier to determine the distance from the current price based on ATR
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df['close'] = df['close'].astype(float)
                    # Apply ATR indicator
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    # Determine the ATR multiplier based on the atr_value
                    if df['atr'][-1]>1<=2:
                        atr_multiplierx = 4
                        atr_multiplier = 5
                    elif df['atr'][-1] >2<=4:
                        atr_multiplierx =2
                        atr_multiplier = 5
                    elif df['atr'][-1] >4:
                        atr_multiplierx =1
                        atr_multiplier = 6
                    else:
                        atr_multiplierx =6
                        atr_multiplier = 6
                    take_profit = current_price - (atr_multiplierx * df['atr'][-1])
                    stop_loss = current_price + (atr_multiplier * df['atr'][-1])
                    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
                    
                    
                    try:
                        # Calculate margin required for trade
                        first_margin = await connection.calculate_margin({
                            'symbol': symbol,
                            'type': 'ORDER_TYPE_SELL',
                            'volume': 0.01,
                            'openPrice':  current_price,
                        })
                        first_margins = float(first_margin['margin'])
                        if first_margins < ((10/100) * 10):
                            result = await connection.create_market_sell_order(
                                symbol,
                                0.01,
                                stop_loss,
                                take_profit,
                                {'trailingStopLoss': {
                                    'distance': {
                                        'distance': 1,
                                        'units': 'RELATIVE_BALANCE_PERCENTAGE'
                                    }
                                }
                            })
                            print('Trade successful, result code is ' + result['stringCode'])
                            continue
                    except Exception as err:
                        print('Trade failed with error:')
                        print(api.format_error(err))
                    

                trades = await connection.get_positions()
                if trades:
                    await asyncio.sleep(1200)
                else:
                    print("--------------------------------------------------")
        await asyncio.sleep(3)  # Sleep for 1 minute before the next iteration
    asyncio.run(main())


'''
# Define the specific combinations of bigger and smaller timeframes
timeframe_combinations = [('1m', '5m'), ('15m', '5m'), ('30m', '15m'), ('1h', '30m'), ('4h', '1h')]

# Call the trading bot function for each combination
for smaller_tf, bigger_tf in timeframe_combinations:
    run_trading_bot(timeframez =smaller_tf,btimeframe= bigger_tf)
'''
import multiprocessing

# Define the specific combinations of bigger and smaller timeframes
timeframe_combinations = [('1m', '5m'), ('15m', '5m'), ('30m', '15m'), ('1h', '30m'), ('4h', '1h')]

# Function to run the trading bot
def run_trading_bot_wrapper(timeframe_tuple):
    smaller_tf, bigger_tf = timeframe_tuple
    print(f"Running with timeframe: {smaller_tf} - {bigger_tf}")
    run_trading_bot(timeframez=smaller_tf, btimeframe=bigger_tf)
    print(f"Finished with timeframe: {smaller_tf} - {bigger_tf}")

# Create a multiprocessing pool
with multiprocessing.Pool() as pool:
    # Map the tasks to the pool for parallel execution
    pool.map(run_trading_bot_wrapper, timeframe_combinations)
