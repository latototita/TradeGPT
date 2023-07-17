import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd



token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or 'cf6ff7fe-5929-4349-872a-841cac56f7dc'
# Define the Head and Shoulders pattern ratios
head_shoulders_ratios = {
    "AB": 0.382,
    "BC": 0.8,
    "CD": 1.272
}
# Define the price target percentage
price_target_percent = 5  # 5% price target

# Define the Gartley pattern ratios
gartley_ratios = {
    "XA": 0.8,
    "AB": 0.382,
    "BC": 0.886,
    "CD": 1.272
}

# Define the Butterfly pattern ratios
butterfly_ratios = {
    "XA": 0.786,
    "AB": 0.382,
    "BC": 0.8,
    "CD": 1.8
}


# Define the Bat pattern ratios
bat_ratios = {
    "XA": 0.382,
    "AB": 0.382,
    "BC": 0.382,
    "CD": 2.8
}

# Define the Crab pattern ratios
crab_ratios = {
    "XA": 0.382,
    "AB": 0.8,
    "BC": 2.24,
    "CD": 3.8
}


# Define the Shark pattern ratios
shark_ratios = {
    "XA": 0.886,
    "AB": 0.5,
    "BC": 0.382,
    "CD": 2.24
}

# Define the Cypher pattern ratios
cypher_ratios = {
    "XA": 0.786,
    "AB": 0.382,
    "BC": 0.8,
    "CD": 1.272
}


# Define the ABCD pattern ratios
abcd_ratios = {
    "AB": 0.382,
    "BC": 0.8,
    "CD": 1.272
}

# Define the Three-Drive pattern ratios
three_drive_ratios = {
    "AB": 0.382,
    "BC": 0.8,
    "CD": 1.272,
    "DE": 0.8,
    "EF": 1.272
}

# Define the Head and Shoulders pattern ratios
head_shoulders_ratios = {
    "AB": 0.382,
    "BC": 0.8,
    "CD": 1.272
}

# Initialize MetaApi client

timeframe = '1m'

# Define parameters
symbol_list = ['XAUUSDm','XAGUSDm','EURJPYm','USDJPYm','EURUSDm','AUDUSDm','USDCHFm','GBPTRYm','USDTRYm','USDCADm','NZDUSDm','GBPCHFm','EURCHFm','EURGBPm','EURAUDm','CHFJPYm','AUDJPYm','AUDNZDm','GBPJPYm','GBPUSDm',]
rsi_period = 14

import pandas as pd
import numpy as np
import pandas_ta as ta

def get_price_target(pattern, prices):
    # Calculate the price target based on the pattern identified
    if pattern == "Bat":
        cd = prices[4] - prices[3]
        target = prices[4] + (cd * bat_ratios["CD"])
    elif pattern == "Gartley":
        cd = prices[4] - prices[3]
        target = prices[4] + (cd * gartley_ratios["CD"])
    elif pattern == "Butterfly":
        cd = prices[4] - prices[3]
        target = prices[4] + (cd * butterfly_ratios["CD"])
    elif pattern == "Crab":
        cd = prices[4] - prices[3]
        target = prices[4] + (cd * crab_ratios["CD"])
    elif pattern == "Shark":
        cd = prices[4] - prices[3]
        target = prices[4] + (cd * shark_ratios["CD"])
    elif pattern == "Cypher":
        cd = prices[4] - prices[3]
        target = prices[4] + (cd * cypher_ratios["CD"])
    elif pattern == "Double Tops":
        target = prices[-1] - (prices[-2] - prices[-1])
    elif pattern == "Double Bottoms":
        target = prices[-1] + (prices[-1] - prices[-2])
    elif pattern == "ABCD":
        cd = prices[3] - prices[2]
        target = prices[3] + (cd * abcd_ratios["CD"])
    elif pattern == "Three-Drive":
        ef = prices[5] - prices[4]
        target = prices[5] + (ef * three_drive_ratios["EF"])
    elif pattern == "Head and Shoulders":
        cd = prices[3] - prices[2]
        target = prices[3] + (cd * head_shoulders_ratios["CD"])
    else:
        target = None

    return target







def identify_candle_pattern(candles):
    # Identify candlestick patterns
    bullish_engulfing = (
        candles[-2]['close'] < candles[-2]['open'] and
        candles[-1]['close'] > candles[-1]['open'] and
        candles[-1]['close'] > candles[-2]['open'] and
        candles[-1]['open'] < candles[-2]['close']
    )
    bearish_engulfing = (
        candles[-2]['close'] > candles[-2]['open'] and
        candles[-1]['close'] < candles[-1]['open'] and
        candles[-1]['close'] < candles[-2]['open'] and
        candles[-1]['open'] > candles[-2]['close']
    )
    hammer = (
        candles[-1]['close'] > candles[-1]['open'] and
        candles[-1]['close'] > candles[-1]['low'] + 0.6 * (candles[-1]['high'] - candles[-1]['low']) and
        candles[-1]['open'] > candles[-1]['low'] + 0.6 * (candles[-1]['high'] - candles[-1]['low'])
    )
    inverted_hammer = (
        candles[-1]['close'] > candles[-1]['open'] and
        candles[-1]['close'] > candles[-1]['low'] + 0.6 * (candles[-1]['high'] - candles[-1]['low']) and
        candles[-1]['open'] < candles[-1]['low'] + 0.4 * (candles[-1]['high'] - candles[-1]['low'])
    )
    shooting_star = (
        candles[-1]['close'] < candles[-1]['open'] and
        candles[-1]['close'] < candles[-1]['low'] + 0.4 * (candles[-1]['high'] - candles[-1]['low']) and
        candles[-1]['open'] > candles[-1]['low'] + 0.6 * (candles[-1]['high'] - candles[-1]['low'])
    )
    hanging_man = (
        candles[-1]['close'] < candles[-1]['open'] and
        candles[-1]['close'] < candles[-1]['low'] + 0.4 * (candles[-1]['high'] - candles[-1]['low']) and
        candles[-1]['open'] < candles[-1]['low'] + 0.4 * (candles[-1]['high'] - candles[-1]['low'])
    )
    doji = (
        abs(candles[-1]['close'] - candles[-1]['open']) < 0.1 * (candles[-1]['high'] - candles[-1]['low'])
    )
    evening_star = (
        candles[-3]['close'] < candles[-3]['open'] and
        candles[-2]['close'] > candles[-2]['open'] and
        candles[-1]['close'] < candles[-1]['open'] and
        candles[-1]['close'] < 0.5 * (candles[-3]['close'] - candles[-3]['open']) + candles[-3]['open']
    )
    harami = (
        candles[-2]['close'] > candles[-2]['open'] and
        candles[-1]['close'] < candles[-1]['open'] and
        candles[-1]['close'] > candles[-2]['open'] and
        candles[-1]['open'] < candles[-2]['close']
    )
    
    # Identify the overall trend
    sma_ = np.mean([candle['close'] for candle in candles[-50:]])
    sma_200 = np.mean([candle['close'] for candle in candles[-200:]])
    trend = 'Up Trend' if sma_ > sma_200 else 'Down Trend'
    
    # Determine buy/sell decision and expected meaning based on the identified pattern and trend
    if bullish_engulfing and trend == 'Up Trend':
        decision = 'buy'
        meaning = 'Bullish Engulfing Pattern identified in an Up Trend'
    elif bearish_engulfing and trend == 'Down Trend':
        decision = 'sell'
        meaning = 'Bearish Engulfing Pattern identified in a Down Trend'
    elif hammer and trend == 'Down Trend':
        decision = 'buy'
        meaning = 'Hammer Pattern identified in a Down Trend'
    elif inverted_hammer and trend == 'Down Trend':
        decision = 'buy'
        meaning = 'Inverted Hammer Pattern identified in a Down Trend'
    elif shooting_star and trend == 'Up Trend':
        decision = 'sell'
        meaning = 'Shooting Star Pattern identified in an Up Trend'
    elif hanging_man and trend == 'Up Trend':
        decision = 'sell'
        meaning = 'Hanging Man Pattern identified in an Up Trend'
    elif doji:
        decision = 'No Trade'
        meaning = 'Doji Pattern identified'
    elif evening_star and trend == 'Up Trend':
        decision = 'sell'
        meaning = 'Evening Star Pattern identified in an Up Trend'
    elif harami and trend == 'Down Trend':
        decision = 'buy'
        meaning = 'Harami Pattern identified in a Down Trend'
    else:
        decision = 'No Trade'
        meaning = 'No recognizable pattern or conflicting signals'
    
    # Concatenate decision and meaning into results
    results = f"{decision}: {meaning}"
    return results

def triple_top(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Define pattern recognition parameter
    triple_top_length = 10
    
    triple_top_pattern = np.all(prices[-triple_top_length:] <= prices[-triple_top_length:].max()) and \
                        np.all(prices[-triple_top_length:] >= prices[-triple_top_length:].min())
    # Determine whether to buy or sell based on the identified patterns
    if triple_top_pattern:
        if prices[-1] < prices[-2]:
            return 'Action: sell'
        else:
            return 'Action: Hold'
    else:
        return 'Empty'

def triple_bottom(candles):
    prices=np.array([candle['close'] for candle in candles])
    triple_bottom_length = 10
    # Convert price data to numpy array
    
    triple_bottom_pattern = np.all(prices[-triple_bottom_length:] >= prices[-triple_bottom_length:].min()) and \
                          np.all(prices[-triple_bottom_length:] <= prices[-triple_bottom_length:].max())
    if triple_bottom_pattern:
        # Place your buy/sell order logic here based on your trading strategy for triple bottom patterns
        # Example:
        if prices[-1] > prices[-2]:
            return 'Action: buy'
        else:
            return 'Action: Hold'
    else:
        return 'Empty'
    
def cal_double_tops(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_double_tops(prices):
        # Check if the prices form a double top pattern
        if len(prices) >= 3:
            if prices[-3] < prices[-2] and prices[-2] >= prices[-1]:
                return True
        return False
    is_double_tops = identify_double_tops(prices)
    if is_double_tops:
        pattern = "Double Tops"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Double Tops"
            string1="Action: sell"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Double Tops"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings='None'
        return strings


def cal_identify_double_bottom(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_double_bottoms(prices):

        # Check if the prices form a double bottom pattern
        if len(prices) >= 3:
            if prices[-3] > prices[-2] and prices[-2] <= prices[-1]:
                return True
        return False
    
    is_double_bottoms = identify_double_bottoms(prices)

    if is_double_bottoms:
        pattern = "Double Bottoms"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Double Bottoms"
            string1="Action: buy"
            result=stringsO+"\n"+string1+"\n"
            return result
        else:
            stringsO="Pattern: Double Bottoms"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings='None'
        return strings

inverse_head_and_shoulders_length = 10

def inverse_head_and_shoulders(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Inverse Head and Shoulders pattern
    inverse_head_and_shoulders_pattern = np.all(prices[-inverse_head_and_shoulders_length:] >= prices[-inverse_head_and_shoulders_length:].max()) and \
                                        np.all(prices[-inverse_head_and_shoulders_length:] <= prices[-inverse_head_and_shoulders_length:].min())

    if inverse_head_and_shoulders_pattern:
        if prices[-1] > prices[-inverse_head_and_shoulders_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-inverse_head_and_shoulders_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return 'Empty'
    else:
        return 'No inverse found'



def cal_identify_head_shoulders_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_head_shoulders_pattern(prices):
        # Check if the price data matches the Head and Shoulders pattern ratios
        ab = prices[1] - prices[0]
        bc = prices[2] - prices[1]
        cd = prices[3] - prices[2]

        if ab > 0 and bc < 0 and cd > 0:
            ratio_ab = abs(ab / bc)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, head_shoulders_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, head_shoulders_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, head_shoulders_ratios["CD"], atol=0.01)
            ):
                return True

        return False

    is_head_shoulders = identify_head_shoulders_pattern(prices)

    if is_head_shoulders:
        pattern = "Head and Shoulders"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Head and Shoulders"
            string1="Action: buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Head and Shoulders"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings='None'
        return strings





overbought_threshold=70
oversold_threshold=20
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
    buy_signal = market_direction == "Sell" and dt['rsi'].iloc[-1] < 30 and dt['slowk'].iloc[-1] < 20 and dt['cci'].iloc[-1] < -100
    sell_signal = market_direction == "Buy" and dt['rsi'].iloc[-1] > 70 and dt['slowk'].iloc[-1] > 80 and dt['cci'].iloc[-1] > 100

    return buy_signal, sell_signal

def check_market_condition_upper(candles):
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
    buy_signal = market_direction == "Buy" and dt['rsi'].iloc[-1] > 30 or dt['slowk'].iloc[-1] > 20 or dt['cci'].iloc[-1] > -100
    sell_signal = market_direction == "Sell" and dt['rsi'].iloc[-1] < 70 or  dt['slowk'].iloc[-1] < 80 or dt['cci'].iloc[-1] < 100

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
                
                buy_signal, sell_signal = check_market_condition(biggerCandles)#check_signals_biggers(biggerCandles, rsi_period)
                if btimeframe=='1m':
                    timeframe='5m'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=check_market_condition_upper(candless)
                elif btimeframe=='5m':
                    timeframe='15m'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=check_market_condition_upper(candless)
                elif btimeframe=='15m':
                    timeframe='30m'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=check_market_condition_upper(candless)
                elif btimeframe=='30m':
                    timeframe='1h'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=check_market_condition_upper(candless)
                elif btimeframe=='1h':
                    timeframe='4h'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=check_market_condition_upper(candless)
                print(buy_signal,sell_signal)
                print(timeframez,btimeframe)
                dt=cal_double_tops(biggerCandles)
                db=cal_identify_double_bottom(biggerCandles)
                hns=cal_identify_head_shoulders_pattern(biggerCandles)
                jcp=identify_candle_pattern(biggerCandles)
                tripplet=triple_top(biggerCandles)
                trippleb=triple_bottom(biggerCandles)
                ihns=inverse_head_and_shoulders(biggerCandles)
                
                keywords = ["buy",]
                if sell_signal==True and Bsell_signal==True and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):#buy_signal==True and Bbuy_signal==True and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):
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
                        atr_multiplierx = 2
                        atr_multiplier = 5
                    elif df['atr'][-1] >2<=4:
                        atr_multiplierx =2
                        atr_multiplier = 5
                    elif df['atr'][-1] >4:
                        atr_multiplierx =1
                        atr_multiplier = 6
                    else:
                        atr_multiplierx =2
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

                keywords = ["sell",]
            
                if buy_signal==True and Bbuy_signal==True and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords): #sell_signal==True and Bsell_signal==True and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):

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
                        atr_multiplierx =2
                        atr_multiplier = 6
                    else:
                        atr_multiplierx =3
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
