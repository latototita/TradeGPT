import pandas as pd
import numpy as np
import pandas_ta as ta

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
    buy_signal = market_direction == "Buy" and dt['rsi'].iloc[-1] > 40 or dt['slowk'].iloc[-1] > 20 or dt['cci'].iloc[-1] > -100
    sell_signal = market_direction == "Sell" and dt['rsi'].iloc[-1] < 60 or  dt['slowk'].iloc[-1] < 80 or dt['cci'].iloc[-1] < 100

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


def generate_entry_signal(candles):
    prices=np.array([candle['close'] for candle in candles])
    short_ma_period = 10  # Shorter moving average period
    long_ma_period = 20  # Longer moving average period

    # Calculate the moving averages
    short_ma = sum(prices[-short_ma_period:]) / short_ma_period
    long_ma = sum(prices[-long_ma_period:]) / long_ma_period

    # Check for a bullish crossover (short ma crossing above long ma)
    buy_signal= short_ma > long_ma and prices[-2] < short_ma

        

    # Check for a bearish crossover (short ma crossing below long ma)
    sell_signal=short_ma < long_ma and prices[-2] > short_ma

    return buy_signal,sell_signal
