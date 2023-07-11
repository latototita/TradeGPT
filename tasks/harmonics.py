


import numpy as np
# Define the tolerance for pattern matching
tolerance = 0.01
rectangle_length = 10
pennant_length = 5
symmetrical_triangle_length = 10
wedge_length = 10
diamond_length = 10
cup_and_handle_length = 10
inverse_head_and_shoulders_length = 10

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


def cal_identify_gartley_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_gartley_pattern(prices):
        # Check if the price data matches the Gartley pattern ratios
        xa = prices[1] - prices[0]
        ab = prices[2] - prices[1]
        bc = prices[3] - prices[2]
        cd = prices[4] - prices[3]

        if xa > 0 and ab < 0 and bc > 0 and cd < 0:
            ratio_ab = abs(ab / xa)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, gartley_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, gartley_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, gartley_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    is_gartley = identify_gartley_pattern(prices)
    if is_gartley:
        pattern = "Gartley"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Gartley"
            string1="Action: Buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Gartley"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings
def cal_identify_butterfly_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_butterfly_pattern(prices):
        # Check if the price data matches the Butterfly pattern ratios
        xa = prices[1] - prices[0]
        ab = prices[2] - prices[1]
        bc = prices[3] - prices[2]
        cd = prices[4] - prices[3]

        if xa < 0 and ab > 0 and bc < 0 and cd > 0:
            ratio_ab = abs(ab / xa)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, butterfly_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, butterfly_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, butterfly_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    is_butterfly = identify_butterfly_pattern(prices)
    if is_butterfly:
        pattern = "Butterfly"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Butterfly"
            string1="Action: Sell"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Butterfly"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings
def cal_identify_bat_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_bat_pattern(prices):
        # Check if the price data matches the Bat pattern ratios
        xa = prices[1] - prices[0]
        ab = prices[2] - prices[1]
        bc = prices[3] - prices[2]
        cd = prices[4] - prices[3]

        if xa > 0 and ab < 0 and bc > 0 and cd < 0:
            ratio_ab = abs(ab / xa)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, bat_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, bat_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, bat_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    is_bat = identify_bat_pattern(prices)

    if is_bat:
        pattern = "Bat"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Bat"
            string1="Action: Buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Bat"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings
def cal_identify_crab_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_crab_pattern(prices):
        # Check if the price data matches the Crab pattern ratios
        xa = prices[1] - prices[0]
        ab = prices[2] - prices[1]
        bc = prices[3] - prices[2]
        cd = prices[4] - prices[3]

        if xa > 0 and ab < 0 and bc > 0 and cd < 0:
            ratio_ab = abs(ab / xa)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, crab_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, crab_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, crab_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    
    is_crab = identify_crab_pattern(prices)

    if is_crab:
        pattern = "Crab"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Crab"
            string1="Action: Sell"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Crab"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings




def cal_identify_shark_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_shark_pattern(prices):
        # Check if the price data matches the Shark pattern ratios
        xa = prices[1] - prices[0]
        ab = prices[2] - prices[1]
        bc = prices[3] - prices[2]
        cd = prices[4] - prices[3]

        if xa > 0 and ab < 0 and bc > 0 and cd < 0:
            ratio_ab = abs(ab / xa)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, shark_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, shark_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, shark_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    is_shark = identify_shark_pattern(prices)

    if is_shark:
        pattern = "Shark"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Shark"
            string1="Action: Buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Shark"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings

def cal_identify_cypher_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_cypher_pattern(prices):
        # Check if the price data matches the Cypher pattern ratios
        xa = prices[1] - prices[0]
        ab = prices[2] - prices[1]
        bc = prices[3] - prices[2]
        cd = prices[4] - prices[3]

        if xa > 0 and ab < 0 and bc > 0 and cd < 0:
            ratio_ab = abs(ab / xa)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, cypher_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, cypher_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, cypher_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    is_cypher = identify_cypher_pattern(prices)
    if is_cypher:
        pattern = "Cypher"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Cypher"
            string1="Action: Sell"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Cypher"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings



def cal_identify_abcd_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_abcd_pattern(prices):
        # Check if the price data matches the ABCD pattern ratios
        ab = prices[1] - prices[0]
        bc = prices[2] - prices[1]
        cd = prices[3] - prices[2]

        if ab > 0 and bc < 0 and cd > 0:
            ratio_ab = abs(ab / bc)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)

            if (
                np.isclose(ratio_ab, abcd_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, abcd_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, abcd_ratios["CD"], atol=0.01)
            ):
                return True

        return False
    is_abcd = identify_abcd_pattern(prices)

    if is_abcd:
        pattern = "ABCD"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: ABCD"
            string1="Action: Buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: ABCD"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings



def cal_identify_three_drive_pattern(candles):
    prices=np.array([candle['close'] for candle in candles])
    def identify_three_drive_pattern(prices):
        # Check if the price data matches the Three-Drive pattern ratios
        ab = prices[1] - prices[0]
        bc = prices[2] - prices[1]
        cd = prices[3] - prices[2]
        de = prices[4] - prices[3]
        ef = prices[5] - prices[4]

        if (
            ab > 0 and bc < 0 and cd > 0 and de < 0 and ef > 0 and
            len(set([abs(bc / ab), abs(cd / bc), abs(de / cd), abs(ef / de)])) == 1
        ):
            ratio_ab = abs(ab / bc)
            ratio_bc = abs(bc / ab)
            ratio_cd = abs(cd / bc)
            ratio_de = abs(de / cd)
            ratio_ef = abs(ef / de)

            if (
                np.isclose(ratio_ab, three_drive_ratios["AB"], atol=0.01) and
                np.isclose(ratio_bc, three_drive_ratios["BC"], atol=0.01) and
                np.isclose(ratio_cd, three_drive_ratios["CD"], atol=0.01) and
                np.isclose(ratio_de, three_drive_ratios["DE"], atol=0.01) and
                np.isclose(ratio_ef, three_drive_ratios["EF"], atol=0.01)
            ):
                return True

        return False

    is_three_drive = identify_three_drive_pattern(prices)

    if is_three_drive:
        pattern = "Three-Drive"
        target = get_price_target(pattern, prices)
        if target is not None:
            stringsO="Pattern: Three-Drive"
            string1="Action: Sell"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Three-Drive"
            strings="Action: Pattern identified, but price target calculation failed."

    else:
        strings=None
        return strings



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
            string1="Action: Buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Head and Shoulders"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings




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
            string1="Action: Sell"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Double Tops"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
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
            string1="Action: Buy"
            strings=f"Price Target: ,  {target}"
            result=stringsO+"\n"+string1+"\n"+strings
            return result
        else:
            stringsO="Pattern: Double Bottoms"
            strings="Action: Pattern identified, but price target calculation failed."
            result=stringsO+"\n"+strings
            return result
    else:
        strings=None
        return strings


import numpy as np

def cal_channel_down(candles):
    # Define the price target percentage
    price_target_percent = 0.5  # 0.5% price target

    # Extract close prices from the price data
    close_prices = np.array([candle['close'] for candle in candles])

    # Identify channel down pattern
    channel_down_length = 10  # Number of candles to consider for channel down
    channel_down_prices = close_prices[-channel_down_length:]
    if channel_down_prices.min() >= close_prices[-1]:
        #print('Pattern: Channel Down')
        price_target = close_prices[-1] - (close_prices[-1] * price_target_percent / 100)
        string = f'Projected Price Target: {price_target}'
        result = "Sell according to channel Down" + "\n" + string
        return result
    else:
        strings=None
        return strings

def cal_channel_up(candles):
    # Define the price target percentage
    price_target_percent = 0.5  # 0.5% price target

    # Extract close prices from the price data
    close_prices = np.array([candle['close'] for candle in candles])

    # Identify channel up pattern
    channel_up_length = 10  # Number of candles to consider for channel up
    channel_up_prices = close_prices[-channel_up_length:]
    if channel_up_prices.max() <= close_prices[-1]:
        #print('Pattern: Channel Up')
        price_target = close_prices[-1] + (close_prices[-1] * price_target_percent / 100)
        string = f'Projected Price Target: {price_target}'
        result = "Buy according to channel up" + "\n" + string
        return result
    else:
        strings=None
        return strings
    

'''
timeframes=['1m','5m','15m','30m','1h','4h','1d']
for symbol in symbols:
    for timeframe in timeframes:
        print(f'{symbol} in {timeframe}')
        print('------------------------------------------------')
        candles=asyncio.run(get_candles(timeframe,symbol))
        print(cal_double_tops(candles))
        print(cal_identify_abcd_pattern(candles))
        print(cal_identify_bat_pattern(candles))
        print(cal_identify_butterfly_pattern(candles))
        print(cal_identify_crab_pattern(candles))
        print(cal_identify_cypher_pattern(candles))
        print(cal_identify_double_bottom(candles))
        print(cal_identify_gartley_pattern(candles))
        print(cal_identify_shark_pattern(candles))
        print(cal_identify_three_drive_pattern(candles))
        print('------------------------------------------------')
'''







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
        decision = 'Buy'
        meaning = 'Bullish Engulfing Pattern identified in an Up Trend'
    elif bearish_engulfing and trend == 'Down Trend':
        decision = 'Sell'
        meaning = 'Bearish Engulfing Pattern identified in a Down Trend'
    elif hammer and trend == 'Down Trend':
        decision = 'Buy'
        meaning = 'Hammer Pattern identified in a Down Trend'
    elif inverted_hammer and trend == 'Down Trend':
        decision = 'Buy'
        meaning = 'Inverted Hammer Pattern identified in a Down Trend'
    elif shooting_star and trend == 'Up Trend':
        decision = 'Sell'
        meaning = 'Shooting Star Pattern identified in an Up Trend'
    elif hanging_man and trend == 'Up Trend':
        decision = 'Sell'
        meaning = 'Hanging Man Pattern identified in an Up Trend'
    elif doji:
        decision = 'No Trade'
        meaning = 'Doji Pattern identified'
    elif evening_star and trend == 'Up Trend':
        decision = 'Sell'
        meaning = 'Evening Star Pattern identified in an Up Trend'
    elif harami and trend == 'Down Trend':
        decision = 'Buy'
        meaning = 'Harami Pattern identified in a Down Trend'
    else:
        decision = 'No Trade'
        meaning = 'No recognizable pattern or conflicting signals'
    
    # Concatenate decision and meaning into results
    results = f"{decision}: {meaning}"
    return results


def analyze_ascending_triangle(candles):
    prices=np.array([candle['close'] for candle in candles])
    length=10
    # Identify ascending triangle pattern
    price_target_percent=5
    ascending_triangle_pattern = np.all(np.diff(prices[-length:]) > 0)
    
    if ascending_triangle_pattern:
        # Determine whether to buy or sell based on the identified ascending triangle pattern
        price_target = prices[-1] + (prices[-1] * price_target_percent / 100)
        return 'Buy Order. Pattern: Ascending Triangle. Projected Price Target: {:.2f}'.format(price_target)
    else:
        return 'No Ascending Triangle pattern identified.'

def analyze_descending_triangle(candles):
    prices=np.array([candle['close'] for candle in candles])
    length=10
    # Identify descending triangle pattern
    price_target_percent=5
    descending_triangle_pattern = np.all(np.diff(prices[-length:]) < 0)
    
    if descending_triangle_pattern:
        # Determine whether to buy or sell based on the identified descending triangle pattern
        price_target = prices[-1] - (prices[-1] * price_target_percent / 100)
        return 'Sell Order. Pattern: Descending Triangle. Projected Price Target: {:.2f}'.format(price_target)
    else:
        return 'No Descending Triangle pattern identified.'

def flag_pattern(candles):
    close_prices=np.array([candle['close'] for candle in candles])
    price_target_percent=5
    flag_length=10
    # Identify flag pattern
    flag_pattern = False

    # Check if the length of close_prices is sufficient
    if len(close_prices) >= flag_length + 1:
        # Check if the last flag_length elements have increasing prices
        increasing_prices = np.all(np.diff(close_prices[-flag_length:]) > 0)
        
        # Check if the previous flag_length elements have decreasing prices
        decreasing_prices = np.all(np.diff(close_prices[-flag_length:-1]) < 0)
        
        # Check if the last price is higher than the previous flag_length price
        last_price_higher = close_prices[-1] > close_prices[-flag_length-1]
        
        # Combine the conditions to identify the flag pattern
        flag_pattern = increasing_prices and decreasing_prices and last_price_higher
    
    if flag_pattern:
        # Determine whether to buy or sell based on the identified flag pattern
        price_target = close_prices[-1] + (close_prices[-1] * price_target_percent / 100)
        return 'Buy Order. Pattern: Flag. Projected Price Target: {:.2f}'.format(price_target)
    else:
        return 'No Flag pattern identified.'




def triple_top(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Define pattern recognition parameter
    triple_top_length = 10
    
    triple_top_pattern = np.all(prices[-triple_top_length:] <= prices[-triple_top_length:].max()) and \
                        np.all(prices[-triple_top_length:] >= prices[-triple_top_length:].min())
    # Determine whether to buy or sell based on the identified patterns
    if triple_top_pattern:
        if prices[-1] < prices[-2]:
            return 'Action: Sell'
        else:
            return 'Action: Hold'
    else:
        return None

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
            return 'Action: Buy'
        else:
            return 'Action: Hold'
    else:
        return None
    

    




def rectangle(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Rectangle pattern
    rectangle_pattern = np.all(prices[-rectangle_length:] <= prices[-rectangle_length:].max()) and \
                        np.all(prices[-rectangle_length:] >= prices[-rectangle_length:].min())

    # Determine whether to buy or sell based on the identified patterns
    if rectangle_pattern:
        if prices[-1] > prices[-rectangle_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-rectangle_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return None
def  pennant(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Pennant pattern
    pennant_pattern = np.all(np.diff(prices[-pennant_length:]) > 0) and np.all(np.diff(prices[-pennant_length+1:]) < 0)
    if pennant_pattern:
        if prices[-1] > prices[-pennant_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-pennant_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return None
def symmetrical_triangle(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Symmetrical Triangle pattern
    symmetrical_triangle_pattern = np.all(np.diff(prices[-symmetrical_triangle_length:]) > 0) and \
                                np.all(np.diff(prices[-symmetrical_triangle_length+1:]) < 0)
    if symmetrical_triangle_pattern:
        if prices[-1] > prices[-symmetrical_triangle_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-symmetrical_triangle_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return None
def wedge(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Wedge pattern (rising wedge and falling wedge)
    wedge_pattern = np.all(np.diff(prices[-wedge_length:]) > 0) or np.all(np.diff(prices[-wedge_length:]) < 0)

    if wedge_pattern:
        if prices[-1] > prices[-wedge_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-wedge_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return None
def diamond(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Diamond pattern
    diamond_pattern = np.all(prices[-diamond_length:] <= prices[-diamond_length:].max()) and \
                    np.all(prices[-diamond_length:] >= prices[-diamond_length:].min())

    if diamond_pattern:
        if prices[-1] > prices[-diamond_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-diamond_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return None
def cup_and_handle(candles):
    prices=np.array([candle['close'] for candle in candles])
    # Identify Cup and Handle pattern
    cup_and_handle_pattern = np.all(prices[-cup_and_handle_length:] <= prices[-cup_and_handle_length:].max()) and \
                            np.all(prices[-cup_and_handle_length:] >= prices[-cup_and_handle_length:].min())
    if cup_and_handle_pattern:
        if prices[-1] > prices[-cup_and_handle_length:].max():
            return 'Buy Decision: Enter long position'
        elif prices[-1] < prices[-cup_and_handle_length:].min():
            return 'Sell Decision: Enter short position'
        else:
            return None

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
            return None
