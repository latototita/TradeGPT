from ast import Delete
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
label_encoder = LabelEncoder()

def knn_train(df,symbol):
    # Step 1: Data Preparation
    # Assuming you have a CSV file named 'data.csv' with columns 'feature1', 'feature2', 'trend', 'time', 'pips', 'duration'

    # Load the data from CSV file
    data = df

    # Extract features and target variables
    timestamps = data['Start_time'].values
    y_trend = data['Trend'].values
    times= data['End_time'].values
    y_pips = data['Pips'].values
    y_duration = data['Duration'].values
    y_end_value= data['End_Value'].values
    y_start_value = data['Start_Value'].values
    datetime_objects = [datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z") for timestamp in timestamps]
    #datetime_objects = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in timestamps]
    #datetime_objects = [np.datetime64(timestamp).astype(datetime) for timestamp in timestamps]
    # Extract relevant features from datetime objects
    years = [dt.year for dt in datetime_objects]
    months = [dt.month for dt in datetime_objects]
    days = [dt.day for dt in datetime_objects]
    hours = [dt.hour for dt in datetime_objects]
    minutes = [dt.minute for dt in datetime_objects]
    seconds = [dt.second for dt in datetime_objects]

    # Assign extracted features to X for training and testing
    X = np.column_stack((years, months, days, hours, minutes, seconds))

    datetime_objects = [datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z") for timestamp in times]
    #datetime_objects = [np.datetime64(timestamp).astype(datetime) for timestamp in times]
    #datetime_objects = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in times]
    data['symbol'] = [f'{symbol}'] * len(data['End_time'])
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
    X =    scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train_trend, y_test_trend, y_train_time, y_test_time, y_train_pips, y_test_pips, y_train_duration, y_test_duration, y_train_symbol, y_test_symbol,y_train_start_value,y_train_end_value,y_test_start_value,y_test_end_value = train_test_split(
        X, y_trend, y_time, y_pips, y_duration, y_symbol,y_start_value,y_end_value ,test_size=0.2, random_state=42)

    # Step 2: Model Training
    # Instantiate the KNN classifier and regressor
    knn_trend2 = KNeighborsClassifier(n_neighbors=1)
    knn_time2 = KNeighborsClassifier(n_neighbors=1)
    knn_pips2 = KNeighborsRegressor(n_neighbors=1)
    knn_duration2 = KNeighborsRegressor(n_neighbors=1)
    knn_symbol2 = KNeighborsRegressor(n_neighbors=1)
    knn_start_value2 = KNeighborsRegressor(n_neighbors=1)
    knn_end_value2 = KNeighborsRegressor(n_neighbors=1)

    # Train the models
    knn_trend2.fit(X_train, y_train_trend)
    knn_time2.fit(X_train, y_train_time)
    knn_pips2.fit(X_train, y_train_pips)
    knn_duration2.fit(X_train, y_train_duration)
    knn_symbol2.fit(X_train, y_train_symbol)
    knn_start_value2.fit(X_train,y_train_start_value)
    knn_end_value2.fit(X_train, y_train_end_value)
    # Test the models
    predictions_trend = knn_trend2.predict(X_test)
    predictions_time = knn_time2.predict(X_test)
    predictions_pips = knn_pips2.predict(X_test)
    predictions_duration = knn_duration2.predict(X_test)
    predictions_symbol =knn_symbol2.predict(X_test)
    predictions_start_value = knn_start_value2.predict(X_test)
    predictions_end_value =knn_end_value2.predict(X_test)
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
    mse_start_value = mean_squared_error(y_test_start_value,predictions_start_value )
    mse_end_value = mean_squared_error(y_test_end_value, predictions_end_value)
    mse_symbol = mean_squared_error(y_test_symbol, predictions_symbols)

    print("Accuracy (Trend): {:.2f}%".format(accuracy_trend * 100))
    print("MSE (Time): {:.2f}".format(mse_time))

    print("Mean Accuracy (Time): {:.2f}%".format(mean_accuracy * 100))

    print("MSE (Duration): {:.2f}".format(mse_duration))
    print("MSE (Symbol): {:.2f}".format(mse_symbol))
    print("MSE (Start Value): {:.2f}".format(mse_start_value))
    print("MSE (End_value): {:.2f}".format(mse_end_value))
    #return knn_trend2, knn_time2, knn_pips2, knn_duration2,knn_symbol2,knn_start_value2,knn_start_value2


def extract(candles):
    import numpy as np
    import pandas as pd
    import pandas_ta as ta
    from datetime import datetime, timedelta
    # Define parameters
    rsi_threshold = 70  # RSI threshold for trend confirmation
    volatility_factor = 1.5  # Adjust the trend duration based on volatility
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    date = df['time']
    open_price = df['open']
    high_price = df['high']
    low_price = df['low']
    close_price = df['close']
    n_fast = 3
    n_slow = 6
    n_signal = 4
    n_fast = 3
    n_slow = 6
    n_signal = 4

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
        return smi

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
        return adx
    df['ADX'] =calculate_adx(candles)
    # Calculate the relative strength index (RSI)
    def calculate_rsi(df, period):
        delta = df.diff()
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

        return macd_line, signal_line
    df['SMA'] = ta.sma(df['close'], length=14)
    # Calculate RSI as a confirmation indicator
    rsi_period = 14  # RSI period
    df['RSI'] = calculate_rsi(close_price, rsi_period)

    # Calculate the average true range (ATR) as a measure of volatility
    atr_period = 14  # ATR period
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    atr = df['ATR'].to_numpy()

    # Exponential Moving Average (EMA)
    def exponential_moving_average(df, period=5):
        df.set_index('time', inplace=True)
        df['close'] = df['close'].astype(float)
        data = df['close']
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode='full')[:len(data)]
        return ema

    df['EMA'] = exponential_moving_average(df)

    # Function to calculate pips covered
    def calculate_pips(entry_price, exit_price):
        pips = abs(exit_price - entry_price)
        return pips

    # Parameters
    min_trend_duration = 5 # Minimum trend duration in minutes
    max_trend_duration =90  # Maximum trend duration in minutes

    # Set ATR threshold
    current_atr_threshold = 0.0  # Set your desired ATR threshold value

    # Set minimum pip threshold
    pip_threshold = 3

    # Initialize variables
    trend_start_index = 0
    trends = []
    current_trend = None
    adx_values= calculate_adx(candles)
    smi=calculate_smi(candles)
    for i in range(len(df)):
        close_price = df['close'][i]
        sma = df['SMA'][i]
        ema = df['EMA'][i]
        rsi = df['RSI'][i]
        atr = df['ATR'][i]

        # Check if trend has ended
        if current_trend is not None and (i - current_trend['start_index']) >= max_trend_duration:
            current_trend['end_index'] = i
            current_trend['end_price'] = close_price
            current_trend['pips'] = calculate_pips(current_trend['start_price'], current_trend['end_price'])
            
            # Check if pips covered meets the threshold
            if current_trend['pips'] >= pip_threshold:
                trends.append(current_trend)

            current_trend = None

        # Check if trend has started
        if current_trend is None:
            if close_price > sma and close_price > ema and rsi > rsi_threshold and atr > current_atr_threshold and  adx_values[i] > 26 and  smi[i] > 0 :
                current_trend = {
                    'start_index': i,
                    'start_price': close_price,
                    'end_index': None,
                    'end_price': None,
                    'pips': None,
                    'direction': 'Up'
                }
            elif close_price < sma and close_price < ema and rsi< (100-rsi_threshold)  and atr > current_atr_threshold and adx_values[i] > 26 and smi[i] < 0:
                current_trend = {
                    'start_index': i,
                    'start_price': close_price,
                    'end_index': None,
                    'end_price': None,
                    'pips': None,
                    'direction': 'Down'
                }

    # Function to check if ADX is greater than a threshold
    def is_adx_greater_than_threshold(df, index, threshold):
        adx = df['ADX'][index]
        return adx > 30
    count=0
    df = pd.DataFrame(columns=['Trend','Start_time','End_time', 'Duration', 'Pips','Start_Value','End_Value'])

    # Print the trends and pips covered
    for trend in trends:
        count+=1
        start_index = trend['start_index']
        start_time = date[start_index]
        end_index = trend['end_index']
        end_time=date[end_index] if end_index is not None else None
        # Convert start and end times to datetime objects
        #start_time = pd.to_datetime(date[start_index])
        #end_time = pd.to_datetime(date[end_index])
        #duration = end_time - start_time
        duration = end_index - start_index + 1
        #if all(is_adx_greater_than_threshold(df, index, 25) for index in range(start_index, end_index+1)):
        #print(f"Start: {trend['start_index']}, End: {trend['end_index']}, Direction: {trend['direction']}, Pips: {trend['pips']}, Start time: {start_time}, End time: {end_time}, Duration: {duration}")
        #print(count)
        df = df.append({'Start_Value':trend['start_index'],'End_Value':trend['end_index'],'Trend': trend['direction'], 'Start_time': start_time,'End_time': end_time,  'Duration': duration, 'Pips': trend['pips']}, ignore_index=True)
        print(len(df))
        return df

from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta
import asyncio,os

token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '615ae0df-2198-4162-9b23-34a4285baa35'
timeframe='1m'
symbol='XAUUSDm'
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
        pages = 9
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
        print()
        return f"Error retrieving candle data: {e}"
candles=asyncio.run(get_candles_m(timeframe,symbol))
df=extract(candles)
knn_train(df=df,symbol='EURUSDm')