strings = ['AUDMXNm', 'AUDNOKm', 'AUDNZDm', 'AUDPLNm',
           'AUDSEKm', 'AUDSGDm', 'AUDTRYm', 'AUDUSDm', 'AUDUSX']

trimmed_strings = []

for string in strings:
    if string.endswith('m'):
        trimmed_string = string[:-1]
    else:
        trimmed_string = string
    print(trimmed_string)

print(trimmed_strings)


'''
#import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
#import talib as ta
from metaapi_cloud_sdk import MetaApi
import os , asyncio
from datetime import datetime, timedelta

token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '615ae0df-2198-4162-9b23-34a4285baa35'

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
        pages = 10
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

# Define the ticker symbol of the stock you want to analyze
ticker_symbol = "AAPL"  # Replace with your desired stock symbol

data=asyncio.run(get_candles_m('1m','XAUUSDm'))#pd.read_csv('/home/omenyo/Desktop/Artificial_intelligence_projects/trading bot/web version/candlestick_data1.csv')
# Convert the 'data' variable to a pandas DataFrame if it's not already
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# Calculate additional technical indicators
data['SMA'] = data['close'].rolling(window=20).mean()  # Simple Moving Average
data['RSI'] = 100 - (100 / (1 + (data['close'].diff(1).clip(lower=0) / data['close'].diff(1).clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()))  # Relative Strength Index
# Adjust the target variables (open, high, low, close, volume of the next 10 candles)


# Adjust the target variables (open, high, low, close, volume of the next candle)
data['target_open'] = data['open'].shift(-1)
data['target_high'] = data['high'].shift(-1)
data['target_low'] = data['low'].shift(-1)
data['target_close'] = data['close'].shift(-1)
data['target_volume'] = data['tickVolume'].shift(-1)

# Drop NaN values from the dataset
data.dropna(inplace=True)

# Define features and targets
features = ['open', 'high', 'low', 'close', 'tickVolume', 'SMA', 'RSI',]
targets = ['target_open', 'target_high', 'target_low', 'target_close', 'target_volume']
X = data[features]
y = data[targets]


X = data[features]
y = data[targets]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features)
    ])

# Create an instance of the regression model
model = MLPRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'model__hidden_layer_sizes': [(100,), (100, 50), (100, 100, 50),(1000,),(200,10,200),800,500,10],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001, 0.01],
    'model__max_iter': [200, 500, 1000],
    'model__learning_rate': ['constant', 'adaptive'],
}

# Create a pipeline for preprocessing and modeling
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Perform random search for hyperparameter tuning
search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=10, cv=5,
                            scoring='neg_mean_squared_error', random_state=42)
search.fit(X_train, y_train)

# Make predictions on the test set
y_pred = search.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
'''
