import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import asyncio,os
from metaapi_cloud_sdk import MetaApi
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '615ae0df-2198-4162-9b23-34a4285baa35'

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
def predictupdown(candles):
    data=candles
    # Convert the 'data' variable to a pandas DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    


    # Calculate the target variable (1 if prices increase, -1 if prices decrease)
    data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)

    # Drop NaN values from the dataset
    data.dropna(inplace=True)

    # Select the features and target variable
    X_close = data[['close', 'target']]
    y_close= data['target']

    # Split the data into train_closeing and test_close sets
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_close_scaled = scaler.fit_transform(X_train_close)
    X_test_close_scaled = scaler.transform(X_test_close)

    # train_close the MLP classifier
    clf_close = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    clf_close.fit(X_train_close_scaled, y_train_close)

    # Make predictions on the test_close set
    y_pred_close = clf_close.predict(X_test_close_scaled)
    predictions_text = ['up' if pred == 1 else 'down' for pred in y_pred_close]
    if y_pred_close[-1]==1:
        print("Next candle will close higher than the close of previous candle")
    else:
        print("Next candle will close lower than the  close of previous candle")
















    data=candles
    # Convert the 'data' variable to a pandas DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # Calculate the target variable (1 if prices increase, -1 if prices decrease)
    data['target'] = np.where(data['high'].shift(-1) > data['high'], 1, -1)

    # Drop NaN values from the dataset
    data.dropna(inplace=True)

    # Select the features and target variable
    X_high = data[['high', 'target']]
    y_high= data['target']

    # Split the data into train_highing and test_high sets
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_high_scaled = scaler.fit_transform(X_train_high)
    X_test_high_scaled = scaler.transform(X_test_high)

    # train_high the MLP classifier
    clf_high = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    clf_high.fit(X_train_high_scaled, y_train_high)

    # Make predictions on the test_high set
    y_pred_high = clf_high.predict(X_test_high_scaled)
    predictions_text = ['up' if pred == 1 else 'down' for pred in y_pred_high]
    if y_pred_high[-1]==1:
        print("Next candle will high higher than the high of previous candle")
    else:
        print("Next candle will high lower than the  high of previous candle")























    data=candles
    # Convert the 'data' variable to a pandas DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # Calculate the target variable (1 if prices increase, -1 if prices decrease)
    data['target'] = np.where(data['low'].shift(-1) > data['low'], 1, -1)

    # Drop NaN values from the dataset
    data.dropna(inplace=True)

    # Select the features and target variable
    X_low = data[['low', 'target']]
    y_low= data['target']

    # Split the data into train_lowing and test_low sets
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_low_scaled = scaler.fit_transform(X_train_low)
    X_test_low_scaled = scaler.transform(X_test_low)

    # train_low the MLP classifier
    clf_low = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    clf_low.fit(X_train_low_scaled, y_train_low)

    # Make predictions on the test_low set
    y_pred_low = clf_low.predict(X_test_low_scaled)
    predictions_text = ['up' if pred == 1 else 'down' for pred in y_pred_low]
    if y_pred_low[-1]==1:
        print("Next candle will low higher than the low of previous candle")
    else:
        print("Next candle will low lower than the  low of previous candle")
    data = candles

    # Convert the 'data' variable to a pandas DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Calculate the target variable (1 if next candle closes lower, 0 otherwise)
    data['target'] = np.where(data['close'].shift(-1) < data['open'].shift(-1), 1, 0)

    # Drop NaN values from the dataset
    data.dropna(inplace=True)

    # Select the features and target variable
    X = data[['close', 'open']]
    y = data['target']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test_scaled)

    if y_pred[-1] == 1:
        print("Next candle will close lower than the open of the previous candle")
    else:
        print("Next candle will not close lower than the open of the previous candle")
candles=asyncio.run(get_candles('30m','XAUUSDm'))
predictupdown(candles)