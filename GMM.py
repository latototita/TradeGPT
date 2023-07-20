import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd
from indicator_collection import *



token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '12efd59b-75c9-43c0-8600-c6e0dbb1bb38'


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
async def load_data():
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    # Load your data using pandas or any other preferred method
    candles =  await account.get_historical_candles(symbol='EURUSD', timeframe='1m', start_time=None, limit=1000)
    data = pd.DataFrame(candles)
    data.set_index('time', inplace=True)
    data['close'] = data['close'].astype(float)
    # Preprocess and clean the data as needed
    # ...

    # Extract the necessary columns for input and output
    inputs = data[['open', 'high', 'low', 'close', 'tickVolume']]
    targets = data[['open', 'high', 'low', 'close', 'tickVolume']]

    # Normalize the inputs and targets if required
    # ...

    # Convert data to numpy arrays
    inputs = inputs.to_numpy()
    targets = targets.to_numpy()

    return inputs, targets
async def main():
    # Load and preprocess data
    X, y = await load_data()

    # Extract the column for closing prices (choose 'close' or another appropriate column)
    close_prices = X[:, 3]  # Assuming 'close' is the 4th column

    # Create an ARIMA model
    p, d, q = 1, 1, 1  # Replace these values with appropriate ARIMA parameters
    model = ARIMA(close_prices, order=(p, d, q))

    # Fit the model to the data
    results = model.fit()

    # Get the last observation from the data (you can change this accordingly)
    last_observation = close_prices[-1]

    # Predict the next 20 candles
    num_predictions = 20
    forecast_values = results.forecast(steps=num_predictions)

    # Print each forecasted sample
    for i, sample in enumerate(forecast_values):
        print(f"Sample {i+1}: Close={sample}")

asyncio.run(main())