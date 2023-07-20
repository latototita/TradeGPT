import pandas as pd
import numpy as np
from hmmlearn import hmm
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
    print(candles)
    data = pd.DataFrame(candles)
    data.set_index('time', inplace=True)
    data['close'] = data['close'].astype(float)
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    # Apply ATR indicator
    df.dropna(inplace=True)
    
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    window_size = 10  # You can adjust the window size as per your preference

    # Create sequences of observations using a sliding window approach
    sequences = []
    for i in range(len(data) - window_size):
        sequence = data.iloc[i: i + window_size].to_numpy().flatten()
        sequences.append(sequence)

    sequences = np.array(sequences)

    # Extract inputs and targets
    inputs = sequences
    targets = sequences

    return inputs, targets

async def main():
    # Load and preprocess data
    X, y = await load_data()

    input_dim = X.shape[1]
    output_dim = y.shape[1]
    num_states = 3  # Number of hidden states
    num_iterations = 100  # Number of Baum-Welch iterations

    # Reshape the data to 2D array (num_samples, num_features)
    X = df.to_numpy().reshape(-1, input_dim)

    # Create an HMM model
    model = hmm.GaussianHMM(n_components=num_states, n_iter=num_iterations)

    # Fit the model to the data (train the HMM)
    model.fit(X)

    # Generate predictions for the next 20 candles based on the trained model
    num_predictions = 20
    start_state = model.startprob_.argmax()  # Choose the state with the highest initial probability

    # Simulate the HMM to generate new samples
    generated_samples, _ = model.sample(num_predictions, random_state=start_state)

    # Print each generated sample
    for i in range(len(generated_samples)):
        sample = generated_samples[i]
        # Reshape the sample back to (window_size, input_dim) shape
        sample = sample.reshape(-1, input_dim)
        for j in range(len(sample)):
            open_price, high_price, low_price, close_price, volume = sample[j]
            print(
                f"Sample {i+1}, Candle {j+1}: Open={open_price}, High={high_price}, Low={low_price}, Close={close_price}, Volume={volume}"
            )

# Run the main function
asyncio.run(main())