import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne
import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd
from indicator_collection import *



token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '86734b47-6b5f-45c7-91e6-c72243b6a293'


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
    X, y =await load_data()

    input_dim = X.shape[1]
    output_dim = y.shape[1]
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32
    hidden_units = 100

    # Reshape the data
    X = X.reshape(-1, 1, input_dim)

    # Define Theano variables
    input_var = T.tensor3('inputs')
    target_var = T.tensor3('targets')

    # Define the GRU architecture
    l_input = lasagne.layers.InputLayer(shape=(None, None, input_dim), input_var=input_var)
    l_gru = lasagne.layers.GRULayer(l_input, num_units=hidden_units)
    l_reshape = lasagne.layers.ReshapeLayer(l_gru, (-1, hidden_units))
    l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=output_dim, nonlinearity=None)
    l_output = lasagne.layers.ReshapeLayer(l_dense, (-1, 1, output_dim))

    # Define the network output and loss function
    network_output = lasagne.layers.get_output(l_output)
    loss = lasagne.objectives.squared_error(network_output, target_var).mean()

    # Get all network parameters (weights and biases)
    params = lasagne.layers.get_all_params(l_output, trainable=True)

    # Define the updates
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)

    # Compile the training function
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Train the GRU network
    for epoch in range(num_epochs):
        for batch in range(0, len(X), batch_size):
            inputs = X[batch:batch + batch_size]
            targets = y[batch:batch + batch_size]
            loss_value = train_fn(inputs, targets)

        # Print the loss after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.4f}")

    # Compile the prediction function
    predict_fn = theano.function([input_var], network_output)

    # Predict on the input data
    predictions = predict_fn(X)

    # Print each candle in the prediction
    for i in range(len(predictions)):
        candle = predictions[i]
        print(f"Candle {i+1}: Open={candle[0, 0]}, High={candle[0, 1]}, Low={candle[0, 2]}, Close={candle[0, 3]}, Volume={candle[0, 4]}")
asyncio.run(main())