import pandas as pd
import pandas_ta as ta
import numpy as np
import theano
import theano.tensor as T


# Set Theano flags before importing Theano
theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'

class RNNModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W_in = theano.shared(
            np.random.uniform(-0.01, 0.01, (input_dim, hidden_dim)).astype(theano.config.floatX)
        )
        self.W_hidden = theano.shared(
            np.random.uniform(-0.01, 0.01, (hidden_dim, hidden_dim)).astype(theano.config.floatX)
        )
        self.W_out = theano.shared(
            np.random.uniform(-0.01, 0.01, (hidden_dim, output_dim)).astype(theano.config.floatX)
        )
        self.b_hidden = theano.shared(np.zeros(hidden_dim, dtype=theano.config.floatX))
        self.b_out = theano.shared(np.zeros(output_dim, dtype=theano.config.floatX))

        self.params = [self.W_in, self.W_hidden, self.W_out, self.b_hidden, self.b_out]

        self.inputs = T.tensor3()
        self.targets = T.tensor3()
        self.hidden_state = T.vector()

        self.update_hidden = T.tanh(T.dot(self.inputs, self.W_in) + T.dot(self.hidden_state, self.W_hidden) + self.b_hidden)
        self.output = T.dot(self.update_hidden, self.W_out) + self.b_out

        self.loss = T.mean((self.output - self.targets) ** 2)
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

    # Split into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(inputs))

    X_train = inputs[:split_index]
    y_train = targets[:split_index]
    X_test = inputs[split_index:]
    y_test = targets[split_index:]

    return X_train, y_train, X_test, y_test
def train_model(inputs, targets, model):
    inputs = inputs.reshape(-1, inputs.shape[2])
    targets = targets.reshape(-1, targets.shape[2])
    learning_rate = 0.001
    gradients = [T.grad(model.loss, param) for param in model.params]
    updates = [(param, param - learning_rate * gradient) for param, gradient in zip(model.params, gradients)]
    train_fn = theano.function(inputs=[model.inputs, model.targets, model.hidden_state], outputs=model.loss,
                               updates=updates)
    return train_fn

def predict_model(inputs, model):
    inputs = inputs.reshape(-1, inputs.shape[2])  # Reshape the inputs for prediction
    predict_fn = theano.function(inputs=[model.inputs, model.hidden_state], outputs=model.output)
    return predict_fn
def main():

    # Load and preprocess data
    X_train, y_train, X_test, y_test =asyncio.run(load_data())
    print(X_train.shape)
    print(y_train.shape)
    
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[2]
    hidden_dim = 100
    learning_rate = 0.001

    # Initialize the hidden state
    hidden_state = np.zeros(hidden_dim, dtype=theano.config.floatX)

    # Reshape the data
    X_train = X_train.reshape(-1, 5, input_dim)
    y_train = y_train.reshape(-1, 1, output_dim)
    X_test = X_test.reshape(-1, 5, input_dim)

    model = RNNModel(input_dim, hidden_dim, output_dim)

    num_epochs = 100
    batch_size = 32
    train_fn = train_model(X_train, y_train, model)


    for epoch in range(num_epochs):
        for batch in range(0, len(X_train), batch_size):
            inputs = X_train[batch:batch + batch_size]
            targets = y_train[batch:batch + batch_size]
            loss = train_fn(inputs, targets, hidden_state)

        # Print the loss after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    predictions = model.predict(X_test, hidden_state)

    # Print each candle in the prediction
    for i in range(len(predictions)):
        candle = predictions[i][0]
        print(f"Candle {i+1}: Open={candle[0]}, High={candle[1]}, Low={candle[2]}, Close={candle[3]}, Volume={candle[4]}")

main()