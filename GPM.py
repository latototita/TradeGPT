import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import VotingRegressor
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
async def main():

    # Load and preprocess data
    X_train, y_train, X_test, y_test =await load_data()
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    learning_rate = 0.001

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    learning_rate = 0.001
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = []

    # Create individual Gaussian Process models
    predictions = []

    for i in range(output_dim):
        kernel = RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5)
        model.fit(X_train_scaled, y_train[:, i])
        models.append(('GP' + str(i+1), model))

        # Predict on test data for each target variable
        target_predictions = model.predict(X_test_scaled)
        predictions.append(target_predictions)

    # Concatenate the predictions along the last axis
    predictions = np.stack(predictions, axis=-1)

    # Print each candle in the prediction
    for i in range(len(predictions)):
        candle = predictions[i]
        print(f"Candle {i+1}: Open={candle[0]}, High={candle[1]}, Low={candle[2]}, Close={candle[3]}, Volume={candle[4]}")

    # Calculate the percentage accuracy and the difference for each prediction
    percentage_accuracies = []
    differences = []
    for i in range(predictions.shape[0]):
        prediction = predictions[i]
        real_value = y_test[i]
        percentage_accuracy = 100 * np.mean(np.isclose(prediction, real_value, rtol=1e-3))
        difference = np.abs(prediction - real_value)
        percentage_accuracies.append(percentage_accuracy)
        differences.append(difference)

    # Calculate the overall percentage accuracy of each model
    model_percentage_accuracies = {}
    for i, (model_name, _) in enumerate(models):
        model_predictions = predictions[:, i]
        model_real_values = y_test[:, i]
        model_percentage_accuracy = 100 * np.mean(np.isclose(model_predictions, model_real_values, rtol=1e-3))
        model_percentage_accuracies[model_name] = model_percentage_accuracy

    # Display the results
    print("Percentage Accuracy and Differences for Each Prediction:")
    for i in range(len(percentage_accuracies)):
        print(f"Prediction {i+1}: Percentage Accuracy = {percentage_accuracies[i]:.2f}%, Difference = {differences[i]}")

    print("\nOverall Percentage Accuracy of Each Model:")
    for model_name, model_percentage_accuracy in model_percentage_accuracies.items():
        print(f"{model_name}: Percentage Accuracy = {model_percentage_accuracy:.2f}%")


asyncio.run(main())