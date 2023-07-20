import pandas as pd
import numpy as np
import asyncio
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from metaapi_cloud_sdk import MetaApi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '12efd59b-75c9-43c0-8600-c6e0dbb1bb38'

#'XAUUSD','XAGUSD','EURJPY','USDJPY','EURUSD',
symbol_list = ['AUDUSD','USDCHF','USDCAD','NZDUSD','GBPCHF','EURCHF','EURGBP','EURAUD','CHFJPY','AUDJPY','AUDNZD','GBPJPY','GBPUSD',]
async def get_candles(timeframe, symbol):
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']
    timeframe = timeframe
    symbol = symbol
    if initial_state not in deployed_states:
        # wait until account is deployed and connected to the broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to the broker (may take a few minutes)')
        await account.wait_connected()

    try:
        # Fetch historical price df
        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
        return candles
    except Exception as e:
        return f"Error retrieving candle df: {e}"







def run_trading_bot(timeframe):
    async def main():
        # Connect to the MetaTrader account

        api = MetaApi(token)
        account = await api.metatrader_account_api.get_account(accountId)
        initial_state = account.state
        deployed_states = ['DEPLOYING', 'DEPLOYED']
        if initial_state not in deployed_states:
            # Wait until the account is deployed and connected to the broker
            print('Deploying account')
            await account.deploy()
        print('Waiting for API server to connect to the broker (may take a few minutes)')
        await account.wait_connected()

        # Connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()

        # Wait until terminal state synchronized to the local state
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()
        # Check for open trades
        trades = await connection.get_positions()#connection.get_orders()
        if len(trades)>50:
            print("There are open trades. Skipping analysis.")
            await asyncio.sleep(1200)
        else:
            for symbol in symbol_list:
                try:
                    # Fetch historical price df
                    candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                except Exception as e:
                    print(f"Error retrieving candle df: {e}")
                    candles=None
                
                df = pd.DataFrame(candles)
                df.set_index('time', inplace=True)
                df['close'] = df['close'].astype(float)
                df['Target_Open'] = df['open'].shift(-20)
                df['Target_High'] = df['high'].shift(-20)
                df['Target_Low'] = df['low'].shift(-20)
                df['Target_Close'] = df['close'].shift(-20)

                # Drop rows with NaN values in the target variables
                df.dropna(subset=['Target_Open', 'Target_High', 'Target_Low', 'Target_Close'], inplace=True)

                # Define the features and target variables
                X = df.drop(columns=['Target_Open', 'Target_High', 'Target_Low', 'Target_Close'])
                y_open = df['Target_Open']
                y_high = df['Target_High']
                y_low = df['Target_Low']
                y_close = df['Target_Close']

                # Split the data into training and test sets
                # Use the last 20 rows as the test dataset for the unseen candles
                X_train, X_test = X.iloc[:-20], X.iloc[-20:]
                y_train_open, y_test_open = y_open.iloc[:-20], y_open.iloc[-20:]
                y_train_high, y_test_high = y_high.iloc[:-20], y_high.iloc[-20:]
                y_train_low, y_test_low = y_low.iloc[:-20], y_low.iloc[-20:]
                y_train_close, y_test_close = y_close.iloc[:-20], y_close.iloc[-20:]

                # Define the ML Regressor models with the best hyperparameters
                regressor_open = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=5, min_samples_leaf=1, random_state=42)
                regressor_high = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=1, random_state=42)
                regressor_low = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)
                regressor_close = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)

                # Create pipelines for preprocessing and model training
                pipeline_open = Pipeline([('scaler', StandardScaler()), ('regressor', regressor_open)])
                pipeline_high = Pipeline([('scaler', StandardScaler()), ('regressor', regressor_high)])
                pipeline_low = Pipeline([('scaler', StandardScaler()), ('regressor', regressor_low)])
                pipeline_close = Pipeline([('scaler', StandardScaler()), ('regressor', regressor_close)])

                # Fit the models on the training dataset
                pipeline_open.fit(X_train, y_train_open)
                pipeline_high.fit(X_train, y_train_high)
                pipeline_low.fit(X_train, y_train_low)
                pipeline_close.fit(X_train, y_train_close)

                # Use the test dataset (next 20 unseen candles) to make predictions
                predicted_open = pipeline_open.predict(X)
                predicted_high=pipeline_high.predict(X)
                predicted_low = pipeline_low.predict(X)
                predicted_close = pipeline_close.predict(X)


                '''
                # Create a DataFrame to store the predictions with corresponding titles
                predictions_df = pd.DataFrame({'open': predicted_open,
                                            'high': predicted_high,
                                            'low': predicted_low,
                                            'close': predicted_close})

                # Save the predictions to a CSV file
                print(predictions_df)
                '''
                #predictions_df.to_csv('predictions.csv', index=False)

                
                
                print(symbol)
                trades = await connection.get_positions()
                if len(trades)>50:
                    print("There are open trades. Skipping analysis.")
                    await asyncio.sleep(1200)
                    continue
                prices = await connection.get_symbol_price(symbol)
                current_price = prices['ask']
                take_profit = predicted_close[-1]
                if take_profit>current_price: #  and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):
                    stop_loss = 0# current_price -((80/100)*current_price)                    
                    try:
                        
                        result = await connection.create_market_buy_order(
                            symbol,
                            0.01,
                            stop_loss,
                            take_profit,
                            {'trailingStopLoss': {
                                'distance': {
                                    'distance': 5,
                                    'units': 'RELATIVE_PIPS'
                                }
                            }
                        })
                        print('Trade successful, result code is ' + result['stringCode'])
                        continue
                    
                    except Exception as err:
                        print('Trade failed with error:')
                        print(api.format_error(err))

                take_profit = predicted_close[-1]
                if take_profit<current_price:#  and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):
                    stop_loss = 0# current_price + ((80/100)*current_price)                    
                    try:
                        
                        result = await connection.create_market_sell_order(
                            symbol,
                            0.01,
                            stop_loss,
                            take_profit,
                            {'trailingStopLoss': {
                                'distance': {
                                    'distance': 5,
                                    'units': 'RELATIVE_PIPS'
                                }
                            }
                        })
                        print('Trade successful, result code is ' + result['stringCode'])
                        continue
                    except Exception as err:
                        print('Trade failed with error:')
                        print(api.format_error(err))
                    

                trades = await connection.get_positions()
                if len(trades)>50:
                    await asyncio.sleep(1200)
                else:
                    print("--------------------------------------------------")
            await asyncio.sleep(3)  # Sleep for 1 minute before the next iteration
    asyncio.run(main())


timeframe_combinations = ['1m', '5m', '15m','30m','1h','4h',]
# Call the trading bot function for each combination
for smaller_tf in timeframe_combinations:
    run_trading_bot(smaller_tf)
'''
import multiprocessing
if __name__ == "__main__":
    # Define the specific combinations of bigger and smaller timeframes
    

    # Function to run the trading bot
    def run_trading_bot_wrapper(timeframe_tuple):
        smaller_tf= timeframe_tuple
        print(f"Running with timeframe: {smaller_tf} ")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        run_trading_bot(timeframe=smaller_tf)  # Pass the symbol to the function
        print(f"Finished with timeframe: {smaller_tf}")

    # Create a multiprocessing pool
    with multiprocessing.Pool() as pool:
        # Map the tasks to the pool for parallel execution
        pool.map(run_trading_bot_wrapper, timeframe_combinations)
'''