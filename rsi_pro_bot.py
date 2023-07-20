import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd
from indicator_collection import *



token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '86734b47-6b5f-45c7-91e6-c72243b6a293'

# Define parameters

rsi_period = 14







def run_trading_bot(timeframez,btimeframe):
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
        if len(trades)>10:
            print("There are open trades. Skipping analysis.")
            await asyncio.sleep(1200)
        else:
            for symbol in symbol_list:
                print(symbol)
                trades = await connection.get_positions()
                if len(trades)>10:
                    print("There are open trades. Skipping analysis.")
                    await asyncio.sleep(1200)
                    continue
                try:
                    # Fetch historical price data
                    biggerCandles = await account.get_historical_candles(symbol=symbol, timeframe=btimeframe, start_time=None, limit=1000)
                except Exception as e:
                    print(f"Error retrieving candle data: {e}")

                #buy_signal, sell_signal = check_signals(candles, rsi_period)
                #Bbuy_signal, Bsell_signal = detect_trend_reversals(biggerCandles)
                
                buy_signal, sell_signal = check_market_condition(biggerCandles)#check_signals_biggers(biggerCandles, rsi_period)
                if btimeframe=='1m':
                    timeframe='5m'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=generate_entry_signal(candless)
                elif btimeframe=='5m':
                    timeframe='15m'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=generate_entry_signal(candless)
                elif btimeframe=='15m':
                    timeframe='30m'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=generate_entry_signal(candless)
                elif btimeframe=='30m':
                    timeframe='1h'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=generate_entry_signal(candless)
                elif btimeframe=='1h':
                    timeframe='4h'
                    try:
                        # Fetch historical price data
                        candless = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue
                    Bbuy_signal,Bsell_signal=generate_entry_signal(candless)
                print(buy_signal,sell_signal)
                print(timeframez,btimeframe)
                dt=cal_double_tops(biggerCandles)
                db=cal_identify_double_bottom(biggerCandles)
                hns=cal_identify_head_shoulders_pattern(biggerCandles)
                jcp=identify_candle_pattern(biggerCandles)
                tripplet=triple_top(biggerCandles)
                trippleb=triple_bottom(biggerCandles)
                ihns=inverse_head_and_shoulders(biggerCandles)
                
                keywords = ["buy",]
                if buy_signal==True: #  and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):
                    # Execute trading orders
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframez, start_time=None, limit=1000)
                        
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                            
                    atr_multiplier = 15.0  # Multiplier to determine the distance from the current price based on ATR
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df['close'] = df['close'].astype(float)
                    # Apply ATR indicator
                    
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    # Determine the ATR multiplier based on the atr_value
                    if df['atr'][-1]>1<=2:
                        atr_multiplierx = 2
                        atr_multiplier = 5
                    elif df['atr'][-1] >2<=4:
                        atr_multiplierx =2
                        atr_multiplier = 5
                    elif df['atr'][-1] >4:
                        atr_multiplierx =1
                        atr_multiplier = 6
                    else:
                        atr_multiplierx =2
                        atr_multiplier = 6
                    take_profit = current_price + (atr_multiplierx* df['atr'][-1])
                    stop_loss = current_price - (atr_multiplier * df['atr'][-1])
                    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
                    print('atr',df['atr'][-1])
                    
                    try:
                        # Calculate margin required for trade
                        first_margin = await connection.calculate_margin({
                            'symbol': symbol,
                            'type': 'ORDER_TYPE_BUY',
                            'volume': 0.01,
                            'openPrice':  current_price
                        })
                        first_margins = float(first_margin['margin'])
                        if first_margins < ((10/100) * 10):
                            result = await connection.create_market_buy_order(
                                symbol,
                                0.01,
                                stop_loss,
                                take_profit,
                                {'trailingStopLoss': {
                                    'distance': {
                                        'distance': 1,
                                        'units': 'RELATIVE_BALANCE_PERCENTAGE'
                                    }
                                }
                            })
                            print('Trade successful, result code is ' + result['stringCode'])
                            continue
                        
                    except Exception as err:
                        print('Trade failed with error:')
                        print(api.format_error(err))

                keywords = ["sell",]
            
                if sell_signal==True:#  and any(keyword in var for var in [dt, db, hns,jcp,trippleb,tripplet,ihns] for keyword in keywords):

                    # Execute trading orders
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframez, start_time=None, limit=1000)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                            
                     # Multiplier to determine the distance from the current price based on ATR
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df['close'] = df['close'].astype(float)
                    # Apply ATR indicator
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    # Determine the ATR multiplier based on the atr_value
                    if df['atr'][-1]>1<=2:
                        atr_multiplierx = 4
                        atr_multiplier = 5
                    elif df['atr'][-1] >2<=4:
                        atr_multiplierx =2
                        atr_multiplier = 5
                    elif df['atr'][-1] >4:
                        atr_multiplierx =2
                        atr_multiplier = 6
                    else:
                        atr_multiplierx =3
                        atr_multiplier = 6
                    take_profit = current_price - (atr_multiplierx * df['atr'][-1])
                    stop_loss = current_price + (atr_multiplier * df['atr'][-1])
                    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
                    
                    
                    try:
                        # Calculate margin required for trade
                        first_margin = await connection.calculate_margin({
                            'symbol': symbol,
                            'type': 'ORDER_TYPE_SELL',
                            'volume': 0.01,
                            'openPrice':  current_price,
                        })
                        first_margins = float(first_margin['margin'])
                        if first_margins < ((10/100) * 10):
                            result = await connection.create_market_sell_order(
                                symbol,
                                0.01,
                                stop_loss,
                                take_profit,
                                {'trailingStopLoss': {
                                    'distance': {
                                        'distance': 1,
                                        'units': 'RELATIVE_BALANCE_PERCENTAGE'
                                    }
                                }
                            })
                            print('Trade successful, result code is ' + result['stringCode'])
                            continue
                    except Exception as err:
                        print('Trade failed with error:')
                        print(api.format_error(err))
                    

                trades = await connection.get_positions()
                if len(trades)>10:
                    await asyncio.sleep(1200)
                else:
                    print("--------------------------------------------------")
        await asyncio.sleep(3)  # Sleep for 1 minute before the next iteration
    asyncio.run(main())


'''
# Define the specific combinations of bigger and smaller timeframes
timeframe_combinations = [('1m', '5m'), ('15m', '5m'), ('30m', '15m'), ('1h', '30m'), ('4h', '1h')]

# Call the trading bot function for each combination
for smaller_tf, bigger_tf in timeframe_combinations:
    run_trading_bot(timeframez =smaller_tf,btimeframe= bigger_tf)
'''
import multiprocessing

# Define the specific combinations of bigger and smaller timeframes
timeframe_combinations = [('1m', '5m'), ('15m', '5m'), ('30m', '15m'), ('1h', '30m'), ('4h', '1h')]

# Function to run the trading bot
def run_trading_bot_wrapper(timeframe_tuple):
    smaller_tf, bigger_tf = timeframe_tuple
    print(f"Running with timeframe: {smaller_tf} - {bigger_tf}")
    run_trading_bot(timeframez=smaller_tf, btimeframe=bigger_tf)
    print(f"Finished with timeframe: {smaller_tf} - {bigger_tf}")

# Create a multiprocessing pool
with multiprocessing.Pool() as pool:
    # Map the tasks to the pool for parallel execution
    pool.map(run_trading_bot_wrapper, timeframe_combinations)
