import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd

# Initialize MetaApi client
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or 'cf6ff7fe-5929-4349-872a-841cac56f7dc'
timeframe = '1m'

# Define parameters
symbol_list = ['XAUUSDm','XAGUSDm','GBPUSDm','EURUSDm','AUDUSDm','AUDMXNm', 'AUDNOKm', 'AUDNZDm', 'AUDPLNm', 'AUDSEKm', 'AUDSGDm',
    'AUDTRYm', 'AUDUSDm', 'AUDUSX', 'AUDZARm', 'AUS200m', 'AUXAUD', 'AUXTHB', 'AUXUSD', 'AUXZAR',
    'AVGOm', 'BABAm', 'BACm', 'BATUSDm', 'BAm', 'BCHUSDm', 'BIIBm', 'BMYm', 'BNBUSDm', 'BTCAUDm',
    'BTCCNHm', 'BTCJPYm', 'BTCKRWm', 'BTCTHBm', 'BTCUSDm', 'BTCXAGm', 'BTCXAUm', 'BTCZARm',
    'CADCHFm', 'CADCZKm', 'CADJPYm', 'CADMXNm', 'CADNOKm', 'CADPLNm', 'CADTRYm', 'CHFDKKm',
    'CHFHUFm', 'CHFJPYm', 'CHFMXNm', 'CHFNOKm', 'CHFPLNm', 'CHFSEKm', 'CHFSGDm', 'CHFTRYm',
    'CHFZARm', 'CHTRm', 'CMCSAm', 'CMEm', 'COSTm', 'CSCOm', 'CSXm', 'CVSm', 'CZKPLNm', 'Cm',
    'DE30m', 'DKKCZKm', 'DKKHUFm', 'DKKJPYm', 'DKKPLNm', 'DKKSGDm', 'DKKZARm', 'DOTUSDm', 'DXYm',
    'EAm', 'EBAYm', 'ENJUSDm', 'EQIXm', 'ETHUSDm', 'EURAUDm', 'EURAUX', 'EURCADm', 'EURCHFm',
    'EURCZKm', 'EURDKKm', 'EURGBPm', 'EURGBX', 'EURHKDm', 'EURHKX', 'EURHUFm', 'EURJPX', 'EURJPYm',
    'EURMXNm', 'EURNOKm', 'EURNZDm', 'EURPLNm', 'EURSEKm', 'EURSGDm', 'EURTRYm', 'EURUSDm', 'EURUSX',
    'EURZARm', 'EUXAUD', 'EUXEUR', 'EUXGBP', 'EUXTHB', 'EUXUSD', 'EUXZAR', 'FBm', 'FILUSDm', 'FR40m',
    'Fm', 'GBPAUDm', 'GBPAUX', 'GBPCADm', 'GBPCHFm', 'GBPCZKm', 'GBPDKKm', 'GBPHKX', 'GBPHUFm',
    'GBPILSm', 'GBPJPX', 'GBPJPYm', 'GBPMXNm', 'GBPNOKm', 'GBPNZDm', 'GBPPLNm', 'GBPSEKm', 'GBPSGDm',
    'GBPTRYm', 'GBPUSDm', 'GBPUSX', 'GBPZARm', 'GBXAUD', 'GBXGBP', 'GBXTHB', 'GBXUSD', 'GBXZAR',
    'GILDm', 'GOOGLm', 'HDm', 'HK50m', 'HKDJPYm', 'HKXHKD', 'HKXTHB', 'HKXZAR', 'HUFJPYm', 'IBMm',
    'IN50m', 'INTCm', 'INTUm', 'ISRGm', 'JNJm', 'JP225m', 'JPMm', 'JPXJPY', 'KOm', 'LINm', 'LLYm',
    'LMTm', 'LTCUSDm', 'MAm', 'MCDm', 'MDLZm', 'METAm', 'MMMm', 'MOm', 'MRKm', 'MSFTm', 'MSm', 'MXNJPYm',
    'NADUSD', 'NFLXm', 'NKEm', 'NOKDKKm', 'NOKJPYm', 'NOKSEKm', 'NVDAm', 'NZDCADm', 'NZDCHFm',
    'NZDCZKm', 'NZDDKKm', 'NZDHUFm', 'NZDJPYm', 'NZDMXNm', 'NZDNOKm', 'NZDPLNm', 'NZDSEKm', 'NZDSGDm',
    'NZDTRYm', 'NZDUSDm', 'NZDZARm', 'ORCLm', 'PEPm', 'PFEm', 'PGm', 'PLNDKKm', 'PLNHUFm', 'PLNJPYm',
    'PLNSEKm', 'PMm', 'PYPLm', 'REGNm', 'SBUXm', 'SEKDKKm', 'SEKJPYm', 'SEKPLNm', 'SGDHKDm',
    'SGDJPYm', 'SNXUSDm', 'SOLUSDm', 'STOXX50m', 'THBJPX', 'TMOm', 'TMUSm', 'TRXUSD', 'TRYDKKm',
    'TRYJPYm', 'TRYZARm', 'TSLAm', 'Tm', 'UK100m', 'UKOILm', 'UNHm', 'UNIUSDm', 'UPSm', 'US30_x10m',
    'US30m', 'US500_x100m', 'US500m', 'USDAED', 'USDAEDm', 'USDAMD', 'USDAMDm', 'USDARS', 'USDARSm',
    'USDAZN', 'USDAZNm', 'USDBDT', 'USDBDTm', 'USDBGN', 'USDBGNm', 'USDBHD', 'USDBHDm', 'USDBND',
    'USDBNDm', 'USDBRL', 'USDBRLm', 'USDBYN', 'USDBYR', 'USDCADm', 'USDCHFm', 'USDCLP', 'USDCLPm',
    'USDCNHm', 'USDCNY', 'USDCNYm', 'USDCOP', 'USDCOPm', 'USDCRC', 'USDCZKm', 'USDDKKm', 'USDDZD',
    'USDDZDm', 'USDEGP', 'USDEGPm', 'USDGEL', 'USDGELm', 'USDGHS', 'USDGHSm', 'USDHKDm', 'USDHKX',
    'USDHRK', 'USDHRKm', 'USDHUF', 'USDHUFm', 'USDIDR', 'USDIDRm', 'USDILSm', 'USDINR', 'USDINRm',
    'USDIRR', 'USDISK', 'USDISKm', 'USDJOD', 'USDJODm', 'USDJPX', 'USDJPYm', 'USDKES', 'USDKESm',
    'USDKGS', 'USDKGSm', 'USDKHR', 'USDKRW', 'USDKRWm', 'USDKWD', 'USDKWDm', 'USDKZT', 'USDKZTm',
    'USDLAK', 'USDLBP', 'USDLBPm', 'USDLKR', 'USDLKRm', 'USDMAD', 'USDMADm', 'USDMMK', 'USDMXNm',
    'USDMYR', 'USDMYRm', 'USDNGN', 'USDNGNm', 'USDNOKm', 'USDNPR', 'USDNPRm', 'USDOMR', 'USDOMRm',
    'USDPAB', 'USDPEN', 'USDPHP', 'USDPHPm', 'USDPKR', 'USDPKRm', 'USDPLNm', 'USDPYG', 'USDQAR',
    'USDQARm', 'USDROL', 'USDRON', 'USDRONm', 'USDRUB', 'USDRUBm', 'USDRUR', 'USDRURm', 'USDRWF',
    'USDSAR', 'USDSARm', 'USDSCR', 'USDSEKm', 'USDSGDm', 'USDSYP', 'USDSYPm', 'USDTHBm', 'USDTJS',
    'USDTJSm', 'USDTMT', 'USDTMTm', 'USDTND', 'USDTNDm', 'USDTRYm', 'USDTUSD', 'USDTWD', 'USDTWDm',
    'USDTZS', 'USDUAH', 'USDUAHm', 'USDUGX', 'USDUGXm', 'USDUYU', 'USDUZS', 'USDUZSm', 'USDVND',
    'USDVNDm', 'USDVUV', 'USDVUVm', 'USDXAF', 'USDXOF', 'USDXOFm', 'USDZARm', 'USDZMW', 'USOILm',
    'USTEC_x100m', 'USTECm', 'USXJPY', 'USXRUB', 'USXTHB', 'USXUSD', 'USXZAR', 'VRTXm', 'VZm', 'Vm',
    'WFCm', 'WMTm', 'XAGAUDm', 'XAGEURm', 'XAGGBPm', 'XAGJPYm', 'XAGUSDm', 'XAUAUDm', 'XAUEURm',
    'XAUGBPm','XAUUSDm','XNGUSDm','XOMm','XPDUSDm','XPTUSDm','XRPUSDm','XTZUSDm','ZARJPX','ZARJPYm'
    ]
rsi_period = 14

import pandas as pd
import numpy as np
import pandas_ta as ta

def check_signals(candles, rsi_period):
    # Convert candles to DataFrame
    df = pd.DataFrame(candles)
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)

    # Apply RSI indicator
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)

    # Apply ATR indicator
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Check the most recent crossing
    last_crossing_index = None
    last_crossing_threshold = None

    cross_above_70 = np.where(df['rsi'] > 67)[0]
    cross_below_30 = np.where(df['rsi'] < 33)[0]

    if len(cross_above_70) > 0:
        last_crossing_index = cross_above_70[-1]
        last_crossing_threshold = 70

    if len(cross_below_30) > 0 and (last_crossing_index is None or cross_below_30[-1] > last_crossing_index):
        last_crossing_index = cross_below_30[-1]
        last_crossing_threshold = 30

    if last_crossing_index is not None:
        last_crossing_value = df['rsi'][last_crossing_index]
        current_rsi_value = df['rsi'][-1]
        if abs(current_rsi_value - last_crossing_threshold) <= 10:
            if last_crossing_threshold == 70:
                trend_direction = "Downwards"
                sell_signal = current_rsi_value >= 65 and last_crossing_threshold == 70
                buy_signal = not sell_signal
                if sell_signal:
                    print("Sell signal, crossed 70, now in a downtrend")
            else:
                trend_direction = "Upwards"
                buy_signal = current_rsi_value <= 35 and last_crossing_threshold == 30
                sell_signal = not buy_signal
                if buy_signal:
                    print("Buy signal, crossed 30, now in an uptrend")

            print(f"The RSI crossed {last_crossing_threshold} at index {last_crossing_index} with a value of {last_crossing_value}.")
            print(f"The current trend direction is {trend_direction}.")

            return buy_signal, sell_signal
        else:
            return False, False
    else:
        return False, False

async def main():
    # Connect to the MetaTrader account
    while True:
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

        while True:
            # Check for open trades
            trades = await connection.get_positions()#connection.get_orders()
            if trades:
                print("There are open trades. Skipping analysis.")
                await asyncio.sleep(3600)
            else:
                for symbol in symbol_list:
                    print(symbol)
                    trades = await connection.get_positions()
                    if trades:
                        print("There are open trades. Skipping analysis.")
                        await asyncio.sleep(1000)
                        continue
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=1000)
                        print('Fetched the latest candle data successfully')

                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")

                    buy_signal, sell_signal = check_signals(candles, rsi_period)

                    # Execute trading orders
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                            
                    atr_multiplier = 2.0  # Multiplier to determine the distance from the current price based on ATR
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df['close'] = df['close'].astype(float)
                    # Apply ATR indicator
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                    if buy_signal:
                        take_profit = current_price + (0.5 * df['atr'][-1])
                        stop_loss = current_price - (atr_multiplier * df['atr'][-1])
                        try:
                            # Calculate margin required for trade
                            first_margin = await connection.calculate_margin({
                                'symbol': symbol,
                                'type': 'ORDER_TYPE_BUY',
                                'volume': 0.01,
                                'openPrice':  current_price
                            })
                            first_margins = float(first_margin['margin'])
                            if first_margins < ((4/100) * 10):
                                result = await connection.create_market_buy_order(
                                    symbol,
                                    0.01,
                                    stop_loss,
                                    take_profit,
                                    {'trailingStopLoss': {
                                        'distance': {
                                            'distance': 5,
                                            'units': 'RELATIVE_BALANCE_PERCENTAGE'
                                        }
                                    }
                                })
                            else:
                                continue
                            print('Trade successful, result code is ' + result['stringCode'])
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))
                    if sell_signal:
                        take_profit = current_price - (0.5 * df['atr'][-1])
                        stop_loss = current_price + (atr_multiplier * df['atr'][-1])
                        try:
                            # Calculate margin required for trade
                            first_margin = await connection.calculate_margin({
                                'symbol': symbol,
                                'type': 'ORDER_TYPE_SELL',
                                'volume': 0.01,
                                'openPrice':  current_price,
                            })
                            first_margins = float(first_margin['margin'])
                            if first_margins < ((4/100) * 10):
                                result = await connection.create_market_sell_order(
                                    symbol,
                                    0.01,
                                    stop_loss,
                                    take_profit,
                                    {'trailingStopLoss': {
                                        'distance': {
                                            'distance': 5,
                                            'units': 'RELATIVE_BALANCE_PERCENTAGE'
                                        }
                                    }
                                })
                            else:
                                continue
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))

                    trades = await connection.get_positions()
                    if trades:
                        await asyncio.sleep(1000)
                    else:
                        print("--------------------------------------------------")
                        await asyncio.sleep(300)  # Sleep for 1 minute before the next iteration
            trades = await connection.get_positions()
            if trades:
                await asyncio.sleep(1000) 
            else:
                print("--------------------------------------------------")
        await asyncio.sleep(1000)  # Sleep for 1 minute before the next iteration

asyncio.run(main())
