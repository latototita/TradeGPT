import asyncio
from telegram import Bot
from telegram.error import TelegramError
import time,os
from views import *
from metaapi_cloud_sdk import MetaApi


token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '615ae0df-2198-4162-9b23-34a4285baa35'

symbols = ['XAUUSDm', 'GBPUSDm', 'XAGUSDm', 'AUDUSDm', 'EURUSDm', 'USDJPYm', 'GBPTRYm','AUDCADm','AUDCHFm','AUDJPYm','CADJPYm','CHFJPYm','EURCADm', 'EURAUDm','EURCHFm','EURGBPm','EURJPYm','GBPAUDm','GBPCADm', 'GBPCHFm','GBPJPYm']
timeframes  = ['15m','30m','1h']



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
async def send_message(message):
    bot_token = '6189465976:AAEIHkUfkQIxwaF8qdMDFvS44wErTR2i9_U'
    group_id = '-1001915184051'
    message_text = message
    
    bot = Bot(token=bot_token)
    
    try:
        await bot.send_message(chat_id=group_id, text=message_text)
        print('Message sent successfully!')
    except TelegramError as e:
        print('Error:', e.message)


def telbot():
    word1="Buy"
    word2="Sell"
    for symbol in symbols:
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            bb=str(calculate_bb(candles))
            rsi=str(calculate_rsi(candles))
            ema=str(exponential_moving_average(candles))
            adx=str(calculate_adx(candles))
            csp=str(candlepatterndt(candles))
            if word1.lower() or word2.lower() in bb.lower():
                message=f'For symbol : {symbol} and timeframe : {timeframe},For symbol : {symbol} and timeframe : {timeframe},Bollinger Bands: {bb}'
                asyncio.run(send_message(message))
                time.sleep(240)
            if word1.lower() or word2.lower() in ema.lower():
                message=f'For symbol : {symbol} and timeframe : {timeframe},Exponential moving average :{ema}'
                asyncio.run(send_message(message))
                time.sleep(240)
            if word1.lower() or word2.lower() in rsi.lower():
                message=f'For symbol : {symbol} and timeframe : {timeframe},Relative Strength Index :{rsi}'
                asyncio.run(send_message(message))
                time.sleep(240)
            if word1.lower() or word2.lower() in adx.lower():
                message=f'For symbol : {symbol} and timeframe : {timeframe},Average Directional Index{adx}'
                asyncio.run(send_message(message))
                time.sleep(240)
            if word1.lower() or word2.lower() in csp.lower():
                message=f'For symbol : {symbol} and timeframe : {timeframe},Candles stick pattern :{csp}'
                asyncio.run(send_message(message))
                time.sleep(240)
            time.sleep(240)
        time.sleep(240)
telbot()
