import asyncio
from telegram import Bot
from telegram.error import TelegramError
from tasks.views import *
import time
from tasks.views import run_tasks
from trading_bot_django import settings
symbols = ['XAUUSDm', 'GBPUSDm', 'XAGUSDm', 'AUDUSDm', 'EURUSDm', 'USDJPYm', 'GBPTRYm','AUDCADm','AUDCHFm','AUDJPYm','CADJPYm','CHFJPYm','EURCADm', 'EURAUDm','EURCHFm','EURGBPm','EURJPYm','GBPAUDm','GBPCADm', 'GBPCHFm','GBPJPYm']
timeframes  = ['1h','4h','1d']

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


'''
def testdrive():
    symbols = ['XAUUSDm', 'GBPUSDm', 'XAGUSDm', 'AUDUSDm', 'EURUSDm', 'USDJPYm', 'GBPTRYm','AUDCADm','AUDCHFm','AUDJPYm','CADJPYm','CHFJPYm','EURCADm', 'EURAUDm','EURCHFm','EURGBPm','EURJPYm','GBPAUDm','GBPCADm', 'GBPCHFm','GBPJPYm']
    timeframes  = ['5m','15m','30m','1h','4h','1d']
    for symbol in symbols:
        for timeframe in timeframes:
            List=[]
            candles=asyncio.run(get_candles_m(timeframe,symbol))
            if type(candles)== str:
                word3="Error"
                if word3.lower() in candles.lower() :
                    continue
                else:
                    atr=average_true_range(candles)
                    List.append(atr)
                    ma=moving_average(candles)
                    List.append(ma)
                    parabolic=parabolic_sar(candles)
                    List.append(parabolic)
                    ichimuko=calculate_ichimoku(candles)
                    List.append(ichimuko)
                    stocastic=stochastic_oscillator(candles)
                    List.append(stocastic)
                    fibonachi=calculate_fibonachi(candles)
                    List.append(fibonachi)
                    bb=calculate_bb(candles)
                    List.append(bb)
                    rsi=calculate_rsi()
                    List.append(rsi)
                    macd=str(calculate_macd(candles,n_fast, n_slow, n_signal))
                    List.append(macd)
                    adx=calculate_adx(candles)
                    List.append(adx)
                    adl=calculate_adl(candles)
                    List.append(adl)
                    ema=exponential_moving_average(candles)
                    List.append(ema)
                    trendlines=calculate_trendlines(candles)
                    List.append(trendlines)

                    concatenated_string = "\n".join(List)
    
                    if concatenated_string != "":
                        message=concatenated_string
                        asyncio.run(send_message(message))
        time.sleep(240)
    time.sleep(240)

'''
n_fast = 3
n_slow = 6
n_signal = 4


def telbot():
    for symbol in symbols:
        for timeframe in timeframes:
            candles=asyncio.run(get_candles(timeframe,symbol))
            if type(candles)== str:
                if candles.lower() in ["error"]:
                    break
            else:
                pp=calculate_pivot_point(candles)
                fibonachi=calculate_fibonachi(candles)
                gan_fan=calculate_gannfan(candles)
                sppt=cal_support_resistance(candles)
                trendlines=calculate_trendlines(candles)
                
                if trendlines.lower() in ["sell",] and sppt.lower() in ["sell",] and gan_fan.lower() in ["sell",] and fibonachi.lower() in ["sell",] and pp.lower() in ["sell",]:
                    message=f'For symbol : {symbol} and timeframe : {timeframe},Sell based on Pivot point,Fibonacchi Retracement, Gann Fann, Support  and Resistance and Trendlines'
                    asyncio.run(send_message(message))
<<<<<<< HEAD
                if trendlines.lower() in ["buy",] and sppt.lower() in ["buy",] and gan_fan.lower() in ["buy",] and fibonachi.lower() in ["buy",] and pp.lower() in ["buy",]:
                    message=f'For symbol : {symbol} and timeframe : {timeframe},buy based on Pivot point,Fibonacchi Retracement, Gann Fann, Support  and Resistance and Trendlines'
                    asyncio.run(send_message(message))
                

                #bb=str(calculate_bb(candles))
                rsi=calculate_rsi(candles)
                ema=exponential_moving_average(candles)
                adx=calculate_adx(candles)
                macd=calculate_macd(candles,n_fast, n_slow, n_signal)
                '''
                if  bb.lower() in ["sell","buy","strong"]:
                    message=f'For symbol : {symbol} and timeframe : {timeframe},For symbol : {symbol} and timeframe : {timeframe},Bollinger Bands:  {bb}'
                    List.append(message)
                
                if  ema.lower() in ["sell","buy","strong"]:
                    message=f'For symbol : {symbol} and timeframe : {timeframe},Exponential moving average :{ema}'
                    List.append(message)
                
                if  rsi.lower() in ["sell","buy","strong"]:
                    message=f'For symbol : {symbol} and timeframe : {timeframe},Relative Strength Index :{rsi} possible trend reversal'
                    List.append(message)

                if  adx.lower() in ["sell","buy","strong"]:
=======
                    time.sleep(240)
                
                if word1.lower() or word2.lower() in rsi.lower():
                    message=f'For symbol : {symbol} and timeframe : {timeframe},Relative Strength Index :{rsi} possible trend reversal'
                    asyncio.run(send_message(message))
                    print('Yes')
                    time.sleep(240)'''
                if word1.lower() or word2.lower() in adx.lower():
>>>>>>> ab8bd84057fe4e63cbcf469ca09767151c78ece6
                    message=f'For symbol : {symbol} and timeframe : {timeframe},Average Directional Index{adx}'
                    List.append(message)

                if  macd.lower() in ["sell","buy","strong"]:
                    message=f'For symbol : {symbol} and timeframe : {timeframe},Average Directional Index{adx}'
                    List.append(message)
                    '''

                if ( adx.lower() in ["sell","strong"] and ema.lower() in ["sell","strong"] and macd() in ["sell","strong"]) and (rsi.lower() in ["sell","neautral"]):
                    message=f'For symbol : {symbol} and timeframe : {timeframe}, MACD={macd},EMA={ema},{adx} and {rsi} all, Place a sell order'
                    asyncio.run(send_message(message))
                if ( adx.lower() in ["buy","strong"] and ema.lower() in ["buy","strong"] and macd() in ["buy","strong"]) and (rsi.lower() in ["buy","neautral"]):
                    message=f'For symbol : {symbol} and timeframe : {timeframe}, MACD={macd},EMA={ema},{adx} and {rsi}'
                    asyncio.run(send_message(message))
            time.sleep(240)
        time.sleep(240)
#run_tasks()
telbot()


