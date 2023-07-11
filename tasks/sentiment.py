from textblob import TextBlob
import yfinance as yf
import pandas as pd
import datetime
from trading_bot_django import settings
def cal_sentiment_textblob(symbols):
    results = []
    for symbol in symbols:
        try:
            # Retrieve news headlines using yfinance
            stock = yf.Ticker(symbol)
            news_df = stock.news
            news_df = pd.DataFrame(news_df)
            
            # Sort news by publication time in descending order
            try:
                news_df = news_df.sort_values('providerPublishTime', ascending=False)
            except:
                news_df = news_df.sort_values('uuid', ascending=False)
            
            print(f"Sentiment analysis for symbol: {symbol}")
            print("--------------------------------------")
            
            symbol_results = []
            
            # Perform sentiment analysis on the latest news headlines
            for index, row in news_df.iterrows():
                headline = row['title']
                # Perform sentiment analysis using TextBlob
                blob = TextBlob(headline)
                sentiment = blob.sentiment.polarity
                
                if sentiment > 0:
                    sentiment_label = 'The news excites Investors,High purchases expected and Price(Value) expected to Rise'
                elif sentiment < 0:
                    sentiment_label = 'The news depresses Investors,High sale expected and Price(Value) expected to Fall'
                else:
                    sentiment_label = 'The News has No/Little effect on the Prices'
                thumbnail_data=row['thumbnail']
                thumbnail_url = thumbnail_data['resolutions'][0]['url']
                result = {
                    'symbol': symbol,
                    'headline': headline,
                    'publisher': row['publisher'],
                    'link': row['link'],
                    'providerPublishTime': row['providerPublishTime'],
                    'thumbnail': thumbnail_url,
                    'sentiment_score': sentiment,
                    'sentiment_label': sentiment_label
                }
                
                symbol_results.append(result)
                # Limit the analysis to the latest news

            
            results.extend(symbol_results)
        except:
            pass
    
    return results


symbols = ['AAPL', ]#'GOOGL', 'MSFT', 'AMZN', 'FB', 'JPM', 'BAC', 'WMT', 'V', 'JNJ', 'MA', 'T', 'VZ', 'PG', 'INTC', 'HD', 'DIS', 'TSLA', 'NVDA', 'UNH']
#currency_pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF', 'NZDUSD', 'USDCAD', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURGBP', 'EURCHF', 'EURAUD', 'GBPCHF', 'EURCAD', 'AUDCAD', 'CADJPY', 'CHFJPY', 'AUDNZD', 'NZDJPY']

#symbols.extend(currency_pairs)
sentiment_results = cal_sentiment_textblob(symbols)
settings.results=sentiment_results
# Print the sentiment results
for result in sentiment_results:
    print('Symbol:', result['symbol'])
    print('Headline:', result['headline'])
    print('Publisher:', result['publisher'])
    print('Link:', result['link'])
    print('Provider Publish Time:', result['providerPublishTime'])
    print('Thumbnail:', result['thumbnail'])
    print('Sentiment Score:', result['sentiment_score'])
    print('Sentiment Label:', result['sentiment_label'])
    print('--------------------------------------')
