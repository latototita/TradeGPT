import yfinance as yf
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Set up NLTK
nltk.download('vader_lexicon')
'''def news_analysis(symbol):
    sia = SentimentIntensityAnalyzer()
    # Retrieve news headlines using yfinance
    if symbol.endswith('m'):
        symbol = symbol[:-1]
    
    stock = yf.Ticker(symbol)
    news_df = stock.news
    # Convert the list of dictionaries into a DataFrame
    news_df = pd.DataFrame(news_df)

    results=[]
    # Perform sentiment analysis on the news headlines
    for _, row in news_df.iterrows():
        headline = row['title']
        sentiment_score = sia.polarity_scores(headline)
        sentiment = sentiment_score['compound']
        
        if sentiment >= 0.05:
            sentiment_label = f'Bullish pressure,Trend Expected to Rise for Symbol {symbol}'
        elif sentiment <= -0.05:
            sentiment_label =  f'Bearish pressure,Trend Expected to Fall for Symbol {symbol}'
        else:
            sentiment_label =f'Neutral, Effects on the Symbol {symbol} indicisive Yet.'
        sentiment=(sentiment*100)
    return headline,sentiment,sentiment_label
'''

def news_analysis(symbol):
    sia = SentimentIntensityAnalyzer()
    # Retrieve news headlines using yfinance
    if symbol.endswith('m'):
        symbol = symbol[:-1]
    stock=yf.Ticker(symbol)
    news_df = stock.news
    
    # Convert the list of dictionaries into a DataFrame
    news_df = pd.DataFrame(news_df)
    
    results = []
    # Perform sentiment analysis on the news headlines
    
    for _, row in news_df.iterrows():
        headline = row['title']
        link = row['link']
        if 'relatedTickers' in row or row['relatedTickers'] != [] or row['relatedTickers'] !='nan':
            relatedTickers = row['relatedTickers']
        else:
            relatedTickers = 'Empty'

        thumbnail = row['thumbnail']
    
        thumbnail_url = ''
        if isinstance(thumbnail, dict):
            resolutions = thumbnail.get('resolutions', [])
            if resolutions:
                thumbnail_url = resolutions[0].get('url', '')
        
        sentiment_score = sia.polarity_scores(headline)
        sentiment = sentiment_score['compound']
        
        if sentiment >= 0.05:
            sentiment_label = f'Bullish pressure, Trend Expected to Rise for Symbol {symbol}'
        elif sentiment <= -0.05:
            sentiment_label = f'Bearish pressure, Trend Expected to Fall for Symbol {symbol}'
        else:
            sentiment_label = f'Neutral, Effects on the Symbol {symbol} are Indecisive Yet.'
        
        sentiment = int(sentiment * 100)
        
        

    
    return headline,link,relatedTickers,thumbnail_url,sentiment,sentiment_label
