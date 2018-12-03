import numpy as np
import pandas as pd
import Signals_Testing as st
import sys
sys.path.append('../')
from pytz import timezone
import time
import datetime
import json
import requests
from textblob import TextBlob
from tinydb import TinyDB, Query
from newsapi import NewsApiClient


KEY = '65af39f3a557484fa8d644068a161070'
FILTER = '&currencies=BTC&filter=important'
#See available exchanges and currencies

db = TinyDB('NewsAPI_DB.json')

newsapi = NewsApiClient(api_key=KEY)
sources = newsapi.get_sources()
print(sources)

articles = newsapi.get_everything(q='bitcoin',
                                      sources='crypto-coins-news',
                                      #domains='bbc.co.uk,techcrunch.com',
                                      from_param='2018-11-15',
                                      to='2018-11-16',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

articles = articles['articles']
for article in articles:
    print(article['title'])
    print(article['description'])


'''
for i in range(0, 2):

    POLARITY = []
    POLARITY_ADJ = []
    url = "https://cryptopanic.com/api/posts/?auth_token=" + KEY + FILTER
    response = requests.get(url).json()
    results = response['results']
    for item in results:
        title = item['title']
        sentiment = TextBlob(title).sentiment
        POLARITY.append(sentiment.polarity)
        POLARITY_ADJ.append(float(sentiment.polarity) * sentiment.subjectivity)

    pol_mean = np.array(POLARITY).mean()
    pol_adj_mean  = np.array(POLARITY_ADJ).mean()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')


    print(now)
    db.insert({'designation': 0, 'time': now, 'avg_polarity': pol_mean, 'avg_polarity_adj': pol_adj_mean})

    time.sleep(50)
'''


'''
User = Query()
search_res = db.search(User.designation == 0)
print(search_res)
'''