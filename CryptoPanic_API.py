import numpy as np
import pandas as pd
import Signals_Testing as st
import sys
sys.path.append('../')
from pytz import timezone
import requests
import time
import datetime
import json
import requests
from textblob import TextBlob
from tinydb import TinyDB, Query

KEY = '2fbdda0382df29cae0bfaca9638c457e17089aeb'
FILTER = '&currencies=BTC&filter=important'
#See available exchanges and currencies

db = TinyDB('CryptoPanic_DB.json')

User = Query()
search_res = db.search(User.designation == 0)
print(search_res)
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

    time.sleep(1)

'''