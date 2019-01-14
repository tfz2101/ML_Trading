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

KEY = '2fbdda0382df29cae0bfaca9638c457e17089aeb'
FILTER = '&currencies=BTC&filter=important'
#See available exchanges and currencies

db = TinyDB('CryptoPanic_DB.json')
designation = 1

#RECORD SENTIMENT/POSITIVES/NEGATIVES OF LAST 20 IMPORTANT ARTICLES IN 5 MIN INTERNVALS
INTERVAL = 300


for i in range(0, 10000):

    #Important Cateory of news sentiment
    POLARITY = []
    POLARITY_ADJ = []
    PRINT_TIMES  = []
    POSITIVES = []
    NEGATIVES = []
    url = "https://cryptopanic.com/api/posts/?auth_token=" + KEY + FILTER
    response = requests.get(url).json()
    results = response['results']
    print(results)
    for item in results:
        title = item['title']
        sentiment = TextBlob(title).sentiment
        POLARITY.append(sentiment.polarity)
        POLARITY_ADJ.append(float(sentiment.polarity) * sentiment.subjectivity)

        positive = item['votes']['positive']
        POSITIVES.append(positive)
        negative = item['votes']['negative']
        NEGATIVES.append(negative)
        print_time = str(item['created_at'])
        PRINT_TIMES.append(print_time)

    pol_mean = np.array(POLARITY).mean()
    pol_adj_mean  = np.array(POLARITY_ADJ).mean()
    latest_created_time =  PRINT_TIMES[0]
    avg_positives = np.array(POSITIVES).mean()
    avg_negatives =  np.array(NEGATIVES).mean()

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')
    print(now)

    db.insert({'designation': 0, 'time': now, 'latest_created_time': latest_created_time, 'avg_positives': avg_positives, 'avg_negatives': avg_negatives, 'avg_polarity': pol_mean, 'avg_polarity_adj': pol_adj_mean})

    time.sleep(INTERVAL)



#RETRIEVE ALL RECORDS
'''
User = Query()
search_res = db.search(User.designation == 0)
search_res = pd.DataFrame(search_res)
print(search_res)
st.write_new(search_res, 'CyptoPanic_Sentiment_Data.xlsx', 'data')
'''