import pandas as pd
import numpy as np
import datetime as dt
import time as time
import os
from pytrends.request import TrendReq
import ML_functions as mlfcn
import Signals_Testing as st
pytrends = TrendReq(hl='en-US', tz=300)

kw_list = ['Bitcoin', 'ethereum', 'cryptocurrency']
GPROP = 'news'
#pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

results = pytrends.get_historical_interest(kw_list, year_start=2018, month_start=2, day_start=1, hour_start=0, year_end=2018, month_end=11, day_end=1, hour_end=0, cat=0, geo='', gprop=GPROP, sleep=0)


print(results)


st.write_new(results, 'google_trends_btc.xlsx','sheet1')
