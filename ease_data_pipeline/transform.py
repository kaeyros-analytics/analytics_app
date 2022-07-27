import sqlite3
import pandas as pd

from datetime import date
from datetime import datetime
import extract

class tranform:   
    def newsfeatures(df):
        df['timestamp_start_1'] = pd.to_datetime(df['timestamp_start'])
        df['timestamp_end_1'] = pd.to_datetime(df['timestamp_end'])
        df['Month'] = df['timestamp_start_1'].dt.month
        df['year'] = df['timestamp_start_1'].dt.year
        df['hour'] = df['timestamp_start_1'].dt.hour
        df['dayofweek'] = df['timestamp_start_1'].dt.dayofweek
        df['Min_end'] = df['timestamp_end_1'].dt.minute
        df['Min_start'] = df['timestamp_start_1'].dt.minute
        df['Min_diff'] = abs(df['Min_end'] - df['Min_start'])
        
        return(df)

    df = newsfeatures(extract.extract.extraction())
    IP_addresses_each_user = df.groupby('ip')['location_offer'].value_counts()
    most_visited_service = df.service_name.value_counts()
    most_traffic_time = df.hour.value_counts()
    most_traffic_year_time = df.groupby('year')['hour'].value_counts()
    most_traffic_month_time = df.groupby('year')['Month'].value_counts()
    mostpurchasse_offer = df.groupby('offer_name')['purchase_y_n'].value_counts()
    df_finance_volume = df.query('purchase_y_n > 0')
    df_finance_volume  = df_finance_volume.filter(['service_name', 'offer_name', 'price_offer', 'Month', 'dayofweek','year'])
    df_finance_volume.groupby('offer_name')['price_offer'].mean()