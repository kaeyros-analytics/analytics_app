
import sqlite3
import pandas as pd

from datetime import date
from datetime import datetime

# Create your connection.
# read sqlite data to pandas dataframe
#/c/Users/user/dags/ease_data_pipeline/

cnx = sqlite3.connect('/c/Users/user/dags/ease_data_pipeline/db.sqlite3')
df = pd.read_sql_query("SELECT * FROM test_project_trackerdata", cnx)
cnx.close

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

df = newsfeatures(df)

# data in db store
# the data name is here the current day's date.
# the processed data are stored in a database

# Define Path to new database
drive_letter = "/c/Users/user/dags/ease_data_pipeline/" 
file_name = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
file_exten = ".sqlite"
dbPath = drive_letter + file_name + file_exten
# connection to database
connection = sqlite3.connect(dbPath)
df.to_sql("data_to_analyse", connection, if_exists = "replace", index=False)
#df_new = pd.read_sql("select * from data_to_analyse", connection)
connection.close()