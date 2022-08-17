from sqlalchemy import create_engine
import pymysql
import mysql.connector
import pandas as pd

from datetime import date
from datetime import datetime
import transform


# data in db store
# the data name is here the current day's date.
# the processed data are stored in a database
host = "172.23.64.1"
user = "cartelle"
passw= "2021"
#charset="utf8mb4"
# Name of new database
new_db_name = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
#new_db_name = "test2"
print(new_db_name)

# Create new database
charSet  = "utf8mb4"     # Character set
cusrorType = pymysql.cursors.DictCursor

# Connection to server
#conn = pymysql.connect(host="172.23.64.1", user="cartelle", password="2021", charset=charSet, cursorclass=cusrorType)
conn = pymysql.connect(host=host, user=user, password=passw, charset=charSet, cursorclass=cusrorType)
cursorInsatnce  = conn.cursor()
cursorInsatnce.execute("CREATE DATABASE %s"%new_db_name)
conn.close()
# connection to database        
#sqlEngine   = create_engine('mysql+pymysql://cartelle:2021@172.23.64.1/%s'%new_db_name, pool_recycle=3600)
sqlEngine   = create_engine("mysql+pymysql://" + user + ":" + passw +"@" + host +"/" + new_db_name, pool_recycle=3600)
dbConnection = sqlEngine.connect()
try:
    # Get dataframe from transform class 
    df = transform.tranform.df
    # Save to new database
    frame = df.to_sql("data_to_analyse", dbConnection, if_exists="replace")
    
    # Get ip adresse from transform class
    IP_addresses_each_user = transform.tranform.IP_addresses_each_user
    # Save new table in new database 
    IP_addresses_each_user.to_sql("ip_addresses_each_user", dbConnection, if_exists = "replace", index=False)
    
except ValueError as vx:
    print(vx)
except Exception as ex:   
    print(ex)
else:
    print("Table data_to_analyse created successfully.");   
finally:
    dbConnection.close()

#df.to_sql("data_to_analyse", connection, if_exists = "replace", index=False)
#IP_addresses_each_user = transform.tranform.IP_addresses_each_user
#IP_addresses_each_user.to_sql("IP_addresses_each_user", connection, if_exists = "replace", index=False)
#most_visited_service = transform.tranform.most_visited_service
#most_visited_service.to_sql("most_visited_service", connection, if_exists = "replace", index=False)
#most_traffic_time = transform.tranform.most_traffic_time
#most_traffic_time.to_sql("most_traffic_time", connection, if_exists = "replace", index=False)
#most_traffic_year_time = transform.tranform.most_traffic_year_time
#most_traffic_year_time.to_sql("most_traffic_year_time", connection, if_exists = "replace", index=False)
#most_traffic_month_time= transform.tranform.most_traffic_month_time
#most_traffic_month_time.to_sql("most_traffic_month_time", connection, if_exists = "replace", index=False)
#mostpurchasse_offer= transform.tranform.mostpurchasse_offer
#mostpurchasse_offer.to_sql("mostpurchasse_offer", connection, if_exists = "replace", index=False)
#df_finance_volume = transform.tranform.df_finance_volume 
#df_finance_volume.to_sql("df_finance_volume", connection, if_exists = "replace", index=False)
#df_new = pd.read_sql("select * from data_to_analyse", connection)
