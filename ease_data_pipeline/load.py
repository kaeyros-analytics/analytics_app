import sqlite3
import pandas as pd

from datetime import date
from datetime import datetime
import transform

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
#write the data to the database 
df = transform.tranform.df
df.to_sql("data_to_analyse", connection, if_exists = "replace", index=False)
IP_addresses_each_user = transform.tranform.IP_addresses_each_user
IP_addresses_each_user.to_sql("IP_addresses_each_user", connection, if_exists = "replace", index=False)
most_visited_service = transform.tranform.most_visited_service
most_visited_service.to_sql("most_visited_service", connection, if_exists = "replace", index=False)
most_traffic_time = transform.tranform.most_traffic_time
most_traffic_time.to_sql("most_traffic_time", connection, if_exists = "replace", index=False)
most_traffic_year_time = transform.tranform.most_traffic_year_time
most_traffic_year_time.to_sql("most_traffic_year_time", connection, if_exists = "replace", index=False)
most_traffic_month_time= transform.tranform.most_traffic_month_time
most_traffic_month_time.to_sql("most_traffic_month_time", connection, if_exists = "replace", index=False)
mostpurchasse_offer= transform.tranform.mostpurchasse_offer
mostpurchasse_offer.to_sql("mostpurchasse_offer", connection, if_exists = "replace", index=False)
df_finance_volume = transform.tranform.df_finance_volume 
df_finance_volume.to_sql("df_finance_volume", connection, if_exists = "replace", index=False)

#df_new = pd.read_sql("select * from data_to_analyse", connection)
connection.close()
#End