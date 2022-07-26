import sqlite3
import pandas as pd


# Create your connection.
# read sqlite data to pandas dataframe
#/c/Users/user/dags/ease_data_pipeline/

class extract:
    def extraction():
        cnx = sqlite3.connect('/c/Users/user/dags/ease_data_pipeline/db.sqlite3')
        df = pd.read_sql_query("SELECT * FROM test_project_trackerdata", cnx)
        cnx.close
        return df