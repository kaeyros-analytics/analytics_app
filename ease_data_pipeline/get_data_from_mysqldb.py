
from sqlalchemy import create_engine
import pymysql
import pandas as pd



class get_data_class:  
    def get_data(db_name):        
        sqlEngine       = create_engine('mysql+pymysql://cartelle:2021@192.168.224.1/%s'%db_name, pool_recycle=3600)
        dbConnection    = sqlEngine.connect()
        df           = pd.read_sql("select * from ease.test_project_trackerdata", dbConnection)
        #print(df)
        pd.set_option('display.expand_frame_repr', False)
        dbConnection.close()
        return df
db_name = "ease"
get_data_class.get_data(db_name)

