
from sqlalchemy import create_engine
import pymysql
import pandas as pd

sqlEngine       = create_engine('mysql+pymysql://root:2021@127.0.0.1/ease', pool_recycle=3600)
dbConnection    = sqlEngine.connect()
frame           = pd.read_sql("select * from ease.test_project_trackerdata", dbConnection);

pd.set_option('display.expand_frame_repr', False)

print(frame)

dbConnection.close()