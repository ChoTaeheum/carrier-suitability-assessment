# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:36:50 2021

@author: modn
"""

import pymysql
pymysql.install_as_MySQLdb()
import _mysql
from sqlalchemy import create_engine
import pandas


DB_HOST = "211.233.58.16"
PORT = 3306  #포트번호
USER_NAME = "thcho"
USER_PSWD = "2281"

class CallDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.engine = create_engine("mysql+mysqldb://{}:{}@{}/{}".format(USER_NAME, USER_PSWD, DB_HOST, self.db_name), encoding='utf-8')
	
    
    def query_db(self, query):
        dbconn = _mysql.connect(host=DB_HOST, user=USER_NAME, passwd=USER_PSWD, port=PORT, db=self.db_name)
        self.cursor = dbconn.cursor()
        self.cursor.execute(query)
        dbconn.commit()
        dbconn.close()
    
    def to_db(self, data, table_nm):
        df = pandas.DataFrame(data)
        df.to_sql(name=table_nm, con=self.engine, if_exists='append', index=False)
        
    def from_db(self, query):
        dbconn = _mysql.connect(host=DB_HOST, user=USER_NAME, passwd=USER_PSWD, port=PORT, db=self.db_name)
        self.cursor = dbconn.cursor()
        self.cursor.execute(query)
        table = self.cursor.fetchall()
        col_names = {i: nm[0] for i, nm in enumerate(self.cursor.description)}
        dbconn.commit()
        dbconn.close()
        return pandas.DataFrame(table).rename(columns = col_names)