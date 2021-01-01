import pymysql
import pandas as pd
from sqlalchemy.engine import create_engine
import pdb

class MySQL:
    def __init__(self,host,port,user,password,db_name=None):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.user = user
        self.password = password


    def connect(self):
        if self.db_name is None:
            conn = pymysql.connect(self.host, user=self.user, port=self.port,
                                   passwd=self.password)
        else:
            conn = pymysql.connect(self.host, user=self.user, port=self.port,
                                   passwd=self.password, db=self.db_name)
        return conn

    def get_data_from_RDS(self,query):
        outputData = pd.read_sql(query, self.connect())
        return outputData

    def call_procedure(self,procedure_name):
        cursor = self.connect().cursor()
        cursor.callproc(self.db_name+'.'+procedure_name)


    def update_table(self,tablename,update_col,update_val,condition):
        conn = self.connect()
        cursor = conn.cursor()
        sql_statement = "UPDATE " + self.db_name + "." + tablename + " SET " + update_col + "= '" + update_val + "' where " + condition
        print(sql_statement)
        cursor.execute(sql_statement)
        conn.commit()
        conn.close()
        print('record updated')

    def append_insert(self,table_name,df,type='replace'):
        conn_str = 'mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}'
        engine = create_engine(conn_str.format(
            port=self.port,
            user=self.user,
            password=self.password,

            host=self.host,
            database_name=self.db_name))
        df.to_sql(name=table_name, con=engine, if_exists=type, index=False)

    def upsert_data(self,table_name,col_val_mapper,update_col_values=None):
        conn = self.connect()
        cursor = conn.cursor()
        query_maker = """
        INSERT INTO """+table_name

        queries_to_execute = []
        col_to_insert = "("
        values_to_insert = "("
        for colname,values in col_val_mapper:
            col_to_insert += colname + ","
            values_to_insert += "%s,"
        col_to_insert = col_to_insert[:-1] + ")"
        values_to_insert = values_to_insert[:-1] + ")"
        if update_col_values is not None:
            pass

        cursor.execute(query_maker + col_to_insert + " VALUES " + values_to_insert + " ON DUPLICATE KEY UPDATE ")

        conn.commit()
        conn.close()


    def replace_data(self,tableName,df_to_rw,condition_cols):

        for item, rows in df_to_rw[condition_cols].iterrows():
            conditionMapper = ''
            for col in condition_cols:
                if conditionMapper != '':
                    conditionMapper = conditionMapper + " and " + col + ' = "' + str(rows[col]) + '"'
                else:
                    conditionMapper = col + ' = "' + str(rows[col]) + '"'
            self.execute_query(f"Delete from {self.db_name}.{tableName} where "+ conditionMapper)
        conn_str = 'mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}'
        engine = create_engine(conn_str.format(
            port=self.port,
            user=self.user,
            password=self.password,
            host=self.host,
            database_name=self.db_name))
        df_to_rw.to_sql(name=tableName, con=engine, if_exists='append', index=False)
        print('Data Replacement Completed!!!')

    def execute_query(self,query):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()