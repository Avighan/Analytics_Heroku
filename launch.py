#https://cloud.mongodb.com/v2/5fef13ddbf15a760de263a87#security/network/accessList

import pdb
import mysql as mysql
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_theme as stt
from streamlit.report_thread import get_report_ctx
import  mongo_Sessions_storage as mongo
import ml_modeling as ml
import psycopg2
from sqlalchemy import create_engine
import data_sources as ds
import analytics as ana


def get_session_id():
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id
    return session_id

#Creating PostgreSQL client
#engine = create_engine('postgresql://postgres:admin@localhost:5432/Sessions')

#Creating MongoDB Client - Local Client
mongoSession = mongo.MongoDB(session_info='AnalyticsSession',host="mongodb://localhost:27017/")

#Creating MongoDB Client - My Cloud Client
#mongoSession = mongo.MongoDB(session_info='AnalyticsSession',host="mongodb://localhost:27017/")
mongoSession = mongo.MongoDB(session_info='AnalyticsSession',host="mongodb+srv://admin:admin@cluster0.9kpyj.mongodb.net/<dbname>?retryWrites=true&w=majority")

#Getting session ID
session_id = get_session_id()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Analytics',
                        page_icon='',
                        layout='wide',
                        initial_sidebar_state='expanded')

container = st.sidebar.beta_container()
initial_options = container.selectbox("Pages",["Load(Connect) Data","Analysis/Reporting Data","ML Modeling","Simulation"])
if initial_options == 'Load(Connect) Data':
    data = ds.run(st, mongoSession, session_id)
elif initial_options == 'Analysis/Reporting Data':
    if len(pd.DataFrame(mongoSession.get_session_info({'session_id':session_id+'_df'})["data_dict"]))>0:
        data = pd.DataFrame(mongoSession.get_session_info({'session_id':session_id+'_df'})["data_dict"])
        ana.run(st,data)
elif initial_options == "ML Modeling":
    if len(pd.DataFrame(mongoSession.get_session_info({'session_id': session_id + '_df'})["data_dict"])) > 0:
        data = pd.DataFrame(mongoSession.get_session_info({'session_id': session_id + '_df'})["data_dict"])
        ml.run(st, data)
elif initial_options == 'Simulations':
    pass
