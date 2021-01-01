import pdb
import mysql as mysql
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_theme as stt
from streamlit.report_thread import get_report_ctx

import psycopg2
from sqlalchemy import create_engine

import session_details as session

import data_sources as ds
import analytics as ana

#Things to add
# Data Source
# Analytics
# Data Preparation
# Customized Coding
# Machine Learning
# Prediction
# Simulation
#stt.set_theme({'primary': '#1b3388'})

st.set_page_config(page_title='Analytics',
                        page_icon='',
                        layout='wide',
                        initial_sidebar_state='expanded')


def get_session_id():
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id
    return session_id


#Creating PostgreSQL client
engine = create_engine('postgresql://postgres:admin@localhost:5432/Sessions')

#Getting session ID
session_id = get_session_id()

#Creating session state tables
engine.execute("CREATE TABLE IF NOT EXISTS %s (size text)" % (session_id))
len_table =  engine.execute("SELECT COUNT(*) FROM %s" % (session_id));
len_table = len_table.first()[0]

if len_table == 0:
    engine.execute("INSERT INTO %s (size) VALUES ('1')" % (session_id));

# Creating pages
page = st.sidebar.selectbox('Select page:', ('Page One', 'Page Two'))

if page == 'Page One':
    st.write('Hello world')

elif page == 'Page Two':
    size = st.text_input('Matrix size', session.read_state('size', engine, session_id))
    session.write_state('size', size, engine, session_id)
    size = int(session.read_state('size', engine, session_id))

    if st.button('Click'):
        data = [[0 for (size) in range((size))] for y in range((size))]
        df = pd.DataFrame(data)
        session.write_state_df(df, engine, session_id + '_df')

    if session.read_state_df(engine, session_id + '_df').empty is False:
        df = session.read_state_df(engine, session_id + '_df')
        st.write(df)

"""
container = st.beta_container()
expander = container.beta_expander('Menu')
with expander:
    c1,c2,c3 = expander.beta_columns(3)
    analytics = c1.checkbox("Analytics")
    ml = c1.checkbox("ML")
    pred = c2.checkbox("Prediction")
    simulation = c3.checkbox("Simulation")

if analytics:
    data = ds.run(st)



"""

