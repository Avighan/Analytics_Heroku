import pandas as pd
import streamlit as st
import mysql as mysql
import pdb


@st.cache(persist=True)
def load_data(path,ds_type,**kwargs):
    if ds_type=='csv' or ds_type=='txt':
        df = pd.read_csv(path,**kwargs)
    elif ds_type == 'mysql':
        if kwargs['mode'] == 'Table':
            df = kwargs['class'].get_data_from_RDS("Select * from " + kwargs['database'] + "." + kwargs['table_selected'])
        else:
            df = kwargs['class'].get_data_from_RDS(kwargs['query'])
    return df


def run(st, mongocls,session_id):
    data = None
    st.text("Connect")
    container = st.beta_container()
    ds_selection = container.selectbox("Data Sources", ['csv', 'txt', 'mysql', 'sql', 'parquet','folder'])

    if ds_selection == 'csv' or ds_selection == 'txt':
        upload_path = st.file_uploader('Upload Dataset', type=["csv", "txt"])
        sub_container = st.beta_container()
        c1, c2, c3 = sub_container.beta_columns(3)
        header = c1.text_input("Header")
        separator = c2.text_input("Enter Seperator")

        if header == '':
            header = None
        else:
            header = eval(header)

        if separator == '':
            separator = ','

        kwargs = c3.text_area("Additional Parameters (keyword arg format)")
        if kwargs == '':
            kwargs = None
        else:
            kwargs = eval(kwargs)
        connect_btn = st.button("Execute Query !!!")
        if connect_btn:
            if upload_path is None:
                st.warning("Select upload file!!!")
            else:
                if kwargs is None:
                    data = load_data(path=upload_path, ds_type=ds_selection, header=header, sep=separator)
                    mongocls.delete_session({'session_id':session_id+'_df'})
                    mongocls.write_session_info({'session_id':session_id+'_df','data_dict':data.to_dict("records")})
                else:
                    data = load_data(path=upload_path, ds_type=ds_selection, header=header, sep=separator)
                    mongocls.delete_session({'session_id':session_id+'_df'})
                    mongocls.write_session_info({'session_id':session_id+'_df','data_dict':data.to_dict("records")})
    elif ds_selection == 'parquet':
        upload_path = st.file_uploader('Upload Dataset', type=["parquet"])
        connect_btn = st.button("Execute Query !!!")
        if connect_btn:
            if upload_path is None:
                st.warning("Select upload file!!!")
            else:
                data = load_data(path=upload_path, ds_type=ds_selection)
                mongocls.delete_session({'session_id':session_id+'_df'})
                mongocls.write_session_info({'session_id':session_id+'_df','data_dict':data.to_dict("records")})
    elif ds_selection == 'mysql':
        sub_container = st.beta_container()
        host = sub_container.text_input("Enter Host:", value='localhost')
        user = sub_container.text_input("Enter User:", value='root')
        password = sub_container.text_input("Enter Password:", type='password', value='password')
        port = sub_container.number_input("Enter port:", min_value=0, max_value=999999, step=1, value=3306)
        mysql_cls = mysql.MySQL(host=host, user=user, password=password, port=port)
        database_list = mysql_cls.get_data_from_RDS('SHOW DATABASES;')
        database = st.selectbox("Databases", database_list.Database.tolist())
        mode = st.radio("Mode", ['Table', 'Query'])
        if database != '':
            if mode == 'Table':
                table_lists = mysql_cls.get_data_from_RDS(
                    "SELECT table_name  FROM information_schema.tables  WHERE table_schema = '" + database + "';")
                table_selected = st.selectbox("Tables", table_lists.iloc[:, 0].tolist())
            else:
                query = st.text_area("Enter your query")

            connect_btn = st.button("Execute Query !!!")
            if connect_btn:
                if mode == 'Table':
                    data = load_data(path=None, ds_type=ds_selection,
                                          **{'mode': mode, 'class': mysql_cls, 'database': database,
                                             'table_selected': table_selected})
                    # data = mysql_cls.get_data_from_RDS("Select * from "+database+"."+table_selected)
                    st.success('Query executed successfully!!!')
                else:
                    data = load_data(path=None, ds_type=ds_selection,
                                          **{'mode': mode, 'class': mysql_cls, 'query': query})
                    # data = mysql_cls.get_data_from_RDS(query)
                    st.success('Query executed successfully!!!')
                mongocls.delete_session({'session_id':session_id+'_df'})
                mongocls.write_session_info({'session_id':session_id+'_df','data_dict':data.to_dict("records")})
        else:
            st.warning("Select a Database!!!")

    if data is not None:
        st.text('Data Loaded')
        st.write(data.head())
        #session.write_state_df(data, engine, session_id + '_df')
        return data
    else:
        st.warning('No data!!!')