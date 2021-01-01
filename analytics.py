import pdb
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import Statistics as stats
import base64
from pandas_profiling import ProfileReport
import os
from streamlit_pandas_profiling import st_profile_report

def run(st,data):
    expander = st.beta_expander("Menu")
    with expander:
        ana_choice = st.radio("Analysis",["Data","Visualization","Statistics","Data Profiling"])
        if ana_choice == 'Data':
            data_options = st.selectbox("",["View Records","Data Correlation","Pivot"])
            if data_options == "View Records":
                c1,c2 = st.beta_columns(2)
                top_bottom_options = c1.radio("Records",["Top","Bottom"])
                num_rec = c2.number_input("No. of Records:", min_value=0, max_value=100, step=1, value=10)
                if top_bottom_options == 'Top':
                    st.table(data.head(num_rec))
                else:
                    st.table(data.tail(num_rec))
            elif data_options == "Data Correlation":
                select_columns = st.multiselect("Select Columns",data.columns.tolist())
                corr_view = st.radio("Correlation View",["Table","Chart"])
                if corr_view == 'Table':
                    if len(select_columns)==0:
                        st.table(data.corr())
                    else:
                        st.table(data[select_columns].corr())
                else:
                    if len(select_columns) == 0:
                        st.write(sns.heatmap(data.corr(), annot=True))
                        st.pyplot()
                    else:
                        st.write(sns.heatmap(data[select_columns].corr(), annot=True))
                        st.pyplot()
            elif data_options == 'Pivot':
                dimensions = st.multiselect("Select X axis columns",data.columns.tolist())
                measures = st.multiselect("Select Y axis columns", data.columns.tolist())
                numeric_cols = st.multiselect("Aggregation columns", data.columns.tolist())
                aggregation_operations = st.selectbox("Aggregation Operation",['sum','average','median','count'])
                button = st.button("Execute!!!")
                if button:
                    if len(numeric_cols) > 0 :
                        if aggregation_operations == 'sum':
                            operation = np.sum
                        elif aggregation_operations == 'average':
                            operation = np.mean
                        elif aggregation_operations == 'median':
                            operation = np.median
                        elif aggregation_operations == 'count':
                            operation = np.count_nonzero
                        pivot_table = pd.pivot_table(data,values=numeric_cols,index=measures,columns=dimensions,aggfunc=operation)
                        st.table(pivot_table)
        elif ana_choice == "Visualization":
            chart_options = st.selectbox('Charts',['Bar','Line','Heatmap','Distplot','Customized'])
            if chart_options == 'Bar':
                x_col = st.selectbox('X',data.columns.tolist())
                y_col = st.selectbox('Y', data.columns.tolist())
                hue_color = st.checkbox("Add color column")
                direction = st.radio('chart direction',['vertical','horizontal'])
                button = st.button("Execute!!!")
                if button:
                    if direction == 'vertical':
                        chart_direction = 'v'
                    else:
                        chart_direction = 'h'
                    if hue_color:
                        hue_col = st.selectbox('hue', data.columns.tolist())
                        if hue_col:
                            st.write(sns.barplot(x=x_col, y=y_col, hue=hue_col, data=data,orient=chart_direction))
                            st.pyplot()
                        else:
                            st.write(sns.barplot(x=x_col, y=y_col, data=data,orient=chart_direction))
                            st.pyplot()
                    else:
                        st.write(sns.barplot(x=x_col, y=y_col, data=data,orient=chart_direction))
                        st.pyplot()
            elif chart_options == 'Line':
                x_col = st.selectbox('X', data.columns.tolist())
                y_col = st.selectbox('Y', data.columns.tolist())
                hue_color = st.checkbox("Add color column")
                if hue_color:
                    hue_col = st.selectbox('hue', data.columns.tolist())
                button = st.button("Execute!!!")
                if button:
                    if hue_color:
                        if hue_col:
                            st.write(sns.lineplot(x=x_col, y=y_col, hue=hue_col, data=data))
                            st.pyplot()
                        else:
                            st.write(sns.lineplot(x=x_col, y=y_col, data=data))
                            st.pyplot()
                    else:
                        st.write(sns.lineplot(x=x_col, y=y_col, data=data))
                        st.pyplot()
            elif chart_options == 'Heatmap':
                select_columns = st.multiselect("Select Columns", data.columns.tolist())
                button = st.button("Execute!!!")
                if button:
                    if len(select_columns) == 0:
                        st.write(sns.heatmap(data, annot=True))
                        st.pyplot()
                    else:
                        st.write(sns.heatmap(data[select_columns], annot=True))
                        st.pyplot()
            elif chart_options == 'Distplot':
                x_col = st.selectbox('X', data.columns.tolist())
                col = st.selectbox('column', data.columns.tolist())
                row = st.selectbox('row', data.columns.tolist())
                button = st.button("Execute!!!")
                if button:
                    st.write(sns.displot(
                        data, x=x_col, col=col, row=row,
                        binwidth=3, height=3, facet_kws=dict(margin_titles=True),
                    ))
                    st.pyplot()
            elif chart_options == 'Customized':
                code_area = st.text_area("""Enter your chart script, Return result to value.
                e.g. 
                a = 3
                b = 4
                value = a + b!!!, Don't enter data parameter !!!""")


                button = st.button("Execute!!!")
                if button:
                    loc = {}
                    exec(code_area, {'data':data}, loc)
                    return_workaround = loc['value']
                    st.write(return_workaround)
                    st.pyplot()
        elif ana_choice == 'Statistics':
            test_selection = st.selectbox('Category',
                                          ['Value Count', 'Normality Test', 'Correlation Test', 'Stationary Test',
                                           'Parametric Test',
                                           'Non Parametric Test'])
            statistics = stats.Statistics(data)
            if test_selection == 'Value Count':
                select_columns = st.selectbox("Select Columns",data.columns.tolist())
                mode = st.radio('Value Counts',['Table','Chart'])
                if mode == 'Table':
                    value_counts = statistics.__get__stats__(select_columns)
                    st.table(value_counts)
                else:
                    value_counts = statistics.__get__stats__(select_columns)
                    st.write(value_counts[:20].plot(kind='barh'))
                    st.pyplot()
            elif test_selection == 'Normality Test':
                st.write("""
                        Tests whether a data sample has a Gaussian distribution. \n
                        H0: the sample has a Gaussian distribution. \n
                        H1: the sample does not have a Gaussian distribution""")

                select_test = st.selectbox('Tests', ['ShapiroWilk', 'DAgostino', 'AndersonDarling'])
                col = st.selectbox('Select Column', data.columns.tolist())
                text_option = st.checkbox('Text')
                chart_option = st.checkbox('Chart')
                if text_option:
                    t,p = statistics.normality_tests(data[col], test_type=select_test)
                    st.write('#### ' + t + " (" + str(p) + ")")
                if chart_option:
                    st.write(sns.kdeplot(x=col,data=data))
                    st.pyplot()

        elif ana_choice == 'Data Profiling':
            st.markdown("""
            ##### The Data Profiling is done automatically using Pandas Profiling tool.\n \n \n \n
            """)
            limited_records = st.checkbox("Execute on Limited Records!!!")
            select_columns = st.multiselect("Select Columns", data.columns.tolist())
            if len(select_columns) == 0:
                cols = data.columns.tolist()
            else:
                cols = select_columns
            if limited_records:
                num_rec = st.number_input("No. of Records:", min_value=0, max_value=1000000, step=1, value=100)
            else:
                num_rec = len(data)
            execute_profiling = st.button('Execute!!!')
            if execute_profiling:
                st.title(f"Pandas Profiling on {num_rec} records")

                report = ProfileReport(data[cols].loc[:num_rec,:], explorative=True)
                st.write(data)
                st_profile_report(report)


