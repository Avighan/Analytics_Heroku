import streamlit as st
import pandas as pd
import numpy as np
import Data_Preprocessing as dp
import ML_models as ml_models
import pickle
import ML_Metrics as ml_metrics
import pdb


def get_data(data,mongocls,session_id):
    try:
        if len(pd.DataFrame(mongocls.get_session_info({'session_id': session_id + '_ml_df'})["data_dict"])) > 0:
            data = pd.DataFrame(mongocls.get_session_info({'session_id':session_id+'_ml_df'})["data_dict"])
            return data
    except:
        return data


def run(st,data,mongocls,session_id):
    st.markdown("### Steps -")

    view_sample_data = st.checkbox("View Sample Data - ")
    if view_sample_data:

        if mongocls.get_session_info({'session_id': session_id + '_ml_df'}) is not None:
            st.dataframe(pd.DataFrame(mongocls.get_session_info({'session_id':session_id+'_ml_df'})["data_dict"]).head())
        else:
            st.dataframe(data.head())

    c_load,c_session,c_restrict_columns = st.beta_columns(3)
    if c_load.button("Reload Data!!!"):
        data = get_data(data,mongocls,session_id)


    if c_session.button("Reset Session!!!"):
        mongocls.delete_session({'session_id': session_id + '_ml_df'})

    if c_restrict_columns.checkbox("Restrict Columns: "):
        restrict_columns = st.multiselect("Restrict Columns ", data.columns.tolist())
        data = data[restrict_columns]

    expander = st.beta_expander("Data Preparation:",expanded=False)


    data = get_data(data,mongocls,session_id)
    columns = data.columns.tolist()
    with expander:
            c1,c2,c3 = st.beta_columns(3)
            fs = c1.checkbox("Feature Selection")
            if fs:
                st.text("Currently under development!!!")
                #fs_option = st.selectbox("Selection Option",['SelectKBest','RFE','PCA','LDA'])

            dimpute = c2.checkbox("Data Imputers")
            if dimpute:
                impute_options = st.selectbox("Impute Option",['SimpleImputer'])
                if impute_options:

                    imputer = dp.Imputers(dataframe=data,select_imputer=impute_options)
                    imputer.select_imputers(imputerSelect = impute_options)
                    data = imputer.fit_transform()
                    data = pd.DataFrame(data, columns=columns)
                    mongocls.delete_session({'session_id': session_id + '_ml_df'})
                    mongocls.write_session_info(
                        {'session_id': session_id + '_ml_df', 'data_dict': data.to_dict("records")})
                    st.dataframe(data.head())
    expander_enc = st.beta_expander("Data Encoding", expanded=False)
    with expander_enc:
        encode = st.checkbox("Apply Encoding")
        if encode:
            c1_encode,c2_encode,c3_encode = st.beta_columns(3)
            encoding_option = c1_encode.selectbox("Select Encoder",['LabelEncoder','OneHotEncoder','OrdinalEncoder','Binarizer','LabelBinarizer','MultiLabelBinarizer'])
            Y_col_encode = c2_encode.selectbox("Select Y (target) (Encoding)", data.columns.tolist())
            cat_columns =  c3_encode.multiselect("Select Categorical Columns to Encode",data.columns.tolist())
            encode_btn = st.button("Encode Data!!!")
            if encode_btn:
                if len(cat_columns)==0:
                    encode_cls = dp.Encoders(df=data,y=Y_col_encode)
                else:
                    encode_cls = dp.Encoders(df=data,y=Y_col_encode, cat_columns = cat_columns)
                encode_cls.select_encoder(encode_type=encoding_option)
                data = encode_cls.compile_encoding()
                st.dataframe(data.head())
                mongocls.delete_session({'session_id': session_id + '_ml_df'})
                mongocls.write_session_info(
                    {'session_id': session_id + '_ml_df', 'data_dict': data.to_dict("records")})

    expander_sample = st.beta_expander("Data Sampling", expanded=False)
    with expander_sample:
        c1, c2, c3, c4 = st.beta_columns(4)
        sample_options = c1.selectbox("Sampling Options", ["Over", "Under","RandomOverSampler"])
        sampling_ratio = c2.slider('Sampling Ratio', min_value=0.1, max_value=1.0, step=0.05)
        Y_col = c3.selectbox("Select Y (target) (Sampling)", data.columns.tolist())
        if Y_col != '':
            X_cols = c4.multiselect("Select X Columns (default is all)",
                                    [col for col in data.columns.tolist() if col != Y_col])
            if len(X_cols) <= 0:
                X_cols = [col for col in data.columns.tolist() if col != Y_col]
                X_val = data[X_cols]

            X_val = data[X_cols]
            Y_val = data[Y_col]
        else:
            st.warning("Please select Target column!!!")
        sampler_btn = st.button("Run Sampler")
        if sampler_btn:
            sample_cls = dp.Sampling(df=data[X_cols + [Y_col]], target=Y_col, sampling_option=sample_options)
            X_val, Y_val = sample_cls.run_sampler()
            data = (pd.concat([X_val, Y_val], axis=1))
            st.dataframe(data.head())
            mongocls.delete_session({'session_id': session_id + '_ml_df'})
            mongocls.write_session_info(
                {'session_id': session_id + '_ml_df', 'data_dict': data.to_dict("records")})

    expander_scale = st.beta_expander("Data Scaling", expanded=False)
    with expander_scale:
        scale = st.checkbox("Apply Scaling")
        if scale:
            c1, c2, c3 = st.beta_columns(3)
            scaling_options = c1.selectbox("Sampling Options", ["StandardScaler", "MaxAbsScaler", "MinMaxScaler","RobustScaler",
                                                                "Normalizer","PowerTransformer","QuantileTransformer"])
            Y_col = c2.selectbox("Select Y (target) (Scaling)", data.columns.tolist())
            cat_columns = c3.multiselect("Categorical Columns",[col for col in data.columns.tolist() if col != Y_col])
            scale_btn = expander_scale.button("Scale Data!!!")
            if scale_btn:
                scaling = dp.scaling(df=data,cat_columns=cat_columns,scalar_type=scaling_options,y=Y_col)
                data = scaling.compile_scalar()
                st.dataframe(data.head())
                mongocls.delete_session({'session_id': session_id + '_ml_df'})
                mongocls.write_session_info(
                    {'session_id': session_id + '_ml_df', 'data_dict': data.to_dict("records")})
                pass

    expander_model = st.beta_expander("Model Training", expanded=False)
    with expander_model:
        c1,c2,c3 =  st.beta_columns(3)
        model_type = c1.selectbox("Model Type",['Regression','Classification'])
        y_col = c2.selectbox("Select Target Variable",data.columns.tolist())
        model_exe = ml_models.MLmodels(df=data,y_column=y_col,problem_type=model_type)
        select_models = c3.multiselect("Select ML models",model_exe.get_model_list())
        run_models = st.button("Run Models - ")
        model_storage = {}
        if run_models:
            for model in select_models:
                model_exe.select_model_to_run(model_select=model)
                model_storage[model]= model_exe.compile_modeling()
                train_x, train_y, test_x, test_y = model_exe.get_train_test()
            mongocls.delete_session({'session_id': session_id + '_models_ran'})
            mongocls.write_session_info(
                {'session_id': session_id + '_models_ran','models_trained': pickle.dumps(model_storage),'train_test_data':pickle.dumps(
                    {
                        'train_x':train_x,
                        'train_y':train_y,
                        'test_x':test_x,
                        'test_y':test_y
                    }
                )})
        if len(model_storage.keys())>0:
            st.write("Model Run completed for the below models - ")
            st.code(model_storage)

    expander_metrics = st.beta_expander("Evaluation Metrics", expanded=False)
    with expander_metrics:
        #try:
            models_trained = pickle.loads(mongocls.get_session_info({'session_id': session_id + '_models_ran'})["models_trained"])
            loaded_info = pickle.loads(mongocls.get_session_info({'session_id': session_id + '_models_ran'})["train_test_data"])
            train_x = loaded_info['train_x']
            train_y = loaded_info['train_y']
            test_x = loaded_info['test_x']
            test_y = loaded_info['test_y']
            metric_selected = {}
            for model,trained_model in models_trained.items():
                st.text(model)
                metric_cls = ml_metrics.Metrics(y_test=test_y)
                metric_selected[model] = st.multiselect('Select Metrics to see for the Model ('+model+')',metric_cls.get_metric_list())
            metrics_btn = st.button("Click to see the metrics")
            if metrics_btn:
                for model,metrics in metric_selected.items():
                    for metric in metrics:
                        pdb.set_trace()
                        metric_cls.select_metrics(metric)
                        st.write(metric)
                        st.write(metric_cls.metrics_solve(estimator=models_trained[model], test_x=test_x))

        #except:
        #    st.warning("No Models trained yet!!!")