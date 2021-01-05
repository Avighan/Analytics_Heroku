import pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler, \
    Binarizer, FunctionTransformer, Normalizer, OrdinalEncoder, PowerTransformer, QuantileTransformer, LabelBinarizer, \
    MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
import re
import sys


def input_assigner(kwargs, key, optional=True, defaultSetting=None):
    input_data = {}
    if key in kwargs.keys():
        input_data[key] = kwargs[key]
    else:
        if optional == True:
            input_data[key] = defaultSetting
        else:
            print(f'{key} is minimum requirement to be executed!!!')
            sys.exit()


class Sampling:

    def __init__(self, df, target,sampling_option='Over'):
        self.Y = df[[target]]
        self.X = df[[col for col in df.columns if col != target]]
        self.sampling_option = sampling_option

    def oversample(self, **kwargs):
        oversample = SMOTE(**kwargs)
        X, Y = oversample.fit_resample(self.X, self.Y)
        return X, Y

    def RandomOverSample(self,**kwargs):
        randoversample = RandomOverSampler(**kwargs)
        X, Y = randoversample.fit_resample(self.X, self.Y)
        return X, Y

    def undersample(self, **kwargs):
        undersample = NearMiss(**kwargs)
        X, Y = undersample.fit_resample(self.X, self.Y)
        return X, Y

    def run_sampler(self,**kwargs):
        if self.sampling_option == 'Over':
            return self.oversample(**kwargs)
        elif self.sampling_option == 'RandomOverSampler':
            return self.RandomOverSample(**kwargs)
        else:
            return self.undersample(**kwargs)

class FeatureSelection:

    def __init__(self, **kwargs):

        self.selection_option = {
            'SelectKBest': SelectKBest,
            'RFE': RFE,
            'PCA': PCA,
            'LDA': LatentDirichletAllocation
        }
        if 'selectMethod' not in kwargs.keys():
            self.featureSelect = kwargs['selectMethod']
            self.fs = self.selection_option[self.featureSelect]

    def feature_selection_option(self, selectMethod=None):
        if selectMethod is None:
            self.fs = self.selection_option[self.featureSelect]
        else:
            self.fs = self.selection_option[selectMethod]
        return self.fs

    def set_params(self, **kwargs):
        self.fs(**kwargs)

    def fit(self, **kwargs):
        return self.selection_option.fit(self.df, **kwargs)

    def transform(self, **kwargs):
        return self.selection_option.transform(self.df, **kwargs)

    def fit_transform(self, **kwargs):
        return self.selection_option.fit_transform(self.df, **kwargs)


class Preprocessing:
    get_outlier_data = pd.DataFrame()

    def __init__(self, data, **kwargs):
        self.outlier_detection_methods = {'default': self.default_outlier_method,
                                          'LocalOutlierFactor': LocalOutlierFactor}
        if 'select_method' not in kwargs.keys or kwargs['select_method'] is None:
            self.selected_method = 'default'
        else:
            self.selected_method = kwargs['select_method']
        self.data = data
        self.outlier_to_run = self.outlier_detection_methods[self.selected_method]

    def run_outier(self, series_data, **kwargs):
        outlier_detector = self.outlier_to_run(**kwargs)
        outlier_mask = outlier_detector.fit_predict(series_data)
        return {'outliermask': outlier_mask}

    def default_outlier_method(self, series_data, **kwargs):
        outlier_k = kwargs['k']
        q25, q75 = np.percentile(series_data, 25), np.percentile(series_data, 75)
        iqr = q75 - q25

        cut_off = iqr * outlier_k
        lower, upper = q25 - cut_off, q75 + cut_off

        return {'outliermask': [True if x < lower or x > upper else False for x in series_data]}

    def compile(self, **kwargs):
        for col in self.data.columns:
            self.get_outlier_data[col + '_' + 'outlier'] = self.outlier_to_run(**kwargs)['outliermask']

        return self.get_outlier_data


class Imputers:

    def __init__(self, dataframe, select_imputer=None):
        self.df = dataframe
        if select_imputer is not None:
            self.select_imputer = select_imputer
        else:
            self.select_imputer = None
        self.imputers = {
            'SimpleImputer': SimpleImputer
        }

    def select_imputers(self, imputerSelect, **kwargs):
        print(self.imputers[imputerSelect])
        if self.imputers is None:
            if len(kwargs) > 0:
                self.imputer = self.imputers[imputerSelect](**kwargs)
            else:
                self.imputer = self.imputers[imputerSelect]()
        else:
            if len(kwargs) > 0:
                self.imputer = self.imputers[self.select_imputer](**kwargs)
            else:
                self.imputer = self.imputers[self.select_imputer]()
        return self.imputer

    def fit(self, **kwargs):
        return self.imputer.fit(self.df, **kwargs)

    def transform(self, **kwargs):
        return self.imputer.transform(self.df, **kwargs)

    def fit_transform(self, **kwargs):
        return self.imputer.fit_transform(self.df, **kwargs)


class OneHotEncoding:
    count = 0

    def __init__(self, **kwargs):
        self.df = input_assigner(kwargs, 'df', defaultSetting=None)
        self.column = input_assigner(kwargs, 'col', defaultSetting=None)
        self.encoded_df = None
        self.encoded_class = {}

    def fit(self, column=None):
        if column is None:
            col = self.df[self.column]
        else:
            col = column
        self.one_hot = pd.get_dummies
        return self

    def transform(self, column=None):
        if column is None:
            col = self.df[self.column]
        else:
            col = column
        transformed_df = self.one_hot(col)
        return transformed_df

    def fit_transform(self, column=None):
        if column is None:
            col = self.df[self.column]
        else:
            col = column
        transformed_df = pd.get_dummies(col)
        return transformed_df

    def inverse_transform(self, col, encoder=None):
        if encoder is None:
            inv_df = self.one_hot(col).idxmax(axis=1)
            return inv_df
        else:
            inv_df = encoder(col).idxmax(axis=1)
            return inv_df


class Encoders:

    def __init__(self, **kwargs):
        self.encoded_values = {}
        self.df = kwargs['df'].copy(deep=True)
        self.encoders = {
            'LabelEncoder': LabelEncoder,
            'OneHotEncoder': OneHotEncoding,
            'OrdinalEncoder': OrdinalEncoder,
            'Binarizer': Binarizer,
            'LabelBinarizer': LabelBinarizer,
            'MultiLabelBinarizer': MultiLabelBinarizer
        }

        if 'cat_columns' not in kwargs.keys():
            self.cat_columns = [cat_col for cat_col in self.df.columns if self.df[cat_col].dtype == object]
            self.num_columns = [col for col in self.df.columns if col not in self.cat_columns]
        else:
            self.cat_columns = [cat_col for cat_col in kwargs['cat_columns'] if self.df[cat_col].dtype == object]
            self.num_columns = [col for col in self.df.columns if col not in kwargs['cat_columns']]

        if 'column_wise_encoding_dict' in kwargs.keys():
            self.column_wise_encoding_dict = kwargs['column_wise_encoding_dict']
        else:
            self.column_wise_encoding_dict = None
        if 'encoding_type' in kwargs.keys():
            if kwargs['encoding_type'] is not None:
                self.encoding_type = kwargs['encoding_type']
            else:
                self.encoding_type = 'OneHotEncoder'
            self.encoder = self.select_encoder()
        else:
            self.encoding_type = 'OneHotEncoder'
        self.y = kwargs['y']
        self.encoding_stack = {}  # 'encoder':{},'encoded_data':{},'decoded_data':{}

    def select_encoder(self, encode_type=None, **kwargs):

        if encode_type is not None:
            self.encoding_type = encode_type
        else:
            pass
        self.encoder = self.encoders[self.encoding_type](**kwargs)
        return self.encoder

    def add_to_encoding_stack(self, columnName, encoder, encoded_values, columns_after_encoding, decoded_values,
                              decoding_column_name):
        if columnName not in self.encoding_stack.keys():
            self.encoding_stack[columnName] = {}
        else:
            pass
        self.encoding_stack[columnName]['encoder'] = encoder
        self.encoding_stack[columnName]['encoded_values'] = encoded_values
        self.encoding_stack[columnName]['columns_after_encoding'] = columns_after_encoding
        self.encoding_stack[columnName]['decoded_values'] = decoded_values
        self.encoding_stack[columnName]['decoding_column_name'] = decoding_column_name

    def set_encoder_parameters(self, **kwargs):
        self.encoder = self.encoder(**kwargs)
        return self.encoder

    def fit(self, x, **kwargs):
        fitted_encoded = self.encoder.fit(x, **kwargs)
        return fitted_encoded

    def transform(self, col, x, **kwargs):
        return self.encoded_values[col].transform(x, **kwargs)

    def inverse_transform(self, col, x):
        return self.encoded_values[col].inverse_transform(x)

    def compile_encoding(self):
        encode_df = pd.DataFrame()

        if self.column_wise_encoding_dict is None:

            if self.encoding_type is None:
                print('Encoder not defined!!!')
            for col in self.df.columns:
                if col in self.cat_columns:

                    if self.encoding_type != 'OneHotEncoder':
                        self.encoded_values[col] = self.fit(self.df[col].astype(str))
                        encode_df[col] = self.transform(col, self.df[col].astype(str))
                    else:
                        if col != self.y:
                            self.encoder = self.select_encoder('OneHotEncoder')
                            self.encoded_values[col] = self.fit(x=self.df[col].astype(str))
                            transformed_df = self.transform(col=col, x=self.df[col].astype(str))
                            decoded_df = self.inverse_transform(col=col, x=transformed_df)
                            transformed_df.columns = [col + '_' + str(s) for s in transformed_df.columns]
                            self.df.drop(col, axis=1, inplace=True)
                            encode_df = pd.concat([self.df, transformed_df], axis=1)
                            self.add_to_encoding_stack(col, self.encoded_values[col], transformed_df,
                                                       transformed_df.columns,
                                                       decoded_df, col)
                        elif col == self.y:
                            sent_encoder = self.encoding_type
                            encoder_default_y = self.select_encoder('LabelEncoder')
                            self.encoded_values[col] = self.fit(self.df[col].astype(str))
                            encode_df[col] = self.transform(col, self.df[col].astype(str))
                            decoded_df = self.inverse_transform(col, encode_df[col])
                            encoder_default_y = self.select_encoder(sent_encoder)
                            self.add_to_encoding_stack(col, self.encoded_values[col], None, None,
                                                       decoded_df, col)
                else:
                    encode_df[col] = self.df[col]
        else:
            for encoder, columns in self.column_wise_encoding_dict.items():
                if encoder != 'OneHotEncoder':
                    for column in columns:
                        self.encoder = self.select_encoder(encoder)
                        self.encoded_values[column] = self.fit(self.df[column].astype(str))
                        encode_df[column] = self.transform(column, self.df[column].astype(str))
                        decoded_df = self.inverse_transform(column, encode_df[column])
                        self.add_to_encoding_stack(column, self.encoded_values[column], encode_df,
                                                   encode_df.columns,
                                                   decoded_df, column)
                else:
                    for column in columns:
                        if column != self.y:
                            self.encoded_values[column] = self.fit(self.df[column].astype(str))
                            encode_df[column] = self.transform(column, self.df[column].astype(str))
                            decoded_df = self.inverse_transform(column, encode_df[column])
                            self.add_to_encoding_stack(column, self.encoded_values[column], encode_df,
                                                       encode_df.columns,
                                                       decoded_df, column)
                        elif column == self.y:
                            sent_encoder = self.encoding_type
                            encoder_default_y = self.select_encoder('LabelEncoder')
                            self.encoded_values[column] = self.fit(self.df[column].astype(str))
                            encode_df[column] = self.transform(column, self.df[column].astype(str))
                            decoded_df = self.inverse_transform(column, encode_df[column])
                            encoder_default_y = self.select_encoder(sent_encoder)
                            self.add_to_encoding_stack(self, column, self.encoded_values[column], encode_df,
                                                       encode_df.columns,
                                                       decoded_df, column)

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        encode_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                             encode_df.columns.values]

        return encode_df

    def get_decoding(self, decode_df):
        for col in self.cat_columns:
            decode_df[col] = self.inverse_transform(col, decode_df[col])
        return decode_df

    def get_encoded_df(self):
        return self.encoder

    def get_encoder_list(self):
        return self.encoders

    def get_encoded_stack(self):
        return self.encoding_stack


class scaling:

    def __init__(self, **kwargs):
        self.scalar_stack = {}
        self.scalar_values = {}
        self.df = kwargs['df']
        if 'cat_columns' not in kwargs.keys():
            self.cat_columns = [cat_col for cat_col in self.df.columns if self.df[cat_col].dtype == object]
            self.num_columns = [col for col in self.df.columns if col not in self.cat_columns]
        else:
            self.cat_columns = [cat_col for cat_col in kwargs['cat_columns'] if self.df[cat_col].dtype == object]
            self.num_columns = [col for col in self.df.columns if col not in kwargs['cat_columns']]

        if 'column_wise_scalar_dict' in kwargs.keys():
            self.column_wise_scalar_dict = kwargs['column_wise_scalar_dict']
        else:
            self.column_wise_scalar_dict = None
        if 'scalar_type' in kwargs.keys():
            self.scalar_type = kwargs['scalar_type']
        else:
            self.scalar_type = 'StandardScaler'
        self.y = kwargs['y']
        self.scalars = {'StandardScaler': StandardScaler,
                        'MaxAbsScaler': MaxAbsScaler,
                        'MinMaxScaler': MinMaxScaler,
                        'RobustScaler': RobustScaler,
                        'Normalizer': Normalizer,
                        'PowerTransformer': PowerTransformer,
                        'QuantileTransformer': QuantileTransformer
                        }

    def select_scalar(self, scalar_type=None, **kwargs):
        if scalar_type is None:
            self.scalar_type = 'StandardScaler'
        else:
            self.scalar_type = scalar_type

        self.scalar = self.scalars[self.scalar_type](**kwargs)
        return self.scalar

    def set_scalar_parameters(self, **kwargs):
        self.scalar = self.scalar(**kwargs)
        return self.scalar

    def fit(self, x):

        fitted_scalar = self.scalar.fit(x)

        return fitted_scalar

    def transform(self, col, x):
        return self.scalar_values[col].transform(x)

    def inverse_transform(self, col, x):
        return self.scalar_values[col].inverse_transform(x)

    def fit_transform(self, col, x):
        return self.scalar_values[col].fit_transform(x)

    def add_to_scaling_stack(self, columnName, scaler, scaled_values, inverse_scaled_values):
        if columnName not in self.scalar_stack.keys():
            self.scalar_stack[columnName] = {}
        else:
            pass
        self.scalar_stack[columnName]['scaler'] = scaler
        self.scalar_stack[columnName]['scaled_values'] = scaled_values
        self.scalar_stack[columnName]['inverse_scaled_values'] = inverse_scaled_values

    def compile_scalar(self):
        # scalar_df = pd.DataFrame()
        scalar_df = self.df[:]
        if self.column_wise_scalar_dict is None:
            for col in self.df.columns:
                if col in self.num_columns:
                    if self.y != col:
                        self.scalar = self.select_scalar(self.scalar_type)
                        self.scalar_values[col] = self.fit(self.df[[col]])
                        scalar_df[[col]] = self.transform(col, self.df[[col]])
                        inversed_scaled = self.inverse_transform(col, scalar_df[[col]])
                        self.add_to_scaling_stack(col, self.scalar_values[col], scalar_df[[col]], inversed_scaled)
                    else:
                        pass
                else:
                    scalar_df[col] = self.df[col]
        else:
            for scalar, columns in self.column_wise_encoding_dict.items():
                for col in columns:
                    self.scalar = self.select_scalar(self.scalar_type)
                    self.scalar_values[col] = self.fit(self.df[[col]])
                    scalar_df[[col]] = self.transform(col, self.df[[col]])
                    inversed_scaled = self.inverse_transform(col, scalar_df[[col]])
                    self.add_to_scaling_stack(col, self.scalar_values[col], scalar_df[[col]], inversed_scaled)
        return scalar_df

    def get_scalar_list(self):
        return self.scalars

    def get_scalar_stack(self):
        return self.scalar_stack