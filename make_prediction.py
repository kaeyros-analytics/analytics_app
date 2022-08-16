#!/usr/bin/env python
# coding: utf-8

# IMPORT NECESSARY LIBRARIES

# In[43]:


import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import scipy.stats as st
from scipy import stats
import math
import missingno as msno
from scipy.stats import norm, skew
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn import model_selection
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from mlxtend.regressor import StackingCVRegressor

import plotly.offline as pof
pof.init_notebook_mode()

# to ignore warnings
import warnings
warnings.filterwarnings("ignore")

#to see model hyperparameters
from sklearn import set_config
set_config(print_changed_only = False)

# to show all columns
pd.set_option('display.max_columns', 82)


def predictionD(data):
    
    #LOAD DATASETS

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # DATA PROCCESSING

    used_cols =["GarageCars","TotalBsmtSF","GrLivArea","OverallQual", "SalePrice"]

    def outlier_detection_train(df, n, columns):
        rows = []
        will_drop_train = []
        for col in columns:
            Q1 = np.nanpercentile(df[col], 25)
            Q3 = np.nanpercentile(df[col], 75)
            IQR = Q3 - Q1
            outlier_point = 1.5 * IQR
            rows.extend(df[(df[col] < Q1 - outlier_point)|(df[col] > Q3 + outlier_point)].index)
        for r, c in Counter(rows).items():
            if c >= n: will_drop_train.append(r)
        return will_drop_train

    will_drop_train = outlier_detection_train(train_df, 5, train_df.select_dtypes(["float", "int"]).columns)
    train_df.drop(will_drop_train, inplace = True, axis = 0)

    # define target variable
    y_train = train_df["SalePrice"]

    # combine train and test data for convenience
    train_and_test_df = pd.concat([train_df, test_df], axis = 0)
    train_and_test_df = train_and_test_df[used_cols]
    train_and_test_df = train_and_test_df.drop(["SalePrice"], axis = 1)


    train_and_test_df.rename(columns={"LotArea": "LotArea_m2", 'BsmtFinSF1': 'BsmtFinSF1_m2',"TotalBsmtSF": "TotalBsmtSF_m2", "GrLivArea":"GrLivArea_m2"}, inplace=True)
    #train_and_test_df['OverallCond'] = train_and_test_df['OverallCond'].map({1:"Very Poor", 2:"Poor", 3:"Fair", 4:"Below Average", 5:"Average",
                                                 #6:"Above Average", 7:"Good", 8:"Very Good", 9:"Excellent", 10: "Very Excellent"})
    #train_and_test_df['OverallQual'] = train_and_test_df['OverallQual'].map({1:"Very Poor", 2:"Poor", 3:"Fair", 4:"Below Average", 5:"Average",


                   # 6:"Above Average", 7:"Good", 8:"Very Good", 9:"Excellent", 10: "Very Excellent"})

    #df2['OverallCond'] = df2['OverallCond'].map({"Very Poor": 1, "Poor": 2, "Fair": 3,"Below Average":4, "Average":5,
                                                #"Above Average":6, "Good":7, "Very Good":8, "Excellent":9, "Very Excellent":10})
    #train_and_test_df['OverallQual'] = train_and_test_df['OverallQual'].map({"Very Poor": 1, "Poor": 2, "Fair": 3,"Below Average":4, "Average":5,
                                                #"Above Average":6, "Good":7, "Very Good":8, "Excellent":9, "Very Excellent":10})

    ### Behaving with null values

    # create dataframes consist of total number and percent of missing data

    number_of_missing_df = train_and_test_df.isnull().sum().sort_values(ascending = False)
    percent_of_missing_df = ((train_and_test_df.isnull().sum() / train_and_test_df.isnull().count())*100).sort_values(ascending = False)

    # combine the dataframes and print

    missing_df = pd.concat([number_of_missing_df,
                            percent_of_missing_df],
                            keys = ["total number of missing data", 'total percent of missing data'],
                            axis = 1)

    #Drop unnecessary variables
    #Here we will drop some variables. Because there are many null values of some of these variables. To keep and use these variables in our model will slow performance of the model and do not give any improvement. So lets do this.

    train_and_test_df = train_and_test_df.drop((missing_df[missing_df["total number of missing data"] > 476]).index, axis = 1)


    #Filling null values
    #We will fill null values of numeric variables with the median of that variable. We will fill null values of categoric variables with the most frequent value of that variable.

    numeric_data = [column for column in train_and_test_df.select_dtypes(["int", "float"])]
    categoric_data = [column for column in train_and_test_df.select_dtypes(exclude = ["int", "float"])]

    for col in numeric_data:
        train_and_test_df[col].fillna(train_and_test_df[col].median(), inplace = True)

    for col in categoric_data:
        train_and_test_df[col].fillna(train_and_test_df[col].value_counts().index[0], inplace = True)

    # we select numeric variables of the dataset
    numeric_data = [column for column in train_and_test_df.select_dtypes(["int", "float"])]

    # we check skew degree of that variables
    vars_skewed = train_and_test_df[numeric_data].apply(lambda x: skew(x)).sort_values()

    # we fix skew with 'log1p' function of numpy

    for var in vars_skewed.index:
        train_and_test_df[var] = np.log1p(train_and_test_df[var])

    train_and_test_df = pd.get_dummies(train_and_test_df, drop_first = True)


    x_train = train_and_test_df[:len(train_df)]
    x_test = train_and_test_df[len(train_df):]

    #Cross validation metrics and setup kfold

    k_fold = KFold(n_splits = 15, random_state = 11, shuffle = True)

    def cv_rmse(model, X = x_train):
        rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring = "neg_mean_squared_error", cv = k_fold))
        return rmse


    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_log_error(y, y_pred, squared = False))

    # 1. Build machine learning models 
    #2. Get cross validation scores of the models
    #3. Stack up the models 

    rf = make_pipeline(RobustScaler(),
                       RandomForestRegressor(n_estimators = 2500, max_depth = 15,
                                             min_samples_split = 6, min_samples_leaf = 6,
                                             random_state = 11))

    #1. Fit the models on full train data
    #2. RMSLE scores of the models on full train data

    rf_model = rf.fit(x_train, y_train)

    #RMSLE score of the rf model on full train data
    rf_score = rmsle(y_train, rf_model.predict(x_train))
    #print("RMSLE score of random forest model on full data:", rf_score)
    #RMSLE score of random forest model on full data: 0.0862494408972592
    mod = rf_model.predict(data)
    return(mod)






