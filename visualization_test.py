#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd
import numpy as np
import seaborn as sns
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


import plotly.graph_objects as go
import dash
import dash_core_components as dcc
#import dash_html_components as html
from dash import html
from dash import Input,Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas_datareader.data as web
import datetime
from jupyter_dash import JupyterDash


# to ignore warnings
import warnings
warnings.filterwarnings("ignore")

#to see model hyperparameters
from sklearn import set_config
set_config(print_changed_only = False)

# to show all columns
pd.set_option('display.max_columns', 82)

import make_prediction


# In[125]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
df = train_df


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

number_of_missing_df = df.isnull().sum().sort_values(ascending = False)
percent_of_missing_df = ((df.isnull().sum() / df.isnull().count())*100).sort_values(ascending = False)
missing_df = pd.concat([number_of_missing_df,
                        percent_of_missing_df],
                        keys = ["total number of missing data", 'total percent of missing data'],
                        axis = 1)
df = df.drop((missing_df[missing_df["total number of missing data"] > 476]).index, axis = 1)

#df = df[used_cols]

df.rename(columns={"LotArea": "LotArea_m2", 'BsmtFinSF1': 'BsmtFinSF1_m2',"TotalBsmtSF": "TotalBsmtSF_m2", "GrLivArea":"GrLivArea_m2"}, inplace=True)
df['OverallCond'] = df['OverallCond'].map({1:"Very Poor", 2:"Poor", 3:"Fair", 4:"Below Average", 5:"Average",
                                             6:"Above Average", 7:"Good", 8:"Very Good", 9:"Excellent", 10: "Very Excellent"})
df['OverallQual'] = df['OverallQual'].map({1:"Very Poor", 2:"Poor", 3:"Fair", 4:"Below Average", 5:"Average",
                                             6:"Above Average", 7:"Good", 8:"Very Good", 9:"Excellent", 10: "Very Excellent"})


# In[ ]:





# In[126]:


# supporting functions  ( thig function will help our main function in app for plotting graphs)

#function to create plotly table
def Table(dff):  
    
    #data= df[df['HouseStyle']=='1Story'].reset_index()
    #del data['index']
    
    #, inplace=True)
    data = dff.sort_values(by=['SalePrice'], ascending=False)        
    data= data.head(20)
    data=data.reset_index(drop=True)
    data = data.SalePrice 
    fig= go.Figure()
    fig.add_trace(go.Table( columnwidth = [15,6],

            header=dict(values=['<b> House style <b>','<b>Houses<br>prices<b>'],  
                        line_color='black',font=dict(color='black',size= 14),height=30,
                        fill_color='lightskyblue',
                        align=['left','center']),
                cells=dict(values=[data.index,data.values],
                       fill_color='lightcyan',line_color='grey',
                           font=dict(color='black', family="Lato", size=15),
                       align='left')))
    fig.update_layout( 
                      title ={'text': "<b style:'color:blue;'>Top house style </b>", 'font': {'size': 16}},title_x=0.5,
# title_font_family="Times New Roman",
        title_font_color="slategray",margin=dict(l=0, r=0, b=0,t=27))  
    return fig

# the data will be one column Imp["values"]
def donut(data):
    fig=  px.pie(names=data.features , values= data.percentage,hole=.5,color_discrete_sequence=px.colors.sequential.Blues_r,height=300)
    fig.update_layout(
                        autosize=True,legend_orientation="h",
                  legend=dict(x=0.09, y=0., traceorder="normal"),
                        title ={'text': "<b style:'color:blue;'>percentage contribution of each variable "
                                , 'font': {'size': 14}},title_x=0.5,title_y=0.97,
                        title_font_color="slategray",
                        font_color='slategray',
                paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='lightblue' 
     ,margin=dict( b=0,l=12,r=12)
    )
    
    return fig

 


def sale_price_meter(data):
    
    fig= go.Figure()
    fig.add_trace(
        go.Indicator(
            {
            'mode': 'gauge+number', 
            'number': {'font': {'color': '#1C4E80'}}, 
            'gauge':{'bar': {'color': 'skyblue'},'axis':{'range': [None, 556581]}},
             
            #'delta' :{"reference": 100000, "valueformat": ".0f"},
            'value':  data['SalePrice'].mean() 
       #     'title' :{'text': 'Rating', 'font': {'size': 20}}
             })
        )
    fig.update_layout(height=170,width=350,margin=dict(t=50,b=10))
    fig.update_layout(title ={'text': 'Average sale price', 'font': {'size': 20}},title_x=0.51,title_y=0.97,title_font_color='dimgray')
    
    return fig

def location_sorting(df):
    location_count= df.groupby('MSZoning')['HouseStyle'].count()
    new_df = pd.DataFrame({'MSZoning':location_count.index , 'Count':location_count.values})
    new_df.reset_index(drop=True,inplace=True)
    return new_df.sort_values('Count',ascending=False)


def meter(pred_price):

    fig= go.Figure()
    fig.add_trace(
        go.Indicator(
            {
            'mode': 'gauge+number+delta', 
            'number': {'font': {'color': '#1C4E80'}}, 
            'gauge':{'bar': {'color': 'skyblue'},'axis':{'range': [1, 5]}},
        
            'value':  pred_price
       #     'title' :{'text': 'Rating', 'font': {'size': 20}}
             })
        )
    fig.update_layout(height=170,width=350,margin=dict(t=50,b=10))
    fig.update_layout(title ={'text': '90% price estimate', 'font': {'size': 21}},title_x=0.49,title_y=0.97,title_font_color='dimgray')
    
    return fig


# In[127]:


# dash plotly App 


app =JupyterDash(__name__,external_stylesheets=[dbc.themes.SANDSTONE]
            
              , meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])


# In[128]:


#########################################################  Frontend :App Layout  ############################################################################


# In[129]:


#df['GrLivArea_m2'].unique()


# In[130]:


#GarageCars            5.480514
#TotalBsmtSF_m2        6.221942
#GrLivArea_m2         12.472606
#OverallQual          65.308275


# In[131]:


#list_of_house_style=['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf',
       #'2.5Fin','All']
list_of_OverallQual=['Good', 'Above Average', 'Very Good', 'Average', 'Excellent',
       'Below Average', 'Very Excellent', 'Fair', 'Very Poor', 'Poor', 'All']
#list_of_GrLivArea_m2=['Good', 'Above Average', 'Very Good', 'Average', 'Excellent',
       #'Below Average', 'Very Excellent', 'Fair', 'Very Poor', 'Poor']

list_of_garage_cars=[1.0,2.0,3.0,4.0,5.0,0.0, 'All']
list_of_Zone=['RL', 'RM', 'C (all)', 'FV', 'RH', 'All']


# In[132]:


df['TotalBsmtSF_m2'].describe()


# In[133]:


dropdown1=dcc.Dropdown(list_of_OverallQual,placeholder='ALL',id='dropdown1',style= {'background-color': '#F0FFFF'},multi=False,value='All',className='dropup1')
dropdown2=dcc.Dropdown(list_of_garage_cars,placeholder='All',id='dropdown2',style= {'background-color': '#F0FFFF'},multi=False,value='All',className='dropup2')
#dropdown3=dcc.Dropdown(list_of_Zone,placeholder='All',id='dropdown3',style= {'background-color': '#F0FFFF'},multi=False,value='All',className='dropup3')
dropdown3 =  dcc.Input(id="dropdown3", type="number", placeholder="334",min=334, max=3494, step=3)#,multi=False,value='All',className='dropup3')
dropdown4 =  dcc.Input(id="dropdown4", type="number", placeholder="0",min=0, max=3206, step=3)#,multi=False,value='All',className='dropup4')


# In[134]:


card = dbc.Card([
                dbc.CardBody([dcc.Graph(figure={},style={'height': '100px'})
                            ])
                 ],className='mb-2 h-20')
card2 = dbc.Card([dbc.CardHeader(html.H6('Rating')),
                dbc.CardBody([dcc.Graph(figure={},style={'height': '150px'})
                            ])
                 ],className='mb-2')
# card3 = dbc.Card([dbc.CardHeader(html.H6('Rating')),
#                 dbc.CardBody([dcc.Graph(figure={},style={'height': '150px'})
                            #])
#                  ],className='mb-a')
# card3 = dbc.Card([dbc.CardHeader(html.H6('Rating')),
#                 dbc.CardBody([dcc.Graph(figure={},style={'height': '150px'})
                            #])
#                  ],className='mb-c')


# In[135]:


histogram=dbc.Card([
                dbc.CardBody([dcc.Graph(id='hist',figure={},style={'height': '370px'})
                            ])
                 ],className='mb-2',color= '#FCFBFC')

average_SalePrice=dbc.Card([
                dbc.CardBody([dcc.Graph(id='SalePrice',figure={},style={'height': '167px'})
                            ])
                 ],className='mb-1 ')

rating =dbc.Card([
                dbc.CardBody([dcc.Graph(id='rating',figure={},style={'height': '167px'})
                            ])
                 ],className='mb-1 ')
barplot=dbc.Card([
                dbc.CardBody([dcc.Graph(id='bar',figure={},style={'height': '370px'})
                            ])
                 ],className='mr-0 ')

piechart=dbc.Card([
                dbc.CardBody([dcc.Graph(id='pie',figure={},style={'height': '370px'})
                            ])
                 ],className='mb-1 ')
Company_table=dbc.Card([
                dbc.CardBody([dcc.Graph(id='Company_table',figure={},style={'height': '350px'})
                            ])
                 ],className='mb-1 ')


# In[136]:


app.layout = dbc.Container([dbc.Row( dbc.Col(html.H2(children=' Houses Price Prediction',className=" text-center p-1 border  border-primary border-top-0  p-2",
                                                    style = {'color':'white','background-color':'#1C4E80','font-size':'30px'}),width=12)
    
                                    ),dbc.Row(),
                            
                            
                            dbc.Row([dbc.Col(html.H5('Select overall Quality:',style={'font-size': '15px'}),style={'color':'#1C4E80'},className='mb-2  mt-2',width=2),
                                dbc.Col(dropdown1 ,className='mb-2  mt-1',width=2),
                                     dbc.Col(html.H5('Select garages:',style={'font-size': '15px'}),style={'color':'#1C4E80'},className='mb-2  mt-2',width=1),
                                     dbc.Col(dropdown2,className='mb-2  mt-1',width=1),
                                     dbc.Col(html.H5('Select living area :',style={'font-size': '15px'}),style={'color':'#1C4E80'},className='mb-2  mt-2',width=2),
                                     dbc.Col(dropdown3,className='mb-2  mt-1',width=1),
                                     dbc.Col(html.H5('Select cave area :',style={'font-size': '15px'}),style={'color':'#1C4E80'},className='mb-2  mt-2',width=2),
                                     dbc.Col(dropdown4,className='mb-2  mt-1',width=1)
                                    ]),
                            
                            dbc.Row([dbc.Col(histogram,width=8),
                                     #dbc.Col(average_SalePrice,width=2)
                                     dbc.Col(dbc.Row([average_SalePrice,rating]),width=4) 
                                     #dbc.Col(average_SalePrice,width=8)
                                    
                                    ]),
                            
                            dbc.Row([dbc.Col(barplot,width=4),
                                     dbc.Col(piechart,width=4),
                                     dbc.Col(Company_table,width=4)
                                   
                                    ])    
                           ]
                           
                           ,fluid=True,style= {'background-color':'#F1F1F1'})


# In[137]:


########################################### backend callbacks and functions #####################################################


# In[138]:


@app.callback(
    Output('SalePrice', 'figure'),
    [Input('dropdown1', 'value'),
    Input('dropdown2','value')
   ]
    
)

def average_SalePrice(OverallQual='All',GarageCars='All', CentralAir = 'all'):   
    if OverallQual!= 'All':
        temp_df= df[(df['OverallQual']== OverallQual)] #& (df['SalePrice']< 800000)
        if GarageCars != 'All':
            new_data = temp_df[temp_df['GarageCars']==GarageCars]
           
            fig=sale_price_meter(new_data) 
   
        else:
           fig = sale_price_meter(temp_df)
        return fig
    else:
        temp_df= df[(df['GarageCars']== GarageCars)]
        if GarageCars != 'All':           
            new_data = temp_df[temp_df['OverallQual']==OverallQual]
        else:            
            fig=sale_price_meter(df)
        return fig
       
       
       
       


# In[139]:


@app.callback(
    Output('rating', 'figure'),
    [Input('dropdown1', 'value'),
    Input('dropdown2','value'),
    Input('dropdown3','value'),
    Input('dropdown4','value')
    ]
     )
def rating(OverallQual='Excellent',GarageCars=2,TotalBsmtSF_m2 = 0, GrLivArea_m2= 334):
    data = dict(GarageCars=0.693147, TotalBsmtSF_m2=6.783325, GrLivArea_m2=6.799056,OverallQual =1.791759)
    df1 = pd.DataFrame(data, index=[0])
    #df1['OverallQual'] = df1['OverallQual'].map({"Very Poor": 1, "Poor": 2, "Fair": 3,"Below Average":4, "Average":5,
                                            #"Above Average":6, "Good":7, "Very Good":8, "Excellent":9, "Very Excellent":10})
    
    new_val = make_prediction.predictionD(df1)
    new_val = np.round(float(new_val),2)

    #df1
    #OverallQual = OverallQual
    #GarageCars = GarageCars
    #TotalBsmtSF_m2 = TotalBsmtSF_m2
    #GrLivArea_m2 = GrLivArea_m2
    fig=meter(new_val)
    
    return fig


# In[140]:


#plot_histogram('1Story', 2.0)


# In[141]:


#pie("2Story", 3.0)


# In[142]:


#average_SalePrice('2Story', 4)


# In[143]:


#Company_table("2Story", 3.0)


# In[144]:


################################################### salary histogram  component
@app.callback(
    Output('hist', 'figure'),
    [Input('dropdown1', 'value'),
    Input('dropdown2','value')]
     )

def plot_histogram(OverallQual=dropdown1,GarageCars=dropdown2):
    
    if OverallQual!= 'All':
        temp_df= df[(df['OverallQual']== OverallQual)]
        if GarageCars != 'All':
        
            fig = px.histogram(temp_df[temp_df['GarageCars']==GarageCars],x= 'SalePrice',template='simple_white',
                              color_discrete_sequence=['#1C4E80'],
                               nbins = 20,barmode='group')
            fig.update_layout(title ={'text': "<b style:'color:blue;'>{0}  SalePrice at  {1}</b>".format(OverallQual,GarageCars), 'font': {'size': 16}},title_x=0.5,
                        #title_font_family="Times New Roman",
                               title_font_color="slategray",margin=dict(b=10,r=40), yaxis_title={'text': "Number of houses"}, font_color='slategray')
                       #,

                   # xaxis_title={'text': "<b style:'color:blue;'>Likes</b>", 'font': {'size':15}},
                  #  yaxis_title={'text': "<b style:'color:blue';>Number of Employees </b>"}
                           
                            # plot_bgcolor='rgba(0,0,0,0)'
                              

        elif GarageCars == "All":
            fig = px.histogram(temp_df,x= 'SalePrice',template='simple_white',nbins = 20,barmode='group', color_discrete_sequence=['#1C4E80'])
            fig.update_layout(title ={'text': "<b style:'color:blue;'>{0} SalePrice </b>".format(OverallQual)
                                , 'font': {'size': 16}},title_x=0.5,
                        #title_font_family="Times New Roman",
                                 title_font_color="slategray",margin=dict(b=10,r=40), yaxis_title={'text': "Number of houses"}, font_color='slategray')
                       # font_color='blue',

                   # xaxis_title={'text': "<b style:'color:blue;'>Likes</b>", 'font': {'size':15}},
                   # yaxis_title={'text': "<b style:'color:blue';></b>", 'font': {'size': 15}},
               # paper_bgcolor='white',
               #         plot_bgcolor='rgba(0,0,0,0)'
         
        else:
            
            fig = px.histogram( temp_df,x= 'SalePrice',template='simple_white',nbins = 20,barmode='group',color=GarageCars, color_discrete_sequence=['#1C4E80'])

            
        return fig

    else:

            
        
        if GarageCars!= 'All':
            temp_df= df[(df['GarageCars']== GarageCars)]
            fig = px.histogram(temp_df,x='SalePrice',template='simple_white',nbins = 20,barmode='group', color_discrete_sequence=['#1C4E80'])
            fig.update_layout(autosize=True,title ={'text': "<b style:'color:blue;'>SalePrice at {0}</b>".format(GarageCars), 'font': {'size': 16}},
                              title_x=0.5,title_font_color="slategray",margin=dict(b=10,r=40), yaxis_title={'text': "Number of houses"}, font_color='slategray')
                        #title_font_family="Times New Roman",
                              
                       # font_color='blue',

                   # xaxis_title={'text': "<b style:'color:blue;'>Likes</b>", 'font': {'size':15}},
                   # yaxis_title={'text': "<b style:'color:blue';></b>", 'font': {'size': 15}},
                    #    paper_bgcolor='lightblue',
                      #  plot_bgcolor='rgba(0,0,0,0)')
         
 

        else :
            fig = px.histogram(df[df['SalePrice']< 80000000], x='SalePrice',template='simple_white',nbins = 20, color_discrete_sequence=['#1C4E80'])
                
            fig.update_layout(title ={'text': "<b style:'color:blue;'>Overall price at All GarageCars</b>"
                            , 'font': {'size': 16}},title_x=0.5,title_font_color="slategray",margin=dict(b=10,r=40),yaxis_title={'text': "house price"}, font_color='slategray')
                              
                        #title_font_family="Times New Roman",
                        
                       # font_color='blue',

                   # xaxis_title={'text': "<b style:'color:blue;'>Likes</b>", 'font': {'size':15}},
                   # yaxis_title={'text': "<b style:'color:blue';></b>", 'font': {'size': 15}},
                        #   paper_bgcolor='white',F
                        #   plot_bgcolor='rgba(0,0,0,0)'
    
        return fig


# In[145]:


#pt.head()


# In[146]:


#bar('2Story')


# In[147]:


@app.callback(
    Output('bar', 'figure'),
    Input('dropdown1', 'value'),

     )
def bar(HouseStyle:str):
    if HouseStyle=='All':
        temp_df= location_sorting(train_df)
        fig= px.bar(temp_df,y= 'MSZoning',x='Count',color='MSZoning',template='simple_white',color_discrete_sequence=px.colors.sequential.Blues_r,
                orientation='h'
             )
        fig.update_traces(showlegend=False)
        fig.update_layout(margin=dict(b=10,r=10,l=0),
                        title ={'text': "<b style:'color:blue;'> Top Locations</b>", 'font': {'size': 16}},title_x=0.5,
                        #title_font_family="Times New Roman",
                        title_font_color="slategray",
                        font_color='slategray',

                   # xaxis_title={'text': "<b style:'color:blue;'>Likes</b>", 'font': {'size':15}},
                    yaxis_title={'text': "<b style:'color:blue';></b>", 'font': {'size': 15}},
                    paper_bgcolor='white',
                    plot_bgcolor='rgba(0,0,0,0)')
        return fig
        
    else:
        temp_df = df[df['HouseStyle']== HouseStyle]
    
        temp_df=location_sorting(temp_df)
        fig= px.bar(temp_df,y= 'MSZoning',x='Count',color='MSZoning',template='simple_white',color_discrete_sequence=px.colors.sequential.Blues_r,
                    orientation='h' )
        fig.update_traces(showlegend=False)
        fig.update_layout(
                        title ={'text': "<b style:'color:blue;'>Top Locations for {0}</b>".format(HouseStyle), 'font': {'size': 15}},title_x=0.5,
                        #title_font_family="Times New Roman",
                        title_font_color="slategray",margin=dict(b=10,r=10,l=0),
                        font_color='slategray',

                   # xaxis_title={'text': "<b style:'color:blue;'>Likes</b>", 'font': {'size':15}},
                    yaxis_title={'text': "<b style:'color:blue';></b>", 'font': {'size': 15}},
                    paper_bgcolor='white',
                    plot_bgcolor='rgba(0,0,0,0)')
        return fig


# In[148]:


def data_feat(SalePrice):
    data = {'features': ['OverallQual', 'GrLivArea_m2', 'TotalBsmtSF_m2', 'GarageCars','others'],#, 'BsmtFinSF1_m2'],#,'LotArea_m2',
                        #'YearRemodAdd','OverallCond','CentralAir','HouseStyle','others'],
            'amount': [SalePrice*65.5/100, SalePrice*12.5/100, SalePrice*6.5/100, SalePrice*5.5/100, SalePrice*10/100],
                    #SalePrice*2/100,SalePrice*1/100,SalePrice*0.5/100,SalePrice*0.4/100,SalePrice*0.45/100,SalePrice*10/100],
            'percentage': [65.5, 12.5, 6.5, 5.5, 10]#,2,1,0.5,0.4,0.45,10]
            }
    return(pd.DataFrame(data))

#df = pd.DataFrame(data)


# In[149]:


d1 =data_feat(100000)
d2 =data_feat(150000)
d3 =data_feat(170000)
d4 =data_feat(109000)
d5 =data_feat(320000)


# In[150]:



########################################################  Pie chart component 
@app.callback(
    Output('pie', 'figure'),
    [Input('dropdown1', 'value'),
    Input('dropdown2','value')]
     )
def pie(OverallQual=dropdown1,GarageCars=dropdown2):

    if OverallQual!= 'All':
        temp_df= df[(df['OverallQual']== OverallQual) & (df['SalePrice']< 4000000)]
        if OverallQual!= 'All':
            #new_data = temp_df[temp_df['GarageCars']==GarageCars]
            data= d1#new_data['Employment Status'].value_counts()
            fig=donut(data)
   
        elif GarageCars == "All":
            data = d2#temp_df['Employment Status'].value_counts()
            fig=donut(data)
        else:
            data = d3#temp_df['Employment Status'].value_counts()
            fig=donut(data)
            

            
        return fig

    else:

            
        
        if GarageCars!= 'All':
            #temp_df= df[(df['Location']== location) & (df['Salary']< 4000000)]
            
            data = d4#temp_df['Employment Status'].value_counts()
            fig=donut(data)


        else :
           
            data = d5#df['Employment Status'].value_counts()
            fig=donut(data)
        return fig


# In[151]:


############################################################################# table component 
@app.callback(
    Output('Company_table', 'figure'),
    [Input('dropdown1', 'value'),
    Input('dropdown2','value')]
     )

def Company_table(OverallQual=dropdown1,GarageCars=dropdown2):
    if OverallQual!= 'All':
        temp_df = df[(df['OverallQual']== OverallQual)]#.reset_index()# & (df['SalePrice']< 4000000)]
        #temp_df = del temp_df['index']
        #temp_df=temp_df.reset_index(drop=True)#, inplace=True)
       
        #temp_df= temp_df.head(10)
        fig=Table(temp_df)
    else:
        #temp_df = df.head(10)
        #temp_df = del temp_df['index']
        #temp_df=temp_df.reset_index(drop=True, inplace=True)
        #temp_df = temp_df.reset_index(drop=True)
        #temp_df= temp_df.head(10)
        fig=Table(df)
        #fig=Table(df.head(10))
    return fig   


# In[152]:


if __name__ == '__main__':
    app.run_server(mode='inline',port=8050)


# 
