#!/usr/bin/env python
# coding: utf-8

# ### Loss Analysis Project

# In[36]:


from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from pandas.tseries.offsets import DateOffset
import textwrap
import pyodbc
import plotly
from datetime import datetime
#-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                               # Importing for panel data analysis 
pd.set_option('display.max_columns', None)                                        # Unfolding hidden features if the cardinality is high      
pd.set_option('display.max_colwidth', None)                                       # Unfolding the max feature width for better clearity      
pd.set_option('display.max_rows', None)                                           # Unfolding hidden data points if the cardinality is high
pd.set_option('display.float_format', lambda x: '%.2f' % x)                       # To suppress scientific notation over exponential values
#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                                # Importing package numpys (For Numerical Python)
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
import warnings                                                                   # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")     


# In[2]:


pyodbc.drivers()


# In[3]:


connection_string = "DRIVER={SQL Server Native Client 11.0};SERVER=gsc-scpat-sql-001-d.database.windows.net;DATABASE=SC-PAT-DB;UID=SCPAT;PWD=Ecolab@1234"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

conn = create_engine(connection_url)


# In[27]:


conn


# In[28]:


data_query = textwrap.dedent( """ SELECT [MD_Error]
,[Date]
,SUM(cast([ABS_STAT] as float)) AS [ABS_STAT],SUM(cast([ABS_CONS] as float)) AS [ABS_CONS]
,[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc]
,sum(cast([Consensus] as float)) as [Consensus],sum(cast([Stat]as float)) as [Stat],sum(cast([Actual]as float)) as [Actual]
,[Origin]
FROM [FABIAS_Dashboard].[FALossBaseCal]
GROUP BY
[MD_Error],[Date],[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc],[Origin]
  """)


# In[29]:


df = pd.read_sql(data_query,con=conn)


# In[32]:


data = df


# In[80]:


data.shape


# In[81]:


df.shape


# In[77]:


df.shape


# In[70]:


df['Date'].value_counts().sort_index()


# In[71]:


df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')


# In[73]:


df['Key'] = df['Item'] +","+ df['Material Type'] +","+ df['Div'] +","+ df['Cust Facing Loc'] +","+ df['Sales Org']+","+df['Lag']


# In[76]:


df.drop_duplicates(inplace=True)


# In[74]:


df.head()


# In[ ]:





# ### Creating L12M :

# In[83]:


last_date = df['Date'].max()

# In[84]:


last_12M_date = last_date - pd.DateOffset(months = 12)
last_12M_date


# In[85]:


L12M = df[df['Date'].between(last_12M_date,last_date,inclusive='both')]


# In[90]:


L12M=L12M.groupby(by=['Key']).sum()[['Actual','Stat','Consensus','ABS_STAT','ABS_CONS']].reset_index()


# In[91]:


L12M.rename(columns={'Actual':'L12M_Actual_Sum','Stat':'L12M_Stat_Sum','Consensus':'L12M_Consensus_Sum',
                     'ABS_STAT':'L12M_ABSStat_Sum','ABS_CONS':'L12M_ABSCons_Sum'},inplace=True)


# In[92]:


L12M=L12M.astype({'Key':'string','L12M_Actual_Sum':int,'L12M_Stat_Sum':int,'L12M_Consensus_Sum':int,'L12M_ABSStat_Sum':int,
             'L12M_ABSCons_Sum':int})


# In[94]:


L12M['R6M/R12M'] = 'R12M'


# In[95]:


L12M[L12M['Key'] == '008-1653119,ZKIT,Light,1911,4073,Lag 1']


# ### Creating L6M Table

# In[501]:


""" 
L6M_query = textwrap.dedent( SELECT [MD_Error]
,[Date]
,SUM(cast([ABS_STAT] as float)) AS [ABS_STAT],SUM(cast([ABS_CONS] as float)) AS [ABS_CONS]
,[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc]
,sum(cast([Consensus] as float)) as [Consensus]
,sum(cast([Stat]as float)) as [Stat]
,sum(cast([Actual]as float)) as [Actual]
,[Origin]
FROM [FABIAS_Dashboard].[FALossBaseCal]
where [Date] > Dateadd(Month, -7, getdate()) and [Date] < Dateadd(Month, -1,getdate())
-- WHERE [Item] = '74700PLUS.61R' and [Lag] = 'Lag 2' and [Period] = '202101'
GROUP BY
[MD_Error],[Date],[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc],[Origin] )

"""


# In[96]:


last_date = df['Date'].max()


# In[97]:


last_date


# In[98]:


last_6M_date = last_date - pd.DateOffset(months = 6)
last_6M_date


# In[101]:


L6M = df[df['Date'].between(last_6M_date,last_date,inclusive='both')]


# In[102]:


L6M['Date'].describe()


# In[103]:


L6M=L6M.groupby(by=['Key']).sum()[['Actual','Stat','Consensus','ABS_STAT','ABS_CONS']].reset_index()


# In[104]:


L6M.rename(columns={'Actual':'L6M_Actual_Sum','Stat':'L6M_Stat_Sum','Consensus':'L6M_Consensus_Sum',
                     'ABS_STAT':'L6M_ABSStat_Sum','ABS_CONS':'L6M_ABSCons_Sum'},inplace=True)


# In[105]:


L6M['R6M/R12M'] = 'R6M'


# In[106]:


L6M[L6M['Key'] == '008-1652091,ZKIT,Light,1911,4073,Lag 1']


# ### Creating F12M Table

# In[107]:


F12M_query = textwrap.dedent("""  SELECT [MD_Error]
,[Date]
,SUM(cast([ABS_STAT] as float)) AS [ABS_STAT],SUM(cast([ABS_CONS] as float)) AS [ABS_CONS]
,[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc]
,sum(cast([Consensus] as float)) as [Consensus]
,sum(cast([Stat]as float)) as [Stat]
,sum(cast([Actual]as float)) as [Actual]
,[Origin]
FROM [FALossBaseCal]
WHERE [Date] >= Dateadd(Month, -1,getdate())
AND [Date] <= DATEADD(MONTH, 11, CAST(GETDATE() AS DATE))
GROUP BY
[MD_Error],[Date],[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc],[Origin]  """)


# In[108]:


conn


# In[109]:


F12M_source = pd.read_sql(F12M_query,con=conn)


# In[110]:


F12M_source = F12M_source.convert_dtypes()


# In[111]:


F12M_source['Material Type'] = F12M_source['Material Type'].astype('string')


# In[112]:


#F12M_source['Key'] = F12M_source[['Item','Material Type','Div','Cust Facing Loc','Sales Org','Lag']].agg(','.join, axis=1) 


# In[118]:


F12M_source['Key'] = F12M_source['Item'] +","+ F12M_source['Material Type'] +","+ F12M_source['Div'] +","+ F12M_source['Cust Facing Loc'] +","+ F12M_source['Sales Org']+","+F12M_source['Lag']


# In[119]:


F12M_source['Date'] = pd.to_datetime(F12M_source['Date'])


# In[120]:


F12M = pd.DataFrame(F12M_source.groupby(['Key'])['Actual','Stat','Consensus','ABS_STAT','ABS_CONS'].sum()).reset_index()


# In[121]:


F12M.rename(columns={'Actual':'F12M_Actual_Sum','Stat':'F12M_Stat_Sum','Consensus':'F12M_Consensus_Sum',
                     'ABS_STAT':'F12M_ABSStat_Sum','ABS_CONS':'F12M_ABSCons_Sum'},inplace=True)


# In[122]:


F12M[F12M['Key']=='008-1653403,ZKIT,Light,0150356791,4073,Lag 2']


# ### Creating F6M Table

# In[113]:


F6M_query = textwrap.dedent(""" SELECT [MD_Error]
,[Date]
,SUM(cast([ABS_STAT] as float)) AS [ABS_STAT],SUM(cast([ABS_CONS] as float)) AS [ABS_CONS]
,[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc]
,sum(cast([Consensus] as float)) as [Consensus]
,sum(cast([Stat]as float)) as [Stat]
,sum(cast([Actual]as float)) as [Actual]
,[Origin]
FROM [FALossBaseCal]
WHERE [Date] >= Dateadd(Month, -1,getdate())
AND [Date] <= DATEADD(MONTH, 5, CAST(GETDATE() AS DATE))
GROUP BY
[MD_Error],[Date],[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc],[Origin] """)


# In[114]:


conn


# In[115]:


F6M_source = pd.read_sql(F6M_query,con=conn)


# In[116]:


F6M_source['Key'] = F6M_source['Item'] +","+ F6M_source['Material Type'] +","+ F6M_source['Div'] +","+ F6M_source['Cust Facing Loc'] +","+ F6M_source['Sales Org']+","+F6M_source['Lag']

F6M_source['Date'] = pd.to_datetime(F6M_source['Date'])


# In[123]:


F6M = pd.DataFrame(F6M_source.groupby(['Key'])['Actual','Stat','Consensus','ABS_STAT','ABS_CONS'].sum()).reset_index()


# In[124]:


F6M.rename(columns={'Actual':'F6M_Actual_Sum','Stat':'F6M_Stat_Sum','Consensus':'F6M_Consensus_Sum',
                     'ABS_STAT':'F6M_ABSStat_Sum','ABS_CONS':'F6M_ABSCons_Sum'},inplace=True)


# In[126]:


F6M[F6M['Key']=='00000001541,ZHLB,QSR_DIV,S4952M000000599402,4952,Lag 2']


# ### Creating L12M_Act_count Table

# In[91]:


L12M_act_count_query = textwrap.dedent( """ SELECT [MD_Error],[Date]
,SUM(cast([ABS_STAT] as float)) AS [ABS_STAT],SUM(cast([ABS_CONS] as float)) AS [ABS_CONS]
,[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc]
,sum(cast([Consensus] as float)) as [Consensus]
,sum(cast([Stat]as float)) as [Stat]
,sum(cast([Actual]as float)) as [Actual]
,[Origin]
FROM [FABIAS_Dashboard].[FALossBaseCal]
where [Date] > Dateadd(Month, -13, getdate()) and [Date] < Dateadd(Month, -1,getdate())
-- WHERE [Item] = '74700PLUS.61R' and [Lag] = 'Lag 2' and [Period] = '202101'
GROUP BY
[MD_Error],[Date],[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc],[Origin]   """)


# In[93]:


#L12M_Act_Count_source = pd.read_sql(L12M_act_count_query,con=conn)


# In[131]:


L12M_Act_count = df[df['Date'].between(last_12M_date,last_date,inclusive='both')]


# In[133]:


L12M_Act_count['Key'] = L12M_Act_count['Item'] +","+ L12M_Act_count['Material Type'] +","+ L12M_Act_count['Div'] +","+ L12M_Act_count['Cust Facing Loc'] +","+ L12M_Act_count['Sales Org']+","+L12M_Act_count['Lag']


# In[135]:


L12M_Act_count = L12M_Act_count.convert_dtypes()


# In[136]:


L12_Act_count = L12M_Act_count['Key'].value_counts().reset_index()


# In[137]:


L12_Act_count.rename(columns={'index':'Key','Key':'L12M_Act_Count'},inplace=True)


# In[138]:


L12_Act_count.head()


# ### Creating L3M Table

# In[128]:


last_date = df['Date'].max()
last_date


# In[129]:


last_3M_date = last_date - pd.DateOffset(months=3)
last_3M_date


# In[146]:


L3M = df[df['Date'].between(last_3M_date,last_date,inclusive='both')]


# In[148]:


L3M = pd.DataFrame(L3M['Key'])


# In[ ]:


L3M.drop_duplicates(inplace=True)


# In[150]:


L3M.reset_index(inplace=True)


# In[152]:


L3M['L3M'] = 'Yes'


# In[154]:


L3M = L3M[['Key','L3M']]


# In[169]:


L3M.head()


# In[170]:


L3M.shape


# ### Creating L9M

# In[157]:


last_9M_date = last_date - pd.DateOffset(months=9)


# In[158]:


L9M = df[df['Date'].between(last_9M_date,last_date,inclusive='both')]


# In[166]:


L9M.shape


# In[160]:


L9M['Date'].describe()


# In[161]:


L9M.reset_index(inplace=True)


# In[162]:


L9M['L9M'] = 'Yes'


# In[163]:


L9M.drop(columns=['index', 'MD_Error', 'Date', 'ABS_STAT', 'ABS_CONS', 'Period', 'Lag','Item', 'Material Type', 'Div', 'Sales Org', 'Cust Facing Loc','Consensus', 'Stat', 'Actual', 'Origin'],inplace=True)


# In[165]:


L9M.head()


# In[171]:


L3M_L9M = pd.merge(L3M,L9M,on='Key',how='right')


# In[172]:


L3M_L9M.shape


# ### Creating Dashboard Final Data

# In[209]:


"""
dashboard_query = textwrap.dedent( SELECT [MD_Error]
,[Date]
,SUM(cast([ABS_STAT] as float)) AS [ABS_STAT],SUM(cast([ABS_CONS] as float)) AS [ABS_CONS]
,[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc]
,sum(cast([Consensus] as float)) as [Consensus]
,sum(cast([Stat]as float)) as [Stat]
,sum(cast([Actual]as float)) as [Actual]
,[Origin]
FROM [dbo].[FABIAS_Dashboard.FALossBaseCal]
--WHERE [Date] >= Dateadd(Month, -1,getdate())
--AND [Date] <= DATEADD(MONTH, 11, CAST(GETDATE() AS DATE))
GROUP BY
[MD_Error],[Date],[Period],[Lag],[Item],[Material Type],[Div],[Sales Org],[Cust Facing Loc],[Origin])
"""


# In[210]:


#dashboard_df = pd.read_sql(dashboard_query,con=conn)


# In[201]:


dashboard_df.shape


# In[179]:


dashboard_df = df[['MD_Error','Key']]


# In[177]:


#dashboard_df.drop_duplicates(inplace=True)


# In[181]:


dashboard_df = pd.merge(dashboard_df,F12M,on='Key')


# In[182]:


dashboard_df = pd.merge(dashboard_df,F6M,on='Key')


# In[183]:


dashboard_df = pd.merge(dashboard_df,L12M,on='Key')


# In[184]:


dashboard_df = pd.merge(dashboard_df,L6M,on='Key')


# In[185]:


dashboard_df = pd.merge(dashboard_df,L12_Act_count,on='Key')


# In[186]:


dashboard_df = pd.merge(dashboard_df,L3M_L9M,on='Key')


# In[187]:


dashboard_df['L6M_Actual_Sum'].replace(0,np.NaN,inplace=True)


# In[188]:


dashboard_df['L6M_FA'] = dashboard_df['L6M_ABSStat_Sum']/dashboard_df['L6M_Actual_Sum']


# In[189]:


def con(row):
    if row['L6M_FA'] >= 1:
        val = 0
    elif row['L6M_FA'] < 1:
        val = (1-(row['L6M_FA']))
    else :
        val = np.NaN
    return val


# In[190]:


dashboard_df['L6M_FA_STAT'] = dashboard_df.apply(con,axis=1)


# In[ ]:





# In[193]:


dashboard_df['L6M_FA_CON'] = dashboard_df['L6M_ABSCons_Sum']/dashboard_df['L6M_Actual_Sum']


# In[ ]:





# In[194]:


def cond2(row):
    if row['L6M_FA_CON'] >= 1:
        val = 0
    elif row['L6M_FA_CON'] < 1:
        val = (1- row['L6M_FA_CON'])
    else:
        val = np.NaN
    return val


# In[195]:


dashboard_df['L6M_FA_CONSEN'] = dashboard_df.apply(cond2,axis=1)


# In[196]:


dashboard_df['L6M_FA_CONSEN'].unique()


# In[197]:


def level3(row):
    if row['MD_Error'] == 'Yes':
        val = 'MASTER DATA ERROR'
    elif row['L9M'] == 'NO':
        val = 'TRAIL/NPI'
    elif row['L6M_Actual_Sum'] > row['L6M_Consensus_Sum']:
        val = 'UNDER FORECASTING'
    elif row['L6M_Actual_Sum'] < row['L6M_Consensus_Sum']:
        val = "OVER FORECASTING"
    elif row['L6M_Stat_Sum'] > row['L6M_Consensus_Sum']:
        val = "OUTLIER CORRECTION"
    elif row['L12M_Act_Count'] == 0 and row['F6M_Consensus_Sum'] > 0:
        val = "FORECAST NOT ZEROED OUT"
    elif row['L12M_Act_Count'] > 0 and row['L12M_Act_Count'] < 4 :
        val = "LUMPY DEMAND"
    elif row['L12M_Actual_Sum'] <= 0 and row['F6M_Consensus_Sum'] <= 0:
        val = "DISCONTINUE"
    elif row['L6M_FA_STAT'] > row['L6M_FA_CONSEN']:
        val = "INCORRECT CHOICE OF MODEL"
    else:
        val =  "OK/MANUAL INPUT"
    return val


# In[198]:


dashboard_df['LEVEL03'] = dashboard_df.apply(level3,axis=1)


# In[199]:


def level2_condition(row):
    if row['LEVEL03'] == "MASTER DATA ERROR":
        val = "MASTER DATA"
    elif row['LEVEL03'] == "TRIAL/NPI":
        val = "TRIAL/NPI" 
    elif row['LEVEL03'] == "OVER FORECASTING":
        val = "FORECASTING ADJUSTMENTS" 
    elif row['LEVEL03'] == "UNDER FORECASTING":
        val = "FORECASTING ADJUSTMENTS"
    elif row['LEVEL03'] == "OUTLIER CORRECTION":
        val = "HISTORY ADJUSTMENTS"
    elif row['LEVEL03'] == "FORECAST NOT ZEROED OUT":
        val = "FORECAST MAPPING"
    elif row['LEVEL03'] == "LUMPY DEMAND":
        val = "FORECASTING ADJUSTMENTS"
    elif row['LEVEL03'] == "DISCONTINUE":
        val ="FORECASTING ADJUSTMENTS"
    elif row['LEVEL03'] == "INCORRECT CHOICE OF MODEL":
        val = "FORECASTING MODEL"
    elif row['LEVEL03'] == "OK/MANUAL INPUT":
        val = "OK/MANUAL INPUT"
    elif row['L6M_Stat_Sum']==row['L6M_Consensus_Sum'] and row['L6M_FA_CONSEN'] < 0.75 :
        val = "FORECASTING MODEL"
    elif row['L6M_ABSStat_Sum'] != row['L6M_ABSCons_Sum']:
        val = 'FORECASTING ADJUSTMENTS'
    else:
        val = 'NA'
    return val


# In[200]:


dashboard_df['LEVEL02'] = dashboard_df.apply(level2_condition,axis=1)


# ### Level-02 query condition:
# """
# Added Conditional Column", "LEVEL2", 
# each if [LEVEL3] = "MASTER DATA ERROR" then "MASTER DATA" 
# else if [LEVEL3] = "TRIAL/NPI" then "TRIAL/NPI" 
# else if [LEVEL3] = "OVER FORECASTING" then "FORECASTING ADJUSTMENTS" 
# else if [LEVEL3] = "UNDER FORECASTING" then "FORECASTING ADJUSTMENTS" 
# else if [LEVEL3] = "OUTLIER CORRECTION" then "HISTORY ADJUSTMENTS" 
# else if [LEVEL3] = "FORECAST NOT ZEROED OUT" then "FORECAST MAPPING" 
# else if [LEVEL3] = "LUMPY DEMAND" then "FORECASTING ADJUSTMENTS" 
# else if [LEVEL3] = "DISCONTINUE" then "FORECASTING ADJUSTMENTS" 
# else if [LEVEL3] = "INCORRECT CHOICE OF MODEL" then "FORECASTING MODEL" 
# else if [LEVEL3] = "OK/MANUAL INPUT" then "OK/MANUAL INPUT" 
# else if [L6M_Stat_Sum]=[L6M_Consensus_Sum] and [L6M_FA_CONSEN] <0.75 then "FORECASTING MODEL" 
# else if [L6M_ABSStat_Sum]<>[L6M_ABSCons_Sum] then "FORECASTING ADJUSTMENTS" else ""
# """

# ### Level-01 

# ### Level1 Condition :
# """
# Added Custom2", "LEVEL1", each if [LEVEL2] = "MASTER DATA" then "DATA" 
# else if [LEVEL2]="TRIAL/NPI" then "SALES MARKETING INPUT" 
# else if [LEVEL2] = "FORECASTING ADJUSTMENTS" then "FORECAST PROCESS" 
# else if [LEVEL2] = "HISTORY ADJUSTMENTS" then "FORECAST PROCESS" 
# else if [LEVEL2] = "FORECAST MAPPING" then "FORECAST PROCESS" 
# else if [LEVEL2] = "FORECASTING MODEL" then "FORECAST PROCESS" 
# else if [LEVEL2] = "OK/MANUAL INPUT" then "OK/MANUAL INPUT" else "NA"
# """

# In[202]:


def level01_condition(row):
    if row['LEVEL02'] == "MASTER DATA":
        val = "DATA"
    elif row['LEVEL02'] == "TRIAL/NPI":
        val = "SALES MARKETING INPUT"
    elif row['LEVEL02'] == "FORECASTING ADJUSTMENTS":
        val = "FORECAST PROCESS"
    elif row['LEVEL02'] == "HISTORY ADJUSTMENTS":
        val = "FORECAST PROCESS"
    elif row['LEVEL02'] == "FORECAST MAPPING":
        val = "FORECAST PROCESS"
    elif row['LEVEL02'] == "FORECASTING MODEL":
        val = "FORECAST PROCESS"
    elif row['LEVEL02'] == "OK/MANUAL INPUT":
        val = "OK/MANUAL INPUT"
    else:
        val = 'NA'
    return val


# In[203]:


dashboard_df['LEVEL01'] = dashboard_df.apply(level01_condition,axis=1)


# In[204]:


dashboard_df[['Item','Material_Type','Division','Cust_Facing_Loc','Sales_org','Lag']] = dashboard_df['Key'].str.split(',',5,expand=True)


# In[242]:


dashboard_df['Abs_Error_Cons'] = dashboard_df['L12M_Actual_Sum'] - dashboard_df['L12M_Consensus_Sum']


# In[243]:


dashboard_df['Abs_Error_Cons'] = dashboard_df['Abs_Error_Cons'].apply(lambda x:abs(x))


# ### FA Calculation :
# 
# * Formula : R12M_FA_CONS = IFERROR(IF((SUM(DASHBOARD[L12M_ABSERROR_CONS])/SUM(L12M[L12M_Actual_Sum]))>1,
#                           0,1-(SUM(DASHBOARD[L12M_ABSERROR_CONS])/SUM(DASHBOARD[L12M_Actual_Sum]))),BLANK())
# 

# In[244]:


def FA(row):
    if (row['Abs_Error_Cons']/row['L12M_Actual_Sum']) >1:
        val = 0
    elif (row['Abs_Error_Cons']/row['L12M_Actual_Sum']) <=1:
        val = (1-(row['Abs_Error_Cons']/row['L12M_Actual_Sum']))
    else :
        val = np.NaN
    return val


# In[250]:


dashboard_df['L12M_Actual_Sum'].replace(0,np.NaN,inplace=True)


# In[251]:


dashboard_df['FA'] = dashboard_df.apply(FA,axis=1)


# ### BAIS Calculalation :

# #### Formula :
# 
# * L12M_BIAS_CONSEN = IFERROR(((SUM(DASHBOARD[L12M_Actual_Sum])-   SUM(DASHBOARD[L12M_Consensus_Sum]))/SUM(DASHBOARD[L12M_Consensus_Sum]))*-1,0) 

# In[253]:


dashboard_df['Bais'] = (dashboard_df['L12M_Consensus_Sum']-dashboard_df['L12M_Actual_Sum'])/dashboard_df['L12M_Consensus_Sum']


# In[ ]:





# In[984]:

"""
#dashboard_df = dashboard_df[['MD_Error', 'Key', 'Item', 'Material_Type',
       'Division', 'Cust_Facing_Loc', 'Sales_org', 'Lag','F12M_Actual_Sum', 'F12M_Stat_Sum',
       'F12M_Consensus_Sum', 'F12M_ABSStat_Sum', 'F12M_ABSCons_Sum',
       'F6M_Actual_Sum', 'F6M_Stat_Sum', 'F6M_Consensus_Sum',
       'F6M_ABSStat_Sum', 'F6M_ABSCons_Sum', 'L12M_Actual_Sum',
       'L12M_Stat_Sum', 'L12M_Consensus_Sum', 'L12M_ABSStat_Sum',
       'L12M_ABSCons_Sum', 'L6M_Actual_Sum', 'L6M_Stat_Sum',
       'L6M_Consensus_Sum', 'L6M_ABSStat_Sum', 'L6M_ABSCons_Sum',
       'L12M_Act_Count', 'L3M', 'L9M', 'L6M_FA', 'L6M_FA_STAT', 'L6M_FA_CON',
       'L6M_FA_CONSEN', 'LEVEL03', 'LEVEL02', 'LEVEL01','R6M/R12M_x','R6M/R12M_y']]

"""
# In[943]:


dashboard_df['Material_Type'].unique()


# In[945]:


dashboard_df['Division'].unique()


# In[946]:


dashboard_df['Sales_org'].unique()


# In[947]:


dashboard_df.shape


# In[259]:


dashboard_df.head()


# In[219]:


loss_df = dashboard_df[['Item','LEVEL01','LEVEL02','LEVEL03','Division','Material_Type','Lag','Sales_org','Cust_Facing_Loc']]


# In[230]:


loss_df['Forecast_Consensus'] = dashboard_df['L12M_Consensus_Sum']


# In[233]:


loss_df['Actual'] = dashboard_df['L12M_Actual_Sum']


# In[236]:


loss_df['Abs_Error_Cons'] = dashboard_df['L12M_Actual_Sum'] - dashboard_df['L12M_Consensus_Sum']


# In[240]:


loss_df['Abs_Error_Cons'] = loss_df['Abs_Error_Cons'].apply(lambda x:abs(x))


# In[260]:


loss_df['FA'] = dashboard_df['FA']
loss_df['BAIS'] = dashboard_df['Bais']


# In[ ]:





# In[ ]:





# In[261]:


loss_df[(loss_df['Item']=='9000.91R')&(loss_df['Sales_org']=='4172')]


# In[ ]:





# In[ ]:




