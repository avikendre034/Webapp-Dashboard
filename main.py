
#!/usr/bin/env python
# coding: utf-8

# ### Loss Analysis Project

from dataclasses import dataclass
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from pathlib import Path                                                                                                         
import numpy as np                                       
from PIL import Image
import textwrap
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from datetime import date
from pandas.tseries.offsets import DateOffset
import warnings                                                                          # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore") 

#------Connection__Creation-------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="FA Loss Analysis",page_icon=":bar_chart:",layout="wide")

connection_string = "DRIVER={SQL Server Native Client 11.0};SERVER=gsc-scpat-sql-001-d.database.windows.net;DATABASE=SC-PAT-DB;UID=SCPAT;PWD=Ecolab@1234"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
conn = create_engine(connection_url)

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
#-------Import_Main_Data------------------------------------------------------------------------------------------------------------

data_query = textwrap.dedent( """ select 
[Period],
[Lag],
[Item],
[Material Type],
[Div],
[Sales Org],
[Cust Facing Loc],
[Origin],
sum(cast([Consensus]as float)) as [Consensus],
sum(Cast([Actual] as float)) as [Actual],
sum(cast([Stat] as float)) as [Stat],
[MD_Error],
[Date],
abs(sum(cast([Consensus]as float)) - sum(Cast([Actual] as float))) as ABS_CONS,
abs(sum(cast([Stat] as float)) - sum(Cast([Actual] as float))) as ABS_STAT
from [FABIAS_Dashboard].[FALossBase_T] GROUP BY
[Period],
[Lag],
[Item],
[Material Type],
[Div],
[Sales Org],
[Cust Facing Loc],
[Origin],
[MD_Error],
[Date]
  """)

#--------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

@st.experimental_singleton
def load_data():
    data = pd.read_sql_query(data_query,conn)
    #ddf = dd.from_pandas(data,npartitions=100)
    return data

df = load_data()
df['Key'] = df['Item'] +","+ df['Material Type'] +","+ df['Div'] +","+ df['Cust Facing Loc'] +","+ df['Sales Org']+","+df['Lag']
df.drop_duplicates(subset='Key',inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

#st.write('Source data Shape is',df.shape)

#-------------Creating L12M Table ----------------------------------------------------------------------------------------------------

current_Month_StarDate = (date.today().date() - pd.offsets.MonthBegin(n=1))
last_date = df['Date'].max()
first_date = df['Date'].min()
last_12M_date = current_Month_StarDate - pd.DateOffset(months = 12)
L12M = df[df['Date'].between(last_12M_date,current_Month_StarDate,inclusive='both')]

L12M=L12M.groupby(by=['Key']).sum()[['Actual','Stat','Consensus','ABS_STAT','ABS_CONS']].reset_index()

L12M.rename(columns={'Actual':'L12M_Actual_Sum','Stat':'L12M_Stat_Sum','Consensus':'L12M_Consensus_Sum',
                     'ABS_STAT':'L12M_ABSStat_Sum','ABS_CONS':'L12M_ABSCons_Sum'},inplace=True)

L12M=L12M.astype({'Key':'string','L12M_Actual_Sum':int,'L12M_Stat_Sum':int,'L12M_Consensus_Sum':int,'L12M_ABSStat_Sum':int,
             'L12M_ABSCons_Sum':int})

L12M['R6M/R12M'] = 'R12M'

#--------Creating L6M Table-------------------------------------------------------------------------------------------------------

current_Month_StarDate = df['Date'].max()
last_6M_date = current_Month_StarDate - pd.DateOffset(months = 6)
L6M = df[df['Date'].between(last_6M_date,current_Month_StarDate,inclusive='both')]
L6M=L6M.groupby(by=['Key']).sum()[['Actual','Stat','Consensus','ABS_STAT','ABS_CONS']].reset_index()

L6M.rename(columns={'Actual':'L6M_Actual_Sum','Stat':'L6M_Stat_Sum','Consensus':'L6M_Consensus_Sum',
                     'ABS_STAT':'L6M_ABSStat_Sum','ABS_CONS':'L6M_ABSCons_Sum'},inplace=True)

L6M['R6M/R12M'] = 'R6M'
#L6M.shape
####-------Creating F12M Table---------------------------------------------------------------------------------------------------------
F12M = df[df['Date'].between(last_12M_date,current_Month_StarDate,inclusive='both')]
#F12M = pd.DataFrame(F12M.groupby(['Key'])['Actual','Stat','Consensus','ABS_STAT','ABS_CONS'].sum()).reset_index()

F12M.rename(columns={'Actual':'F12M_Actual_Sum','Stat':'F12M_Stat_Sum','Consensus':'F12M_Consensus_Sum',
                     'ABS_STAT':'F12M_ABSStat_Sum','ABS_CONS':'F12M_ABSCons_Sum'},inplace=True)

#F12M.shape
#-------------------------------------------------------------------------------------------------------------------------------
#---------------------Create F6M Table------------------------------------------------------------------------------------------

F6M = df[df['Date'].between(last_6M_date,current_Month_StarDate,inclusive='both')]
#F6M = pd.DataFrame(F6M.groupby(['Key'])['Actual','Stat','Consensus','ABS_STAT','ABS_CONS'].sum()).reset_index()

F6M.rename(columns={'Actual':'F6M_Actual_Sum','Stat':'F6M_Stat_Sum','Consensus':'F6M_Consensus_Sum',
                     'ABS_STAT':'F6M_ABSStat_Sum','ABS_CONS':'F6M_ABSCons_Sum'},inplace=True)
#F6M.shape
#--------------------------------------------------------------------------------------------------------------------------------------
#-----------------Create L12M_Act_count Table----------------------------------------------------------------------------------------

L12M_Act_count = df[df['Date'].between(last_12M_date,current_Month_StarDate,inclusive='both')]

#L12M_Act_count['Key'] = L12M_Act_count['Item'] +","+ L12M_Act_count['Material Type'] +","+ L12M_Act_count['Div'] +","+ L12M_Act_count['Cust Facing Loc'] +","+ L12M_Act_count['Sales Org']+","+L12M_Act_count['Lag']

#L12M_Act_count = L12M_Act_count.convert_dtypes()
L12_Act_count = L12M_Act_count['Key'].value_counts().reset_index()
L12_Act_count.rename(columns={'index':'Key','Key':'L12M_Act_Count'},inplace=True)
#L12_Act_count.shape
#------------------------------------------------------------------------------------------------------------------------------
#-----------------Creating L3M Table--------------------------------------------------------------------------------------------


last_3M_date = current_Month_StarDate - pd.DateOffset(months=3)
#last_3M_date
L3M = df[df['Date'].between(last_3M_date,current_Month_StarDate,inclusive='both')]
L3M = pd.DataFrame(L3M['Key'])
L3M.drop_duplicates(inplace=True)
L3M.reset_index(inplace=True)
L3M['L3M'] = 'Yes'
L3M = L3M[['Key','L3M']]
#st.write('L3M Shape is :',L3M.shape)

#-------------------------------------------------------------------------------------------------------------------------------
#----------Creating L9M & L3M_L9M Table----------------------------------------------------------------------------------------------------

last_9M_date = current_Month_StarDate - pd.DateOffset(months=9)
L9M = df[df['Date'].between(last_9M_date,current_Month_StarDate,inclusive='both')]
L9M.reset_index(inplace=True)
L9M['L9M'] = 'Yes'
L9M.drop(columns=['index', 'MD_Error', 'Date', 'ABS_STAT', 'ABS_CONS', 'Period', 'Lag','Item', 'Material Type', 'Div', 'Sales Org', 'Cust Facing Loc','Consensus', 'Stat', 'Actual', 'Origin'],inplace=True)

L3M_L9M = pd.merge(L3M,L9M,on='Key',how='right')
L3M_L9M = L3M_L9M.convert_dtypes()

#st.write('L3M_L9M Shape is :',L3M_L9M.shape)
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

#---------Creating Dashboard Table------------------------------------------------------------------------------------
#@st.experimental_singleton
#def load_data():
#    dashboard_df = df[['MD_Error','Key']]
#    return dashboard_df
#dashboard_df = load_data()

dashboard_df = pd.DataFrame(df[['Date','MD_Error','Key']])
dashboard_df = pd.merge(dashboard_df,L12M,on='Key',how='outer')
dashboard_df = pd.merge(dashboard_df,L6M,on='Key',how='outer')
dashboard_df = pd.merge(dashboard_df,F12M,on='Key',how='outer')
dashboard_df = pd.merge(dashboard_df,F6M,on='Key',how='outer')
dashboard_df = pd.merge(dashboard_df,L3M_L9M,on='Key',how='outer')
dashboard_df = pd.merge(dashboard_df,L12_Act_count,on='Key',how='outer')


def con_l6m_fa(row):
    if row['L6M_Actual_Sum'] == 0:
        val = np.NaN
    else :
        val = row['L6M_ABSStat_Sum']/row['L6M_Actual_Sum']
    return val

dashboard_df['L6M_FA'] = dashboard_df.apply(con_l6m_fa,axis=1)


#st.write('Dashboard Table Shape is :',dashboard_df.shape)
#------------Creating L6M_FA_STAT Column-------------------------------------------------------------------

def con(row):
    if row['L6M_FA'] >= 1:
        val = 0
    elif row['L6M_FA'] < 1:
        val = (1-(row['L6M_FA']))
    else :
        val = np.NaN
    return val

dashboard_df['L6M_FA_STAT'] = dashboard_df.apply(con,axis=1)

#------------------Creating L6M_FA_CON Column in Dashboard Table---------------------------------------------

def l6m_fa_cond(row):
    if row['L6M_Actual_Sum'] == 0:
        val = np.NaN
    else :
        val = row['L6M_ABSCons_Sum']/row['L6M_Actual_Sum']
    return val
dashboard_df['L6M_FA_CON'] = dashboard_df.apply(l6m_fa_cond,axis=1)

def cond2(row):
    if row['L6M_FA_CON'] >= 1:
        val = 0
    elif row['L6M_FA_CON'] < 1:
        val = (1- row['L6M_FA_CON'])
    else:
        val = np.NaN
    return val

dashboard_df['L6M_FA_CONSEN'] = dashboard_df.apply(cond2,axis=1)

#-----------------Creating Level-03------------------------------------------------------------------------------

def level3_cond(row):
    if row['MD_Error'] == 'Yes':
        val = 'MASTER DATA ERROR'
    elif row['L9M']  is np.NaN:
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

dashboard_df['LEVEL03'] = dashboard_df.apply(level3_cond,axis=1)

#----------Creating Level-02--------------------------------------------------------------------------------------------------

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

dashboard_df['LEVEL02'] = dashboard_df.apply(level2_condition,axis=1)

#----------------Creating Level-01 Column----------------------------------------------------------------------------------------------------------

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

dashboard_df['LEVEL01'] = dashboard_df.apply(level01_condition,axis=1)

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

dashboard_df[['Item','Material_Type','Division','Cust_Facing_Loc','Sales_org','Lag']] = dashboard_df['Key'].str.split(',',5,expand=True)

dashboard_df['Abs_Error_Cons'] = dashboard_df['L12M_Actual_Sum'] - dashboard_df['L12M_Consensus_Sum']

dashboard_df['Abs_Error_Cons'] = dashboard_df['Abs_Error_Cons'].apply(lambda x:abs(x))

#-------------------------------------------------------------------------------------------------------------------------------

def FA(row):
    if row['L12M_Actual_Sum'] == 0:
        val = np.NaN
    elif (row['Abs_Error_Cons']/row['L12M_Actual_Sum']) >1:
        val = 0
    elif (row['Abs_Error_Cons']/row['L12M_Actual_Sum']) <=1:
        val = (1-(row['Abs_Error_Cons']/row['L12M_Actual_Sum']))
    else :
        val = np.NaN
    return val

dashboard_df['FA'] = dashboard_df.apply(FA,axis=1)

##-------------------Bais----------------------------------------------------------------------------------------------------------------
def bais_cond(row):
    if row['L12M_Consensus_Sum'] > 0:
        val = (row['L12M_Consensus_Sum']-row['L12M_Actual_Sum'])/row['L12M_Consensus_Sum']
    else :
        val = np.NaN
    return val
dashboard_df['Bais'] = dashboard_df.apply(bais_cond,axis=1)
dashboard_df = dashboard_df.convert_dtypes()
#st.write('Dashboard Table Created & Shape is :',dashboard_df.shape)

###--------------------------###-----------------------------------------------###--------------------------------------------------

loss_df = dashboard_df[['Date','Item','LEVEL01','LEVEL02','LEVEL03','Division','Material_Type','Lag','Sales_org','Cust_Facing_Loc']]
loss_df['Forecast_Consensus'] = dashboard_df['L12M_Consensus_Sum']
loss_df['Actual'] = dashboard_df['L12M_Actual_Sum']
loss_df['Abs_Error_Cons'] = dashboard_df['L12M_Actual_Sum'] - dashboard_df['L12M_Consensus_Sum']
loss_df['Abs_Error_Cons'] = loss_df['Abs_Error_Cons'].apply(lambda x:abs(x))
loss_df['FA'] = dashboard_df['FA']
loss_df['BAIS'] = dashboard_df['Bais']


#st.write('Loss Table Shape is :',loss_df.shape)

##----------------------------------##----------------------------------------------------------------------------------------------
##----------------------------------##------------------------------------------------------------------------------------------------


#---------Data Loading-----------------------------------------------------------------------------------------------------------------

#@st.cache
#def load_data(data):
#    data = dashboard_df
    #dash_path = r"C:\Users\kendrav\OneDrive - Ecolab\Documents\Automation_Projects\EU FA Loss Analysis Dev\FA_Loss_Analysis\Data.xlsx"
    #dashboard_df = pd.read_excel(dash_path,sheet_name="Dashboard",engine="openpyxl")
    #dashboard_df = dashboard_df.convert_dtypes()
    #dashboard_df= dashboard_df.astype({'Item':'string', 'Div':'string', 'Cust_Facing_Loc':'string', 'Sales_org':'string'})
#    return data

imag = Image.open(r'C:\Users\kendrav\OneDrive - Ecolab\Documents\Automation_Projects\EU FA Loss Analysis Dev\FA_Loss_Analysis\Ecolab_logo.png')
st.image(imag)
st.title(":bar_chart: FA Loss Analysis Dashboard")
#st.markdown("#__#")


#dashboard_df = load_data(data = dashboard_df)
#st.write('Data Loaded!!!')
#-----Sidebar--------------------------------------------------------------------------------------------------------------------------

level2_list = loss_df['LEVEL02'].unique().tolist()
level1_list = loss_df['LEVEL01'].unique().tolist()
level3_list = loss_df['LEVEL03'].unique().tolist()
lag_list = loss_df['Lag'].unique().tolist()
material_list = loss_df['Material_Type'].unique().tolist()
sales_org_list = loss_df['Sales_org'].unique().tolist()
item_list = loss_df['Item'].unique().tolist()
div_list = loss_df['Division'].unique().tolist()
cust_fac_list = loss_df['Cust_Facing_Loc'].unique().tolist()

st.sidebar.header("Please Filter Here")
level1 = st.sidebar.multiselect("Select LEVEL-01", options=level1_list,default=level1_list)
level2 = st.sidebar.multiselect("Select LEVEL-02", options=level2_list,default=level2_list)
level3 = st.sidebar.multiselect("Select LEVEL-03", options=level3_list,default=level3_list)
material_type = st.sidebar.multiselect("Select Material Type", options=material_list,default=material_list)
sales_org = st.sidebar.multiselect("Select Sales Organization", options=sales_org_list,default=sales_org_list)
item = st.sidebar.multiselect("Select Item",options=item_list,default=item_list)
division = st.sidebar.multiselect("Select Division", options=div_list,default=div_list)
cust_facing_loc = st.sidebar.multiselect("Select Cust Facing Loc", options=cust_fac_list,default=cust_fac_list)
lag = st.sidebar.selectbox("Select LAG", options=lag_list)
date = st.sidebar.date_input('Select Date',value=(first_date,last_date))


#-----------Selection----------------------------------------------------------------------------------------------------------------------------------------------

df_selection = dashboard_df.query(
    """
    LEVEL01 == @level1 & LEVEL02 == @level2 & LEVEL03 == @level3 & Material_Type == @material_type & Lag ==@lag & Sales_org == @sales_org & Item == @item & Division ==@division & Cust_Facing_Loc ==@cust_facing_loc & Date == @date
    """
    )


##--------------------------------------------------------------
cust_funnel_dat=pd.DataFrame(data=loss_df.groupby(['Cust_Facing_Loc']).sum()[['Abs_Error_Cons']].nlargest(10,columns='Abs_Error_Cons'))
cust_funnel_dat.reset_index(inplace=True)
sales_funnel=pd.DataFrame(loss_df.groupby(['Sales_org']).sum()[['Abs_Error_Cons']].nlargest(10,columns='Abs_Error_Cons'))
sales_funnel.reset_index(inplace=True)
sku_funnel=pd.DataFrame(loss_df.groupby(['Item']).sum()[['Abs_Error_Cons']].nlargest(10,columns='Abs_Error_Cons'))
sku_funnel.reset_index(inplace=True)

#st.write(df_selection)
#selection= df_selection.shape
#st.markdown(f"*Number of rows Selected :{selection}*")
#-----------Main Page-------------------------------------------------------------------------------------------------------------------------

left_column, middle_column, right_column = st.columns(3)
with left_column:
    level1_pie = px.pie(df_selection,
                title='Level-01 Percentage',
                names='LEVEL01')
    st.plotly_chart(level1_pie,use_container_width = True, sharing='streamlit')

with middle_column:
    level2_pie = px.pie(df_selection,
                    title='Level-02 Percentage',
                    names='LEVEL02')
    st.plotly_chart(level2_pie,use_container_width = True, sharing='streamlit')

with right_column:
    level3_pie = px.pie(df_selection,
                    title='Level-03 Percentage',
                    names='LEVEL03')
    st.plotly_chart(level3_pie,use_container_width = True, sharing='streamlit')

st.markdown("""--""")

left_column, middle_column, right_column = st.columns(3)

with left_column:
    cust_loc = px.funnel(cust_funnel_dat,
                title='Top 10 customer Facing Location by Consensus Forecast Error',
                x='Abs_Error_Cons',
                y='Cust_Facing_Loc')
    st.plotly_chart(cust_loc,use_container_width = True, sharing='streamlit')

with middle_column:
    sales_org = px.funnel(sales_funnel,
                    title='Top 10 Sales Org by Consensus Forecast Error',
                    x='Abs_Error_Cons',
                    y='Sales_org')
    st.plotly_chart(sales_org,use_container_width = True, sharing='streamlit')

with right_column:
    sku = px.funnel(sku_funnel,
                    title='Top 10 SKUs by Consensus Forecast Error',
                    x='Abs_Error_Cons',
                    y='Item')
    st.plotly_chart(sku,use_container_width = True, sharing='streamlit')


st.write(loss_df)

# ---- HIDE STREAMLIT STYLE -----------------------------------------------------------------------------------------------
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

###---------------------------------------------------------------------------------------------------------------------------------










