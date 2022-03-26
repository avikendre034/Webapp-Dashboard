
               
import pandas as pd
import streamlit as st
import plotly_express as px

st.write('Hellow')

#st.file_uploader('Upload Final Loss Analysis File')
@st.experimental_singleton
def load_dat():
    df = pd.read_excel('loss_df.xlsx')
    return df

df = load_dat()

cust_funnel_dat=pd.DataFrame(data=df.groupby(['Cust_Facing_Loc']).sum()[['Abs_Error_Cons']].nlargest(10,columns='Abs_Error_Cons'))
cust_funnel_dat.reset_index(inplace=True)
sales_funnel=pd.DataFrame(df.groupby(['Sales_org']).sum()[['Abs_Error_Cons']].nlargest(10,columns='Abs_Error_Cons'))
sales_funnel.reset_index(inplace=True)
sku_funnel=pd.DataFrame(df.groupby(['Item']).sum()[['Abs_Error_Cons']].nlargest(10,columns='Abs_Error_Cons'))
sku_funnel.reset_index(inplace=True)


left_column, middle_column, right_column = st.columns(3)
with left_column:
    level1_pie = px.pie(df,
                title='Level-01 Percentage',
                names='LEVEL01')
    st.plotly_chart(level1_pie,use_container_width = True, sharing='streamlit')

with middle_column:
    level2_pie = px.pie(df,
                    title='Level-02 Percentage',
                    names='LEVEL02')
    st.plotly_chart(level2_pie,use_container_width = True, sharing='streamlit')

with right_column:
    level3_pie = px.pie(df,
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
    st.plotly_chart(cust_loc)

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


st.write(df)