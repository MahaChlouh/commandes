import pandas as pd

import streamlit as st

FLUX = ('PR1-MAG', 'PR1-ETRANGER', 'PR3')

#df = pd.read_parquet('traffic_10lines.parquet')

st.title('Commandes Forecaster')


# "with" notation
with st.sidebar:  
    FLUX = st.selectbox(
    'FLUX', FLUX)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    run_forecast = st.button('Forecast')
    
st.write('FLUX selected:', FLUX)
st.write('Days of forecast:', nb_days)
st.write('Date selected:', forecast_date)

#st.write(df.query('home_airport = "{}"'.format(home_airport)).shape[0]