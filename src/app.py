import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly
import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#from joblib import dump, load
from prophet import Prophet
from datetime import datetime, timedelta
from mlforecast import MLForecast
from numba import jit
#from awesome_streamlit.shared import components
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import openpyxl
from dateutil.relativedelta import relativedelta



#To get the data
#df = pd.read_excel('Global_Cde_sites.xlsx')
df = pd.read_excel('Global_Cde_sites_ToTest.xlsx')
df_test = pd.read_excel('Global_Cde_sites_Test.xlsx')
#df_test.rename(columns = {'DATE':'date'}, inplace = True)

default_start_date = datetime.today() - relativedelta(months=1)

#The App title
st.title('Commandes Forecaster')

Site = ('006', '007', '008', '010', '102', 'LEO')

# "with" notation
with st.sidebar:  
    Site = st.selectbox('Site', Site)
    forecast_date = st.date_input('Forecast Start Date', value=default_start_date)
    model_selection = st.selectbox('Model Selection', ['Prophet', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor'])
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    run_forecast = st.button('Forecast')
    
st.write('Site selected:', Site)
st.write('Model selected:', model_selection)
st.write('Days of forecast:', nb_days)
st.write('Date selected:', forecast_date)

# To compute the performances of each model
def performances(y, y_hat):
    delta_y = np.square(y - y_hat)
    mse = np.nanmean(delta_y)
    rmse = np.sqrt(np.nanmean(delta_y))
    absolute_diff = np.abs(delta_y)
    mae = np.mean(absolute_diff)
    return mse, rmse, mae



# The Dataframe by site selected
site_df = df[(df['SITE'] == Site)]
# The Visualization of the Original Dataframe by site selected
st.subheader("Original DataFrame")
st.dataframe(data=site_df, width=600, height=300)

#The Visualization of the chart by route selected
site_df_pre = df.query(f'SITE == "{Site}"')
site_df_pre = site_df_pre.groupby('DATE').agg(cdes_total=('COMMANDE', 'sum')).reset_index()
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=site_df_pre['DATE'], y=site_df_pre['cdes_total'], fill='tozeroy', name=f'{Site}'), row=1, col=1)
graph_label = f'Site selected : {Site}'
fig.update_layout(title=graph_label)



mae = 0.0
rmse = 0.0
r_squared = 0.0


if run_forecast:
    # Build route traffic dataframe
    site_df = df.query(f'SITE == "{Site}"')
    site_df = site_df.groupby('DATE').agg(cdes_total=('COMMANDE', 'sum')).reset_index()
    # the period considered
    forecast_dates = pd.date_range(forecast_date, periods=nb_days)  
    if model_selection == 'Prophet':
        # Prepare the data for prophet
        forecast_date = pd.to_datetime(forecast_date)
        site_df = site_df[site_df['DATE'] <= forecast_date]
        prophet_df = site_df[['DATE', 'cdes_total']].rename(columns={'DATE': 'ds', 'cdes_total': 'y'})
        # Create the model
        model_prophet = Prophet()
        # Fit the model
        model_prophet.fit(prophet_df)
        # Prediction using the forecast dates
        future = pd.DataFrame({'ds': forecast_dates})
        forecast = model_prophet.predict(future)
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'DATE': forecast_dates, 'cdes_total': forecast['yhat']})
        # Compute the performance of the prediction
        true_values = site_df['cdes_total'].values[-nb_days:]
        predicted_values = forecast['yhat'].values
        mse, rmse, mae = performances(true_values, predicted_values)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(dash='dash', color='yellow'), name='Prophet'), row=1, col=1)

    elif model_selection == 'XGBRegressor':
        # Prepare the data for the XGBRegressor
        ref_date = np.min(site_df['DATE']).to_pydatetime()
        X_train = (site_df['DATE'] - ref_date).dt.days.values.reshape(-1, 1)
        X_train_forecast = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)
        y_train = site_df['cdes_total'].values
        # Data Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_forecast_scaled = scaler.transform(X_train_forecast)
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }     
        # Create the model
        xgb_model = XGBRegressor()        
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5)        
        # Fit the model
        #xgb_model.fit(X_train, y_train)        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_        
        # Prediction using the forecast dates
        #xgb_predictions = xgb_model.predict(X_train_forecast)        
        xgb_predictions = best_model.predict(X_train_forecast_scaled)        
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'DATE': forecast_dates, 'cdes_total': xgb_predictions})
        # Compute the performance of the prediction
        true_values = site_df['cdes_total'].values[-nb_days:]
        predicted_values = forecast_df['cdes_total'].values
        mse, rmse, mae = performances(true_values, predicted_values)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast_dates, y=xgb_predictions, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='XGBRegressor'), row=1, col=1)

    elif model_selection == 'RandomForestRegressor':
        # Prepare the data for the RandomForestRegressor
        ref_date = np.min(site_df['DATE']).to_pydatetime()
        X_train = (site_df['DATE'] - ref_date).dt.days.values.reshape(-1, 1)
        X_train_forecast = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)
        y_train = site_df['cdes_total'].values
        # Data Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_forecast_scaled = scaler.transform(X_train_forecast)
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        # Create the model
        rf_model = RandomForestRegressor()
        grid_search = GridSearchCV(rf_model, param_grid, cv=5)
        # Fit the model
        #rf_model.fit(X_train, y_train)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        # Fit the model
        best_model.fit(X_train_scaled, y_train)

        # Prediction using the forecast dates
        #rf_predictions = rf_model.predict(X_train_forecast)
        rf_predictions = best_model.predict(X_train_forecast_scaled)

        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'DATE': forecast_dates, 'cdes_total': rf_predictions})
        # Compute the performance of the prediction
        true_values = site_df['cdes_total'].values[-nb_days:]
        predicted_values = forecast_df['cdes_total'].values
        mse, rmse, mae = performances(true_values, predicted_values)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast_dates, y=rf_predictions, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='RandomForestRegressor'), row=1, col=1)
        
    elif model_selection == 'LGBMRegressor':
        # Prepare the data for the LGBMRegressor
        ref_date = np.min(site_df['DATE']).to_pydatetime()
        X_train = (site_df['DATE'] - ref_date).dt.days.values.reshape(-1, 1)
        X_train_forecast = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)
        y_train = site_df['cdes_total'].values
        # Create the model
        lgb_model = LGBMRegressor()
        # Fit the model
        lgb_model.fit(X_train, y_train)        
        # Prediction using the forecast dates
        lgb_predictions = lgb_model.predict(X_train_forecast)        
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'DATE': forecast_dates, 'cdes_total': lgb_predictions})
        # Compute the performance of the prediction
        true_values = site_df['cdes_total'].values[-nb_days:]
        mse, rmse, mae = performances(true_values, lgb_predictions)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast_dates, y=lgb_predictions, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='LGBMRegressor'), row=1, col=1)
      
    # Unpdate the graph with the performances value as label
    graph_label += f"<br>MSE: {mse:.2f} - RMSE: {rmse:.2f} - <br>MAE: {mae:.2f}"
    fig.update_layout(title=graph_label)

    
# Show the praph with the forecast
st.plotly_chart(fig)
# Visualize the Forecast dataframe
if run_forecast:
    st.subheader("Forecast DataFrame")
    st.dataframe(data=forecast_df, width=600, height=300)
    site_df_test = df_test[(df_test['SITE'] == Site)]
    compare_df = pd.merge(forecast_df, site_df_test, left_on='DATE', right_on='DATE', how='outer')
    st.dataframe(data=compare_df, width=600, height=300)
    
    

    
    
