import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'C:\Users\roy_shaw\Desktop\Completed Reports\CC KPI_and_Live Chat')

df_OECS = pd.read_excel('CC KPI & Live Chat Reports FY23.xlsx', sheet_name='OECS', usecols='A:R')
df_GUY = pd.read_excel('CC KPI & Live Chat Reports FY23.xlsx', sheet_name='Guyana', usecols='A:R')
df_JAM = pd.read_excel('CC KPI & Live Chat Reports FY23.xlsx', sheet_name='Jamaica', usecols='A:Q,S')
df_TTO = pd.read_excel('CC KPI & Live Chat Reports FY23.xlsx', sheet_name='Trinidad and Tobago', usecols='A:Q,S')
df_BAR = pd.read_excel('CC KPI & Live Chat Reports FY23.xlsx', sheet_name='Barbados', usecols='A:Q,S')

df_OECS['Country'] = ['OECS'] * len(df_OECS['Date'])
df_GUY['Country'] = ['Guyana'] * len(df_GUY['Date'])
df_JAM['Country'] = ['Jamaica'] * len(df_JAM['Date'])
df_TTO['Country'] = ['Trinidad and Tobago'] * len(df_TTO['Date'])
df_BAR['Country'] = ['Barbados'] * len(df_BAR['Date'])

df_OECS_ = pd.read_excel('CC KPI & Live Chat Reports FY24.xlsx', sheet_name='OECS', usecols='A:R')
df_GUY_ = pd.read_excel('CC KPI & Live Chat Reports FY24.xlsx', sheet_name='Guyana', usecols='A:R')
df_JAM_ = pd.read_excel('CC KPI & Live Chat Reports FY24.xlsx', sheet_name='Jamaica', usecols='A:Q,S')
df_TTO_ = pd.read_excel('CC KPI & Live Chat Reports FY24.xlsx', sheet_name='Trinidad and Tobago', usecols='A:Q,S')
df_BAR_ = pd.read_excel('CC KPI & Live Chat Reports FY24.xlsx', sheet_name='Barbados', usecols='A:Q,S')

df_OECS_['Country'] = ['OECS'] * len(df_OECS_['Date'])
df_GUY_['Country'] = ['Guyana'] * len(df_GUY_['Date'])
df_JAM_['Country'] = ['Jamaica'] * len(df_JAM_['Date'])
df_TTO_['Country'] = ['Trinidad and Tobago'] * len(df_TTO_['Date'])
df_BAR_['Country'] = ['Barbados'] * len(df_BAR_['Date'])

df_OECS_agg = pd.concat([df_OECS, df_OECS_])
df_GUY_agg = pd.concat([df_GUY, df_GUY_])
df_JAM_agg = pd.concat([df_JAM, df_JAM_])
df_TTO_agg = pd.concat([df_TTO, df_TTO_])
df_BAR_agg = pd.concat([df_BAR, df_BAR_])

df_main = pd.concat([df_OECS_agg, df_GUY_agg, df_JAM_agg, df_TTO_agg, df_BAR_agg], ignore_index=False)

def create_dashboard(df_main, fiscal, country, date_range):
   # Creating dashboard page
    st.set_page_config(page_title='CC KPI Dashboard', page_icon=":bar_chart:", layout="wide")

    # Purposing the filters
    st.sidebar.header("Filter Here for Data Table:")
    fiscal = st.sidebar.multiselect(
        "Select the Fiscal:",
        options=df_main['Fiscal'].unique(),
        default=df_main['Fiscal'].unique()
    )

    country = st.sidebar.multiselect(
        "Select the Country:",
        options=df_main['Country'].unique(),
        default=df_main['Country'].unique()
    )

    # Exclude missing values from date column
    filtered_dates = df_main['Date'].dropna()

    # Convert date range to string representations
    selected_min_date = date_range[0].strftime("%Y-%m-%d")
    selected_max_date = date_range[1].strftime("%Y-%m-%d")

    # Convert date range to pandas Timestamp objects
    selected_min_date = st.sidebar.date_input("Select the minimum date:", value=df_main['Date'].min())
    selected_max_date = st.sidebar.date_input("Select the maximum date:", value=df_main['Date'].max())

    # Convert the selected dates to pandas Timestamp objects
    selected_min_date = pd.Timestamp(selected_min_date)
    selected_max_date = pd.Timestamp(selected_max_date)

    selected_field = st.sidebar.selectbox(
    "Select the Field:",
    options=['Date'],
    index=0
    )

    df_selection = df_main.query("Fiscal == @fiscal & Country == @country")
    df_selection = df_selection[df_selection[selected_field].between(selected_min_date, selected_max_date)]

    # Display the table
    st.title("Table of Contact Center KPIs and measurements")
    st.dataframe(df_selection)

    # Visual 1
    # Convert 'Date' column to datetime and set it to index
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main.set_index('Date', inplace=True)

    #Apply filters to the merged_calls DataFrame
    merged_calls = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    merged_calls = merged_calls['Calls Off'].resample('W').sum().to_frame(name='Calls Off')
    merged_calls['Calls Ans'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'Calls Ans'].resample('W').sum()

    merged_calls = merged_calls.reset_index()

    st.title("Weekly Calls Offered vs Calls Accepted")

    fig, ax = plt.subplots()
    ax.plot(merged_calls['Date'], merged_calls['Calls Off'], label='Calls Offered')
    ax.plot(merged_calls['Date'], merged_calls['Calls Ans'], label='Calls Answered')
    ax.set_xlabel('Week')
    ax.set_ylabel('Call Volume')
    ax.set_title('Weekly Offered vs Accepted Calls')
    ax.legend()
    ax.plot()

    date_range1 = st.slider(
        'Select Date Range',
        merged_calls['Date'].min().to_pydatetime(),
        merged_calls['Date'].max().to_pydatetime(),
        (merged_calls['Date'].min().to_pydatetime(), merged_calls['Date'].max().to_pydatetime()),
        key='date_range'
    )

    filtered_data = merged_calls[(merged_calls['Date'] >= date_range1[0]) & (merged_calls['Date'] <= date_range1[1])]

    fig, ax = plt.subplots()
    ax.plot(filtered_data['Date'], filtered_data['Calls Off'], label='Calls Off')
    ax.plot(filtered_data['Date'], filtered_data['Calls Ans'], label='Calls Ans')
    ax.legend()

    st.pyplot(fig)

    #Visual 2
    st.title("Daily Calls Offered vs Calls Accepted")

    daily_calls = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    daily_calls = daily_calls['Calls Off'].resample('D').sum().to_frame(name='Calls Off')
    daily_calls['Calls Ans'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'Calls Ans'].resample('D').sum()

    daily_calls = daily_calls.reset_index()

    # Create a date range slider
    date_range_ = st.slider(
        'Select Date Range',
        daily_calls['Date'].min().to_pydatetime(),
        daily_calls['Date'].max().to_pydatetime(),
        (daily_calls['Date'].min().to_pydatetime(), daily_calls['Date'].max().to_pydatetime()),
        key='date_range_'
    )

    filtered_data = daily_calls[(daily_calls['Date'] >= date_range_[0]) & (daily_calls['Date'] <= date_range_[1])]
    filtered_offered = filtered_data['Calls Off']
    filtered_answered = filtered_data['Calls Ans']

    bar_width = 0.35

    x = np.arange(len(filtered_offered))

    fig_filtered, ax_filtered = plt.subplots()
    ax_filtered.bar(x, filtered_offered, width=bar_width, label='Calls Offered')
    ax_filtered.bar(x + bar_width, filtered_answered, width=bar_width, label='Calls Answered')

    # Set the x-axis tick locations and labels
    tick_freq = 14
    tick_indices = np.arange(0, len(filtered_data['Date']), tick_freq)
    tick_labels = filtered_data.loc[tick_indices, 'Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.set_ylabel('Call Volume')

    # Customize the filtered plot
    ax_filtered.legend()

    st.pyplot(fig_filtered)

    #Visual 3
    st.title("Percent Change Week over Week Calls Offered and Calls Answered")

    daily_percent = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    daily_percent = daily_percent['WoW % change (offer)'].resample('D').sum().to_frame(name='WoW % change (offer)')
    daily_percent['WoW % change (ans)'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'WoW % change (ans)'].resample('D').sum()

    daily_percent=daily_percent.reset_index()

    filtered_data = daily_percent[(daily_percent['Date'] >= date_range_[0]) & (daily_percent['Date'] <= date_range_[1])]
    filtered_offered = filtered_data['WoW % change (offer)']*100
    filtered_answered = filtered_data['WoW % change (ans)']*100

    bar_width = 0.35

    x = np.arange(len(filtered_offered))

    fig_filtered, ax_filtered = plt.subplots()
    ax_filtered.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_filtered.bar(x, filtered_offered, width=bar_width, label='Percent Change Calls Offered')
    ax_filtered.bar(x + bar_width, filtered_answered, width=bar_width, label='Percent Change Calls Answered')

    # Set the x-axis tick locations and labels
    tick_freq = 14
    tick_indices = np.arange(0, len(filtered_data['Date']), tick_freq)
    tick_labels = filtered_data.loc[tick_indices, 'Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.set_ylabel('Call Volume')

    # Customize the filtered plot
    ax_filtered.legend()
    st.pyplot(fig_filtered)

    #Visual 4
    st.title("Service Level vs ABR vs AU")

    df_metric = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    df_metric = df_metric['SL'].resample('D').sum().to_frame(name='SL')
    df_metric['AU'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'AO/AU'].resample('D').sum()
    df_metric['ABR'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'ABR'].resample('D').sum()

    df_metric = df_metric.reset_index()

    filtered_data = df_metric[(df_metric['Date'] >= date_range_[0]) & (df_metric['Date'] <= date_range_[1])]
    filtered_SL = filtered_data['SL']*100
    filtered_AU = filtered_data['AU']*100
    filtered_ABR = filtered_data['ABR']*100

    bar_width = 0.35

    x = np.arange(len(filtered_SL))

    # Set the x-axis tick locations and labels
    tick_freq = 14
    tick_indices = np.arange(0, len(filtered_data['Date']), tick_freq)
    tick_labels = filtered_data.loc[tick_indices, 'Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    fig_filtered, ax_filtered = plt.subplots()

    # Y-axis values for line chart 1
    y1 = filtered_ABR
    ax_filtered.plot(x, y1, color='blue', label='ABR Line Chart')

    # Y-axis values for line chart 2
    y2 = filtered_SL
    ax_filtered.plot(x, y2, color='orange', label='SL Line Chart')

    y3 = filtered_AU  # Y-axis values for bar chart
    bar_width = 0.4
    ax_filtered.bar(x, y3, width=bar_width, color='gray', align='center', label='AU Bar Chart')

    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_filtered.set_ylabel('KPI Reading')

    ax_filtered.legend()
    st.pyplot(fig_filtered)

create_dashboard(df_main, fiscal=[], country=[], date_range=(df_main['Date'].min(), df_main['Date'].max()))
