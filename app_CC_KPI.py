import pandas as pd
import os
import datetime
from datetime import timedelta
import streamlit as st
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import openpyxl

url_FY23 = r'https://github.com/lightful876/Unicomer_Regional/blob/6fc252686c26cca52d1f8ed87fce0da6d9f3c615/CC%20KPI%20%26%20Live%20Chat%20Reports%20FY23.xlsx'
url_FY24 = 'https://raw.githubusercontent.com/lightful876/Unicomer_Regional/blob/6fc252686c26cca52d1f8ed87fce0da6d9f3c615/CC%20KPI%20%26%20Live%20Chat%20Reports%20FY24.xlsx'
url_FY25 = 'https://raw.githubusercontent.com/lightful876/Unicomer_Regional/blob/6fc252686c26cca52d1f8ed87fce0da6d9f3c615/CC%20KPI%20%26%20Live%20Chat%20Reports%20FY25.xlsx'

df_OECS = pd.read_excel(url_FY23, sheet_name='OECS', usecols='A:X', engine='openpyxl')
df_GUY = pd.read_excel(url_FY23, sheet_name='Guyana', usecols='A:X')
df_JAM = pd.read_excel(url_FY23, sheet_name='Jamaica', usecols='A:X')
df_TTO = pd.read_excel(url_FY23, sheet_name='Trinidad and Tobago', usecols='A:X')
df_BAR = pd.read_excel(url_FY23, sheet_name='Barbados', usecols='A:X')

df_OECS['Country'] = ['OECS'] * len(df_OECS['Date'])
df_GUY['Country'] = ['Guyana'] * len(df_GUY['Date'])
df_JAM['Country'] = ['Jamaica'] * len(df_JAM['Date'])
df_TTO['Country'] = ['Trinidad and Tobago'] * len(df_TTO['Date'])
df_BAR['Country'] = ['Barbados'] * len(df_BAR['Date'])

df_OECS_ = pd.read_excel(url_FY24, sheet_name='OECS', usecols='A:X', engine='openpyxl')
df_GUY_ = pd.read_excel(url_FY24, sheet_name='Guyana', usecols='A:X', engine='openpyxl')
df_JAM_ = pd.read_excel(url_FY24, sheet_name='Jamaica', usecols='A:X', engine='openpyxl')
df_TTO_ = pd.read_excel(url_FY24, sheet_name='Trinidad and Tobago', usecols='A:X', engine='openpyxl')
df_BAR_ = pd.read_excel(url_FY24, sheet_name='Barbados', usecols='A:X', engine='openpyxl')
df_BEL_ = pd.read_excel(url_FY24, sheet_name='Belize', usecols='A:X', engine='openpyxl')

df_OECS_['Country'] = ['OECS'] * len(df_OECS_['Date'])
df_GUY_['Country'] = ['Guyana'] * len(df_GUY_['Date'])
df_JAM_['Country'] = ['Jamaica'] * len(df_JAM_['Date'])
df_TTO_['Country'] = ['Trinidad and Tobago'] * len(df_TTO_['Date'])
df_BAR_['Country'] = ['Barbados'] * len(df_BAR_['Date'])
df_BEL_['Country'] = ['Belize'] * len(df_BEL_['Date'])

df_OECS_FY25 = pd.read_excel(url_FY25, sheet_name='OECS', usecols='A:X', engine='openpyxl')
df_GUY_FY25 = pd.read_excel(url_FY25, sheet_name='Guyana', usecols='A:X', engine='openpyxl')
df_JAM_FY25 = pd.read_excel(url_FY25, sheet_name='Jamaica', usecols='A:X', engine='openpyxl')
df_TTO_FY25 = pd.read_excel(url_FY25, sheet_name='Trinidad and Tobago', usecols='A:X', engine='openpyxl')
df_BAR_FY25 = pd.read_excel(url_FY25, sheet_name='Barbados', usecols='A:X', engine='openpyxl')
df_BEL_FY25 = pd.read_excel(url_FY25, sheet_name='Belize', usecols='A:X', engine='openpyxl')

df_OECS_FY25['Country'] = ['OECS'] * len(df_OECS_FY25['Date'])
df_GUY_FY25['Country'] = ['Guyana'] * len(df_GUY_FY25['Date'])
df_JAM_FY25['Country'] = ['Jamaica'] * len(df_JAM_FY25['Date'])
df_TTO_FY25['Country'] = ['Trinidad and Tobago'] * len(df_TTO_FY25['Date'])
df_BAR_FY25['Country'] = ['Barbados'] * len(df_BAR_FY25['Date'])
df_BEL_FY25['Country'] = ['Belize'] * len(df_BEL_FY25['Date'])

df_OECS_agg = pd.concat([df_OECS, df_OECS_, df_OECS_FY25])
df_GUY_agg = pd.concat([df_GUY, df_GUY_, df_GUY_FY25])
df_JAM_agg = pd.concat([df_JAM, df_JAM_, df_JAM_FY25])
df_TTO_agg = pd.concat([df_TTO, df_TTO_, df_TTO_FY25])
df_BAR_agg = pd.concat([df_BAR, df_BAR_, df_BAR_FY25])
df_BEL_agg = pd.concat([df_BEL_, df_BEL_FY25])

df_main = pd.concat([df_OECS_agg, df_GUY_agg, df_JAM_agg, df_TTO_agg, df_BAR_agg, df_BEL_FY25], ignore_index=False)

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
    df_selection_filtered = df_selection[df_selection['Calls Off'] != 0]

    calls_sum = df_selection_filtered['Calls Off'].sum()
    calls_ans = df_selection_filtered['Calls Ans'].sum()
    svl_avg = df_selection_filtered['SL'].mean()
    abr_avg = df_selection_filtered['ABR'].mean()
    ans_avg = df_selection_filtered['ANS'].mean()
    au_avg = df_selection_filtered['AU'].mean()
    aa_avg = df_selection_filtered['AA'].mean()

    st.write('Calls Offered', f'sum of calls offered over specified interval is {calls_sum:.0f}')
    st.write('Calls Answered', f'sum of calls answered over specified interval is {calls_ans:.0f}')
    st.write('Service Level', f'average of Service Level over specified interval is {svl_avg:.3f}')
    st.write('Abandonment Rate', f'average of ABR over specified interval is {abr_avg:.3f}')
    st.write('Answer Rate', f'average of ANS over specified interval is {ans_avg:.3f}')
    st.write('Agent Utilization', f'average of AU over specified interval is {au_avg:.3f}')
    st.write('Agent Adherence', f'average of Agent Adherence over specified interval in {aa_avg:.3f}')

    st.title('Distribution of Calls by Day')
    day_order = ['D', 'M', 'T', 'W', 'R', 'F', 'S']
    df_selection_dow = df_selection[(df_selection['Day of the week'] != 'HOL') & (df_selection['Day of the week'] != 'D')]
    df_dow = df_selection_dow.dropna(subset=['Calls Off']).groupby(['Day of the week'])['Calls Off'].sum()
    df_dow_ans = df_selection_dow.dropna(subset=['Calls Ans']).groupby(['Day of the week'])['Calls Ans'].sum()
    df_dow = df_dow.reindex(day_order)
    df_dow_ans = df_dow_ans.reindex(day_order)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(day_order, df_dow, label='Calls Offered')
    ax.bar(day_order, df_dow_ans, label='Calls Ans')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Call Volume')
    ax.set_title('Grouped Daily Offered vs Accepted Calls')
    st.pyplot(fig.figure)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(merged_calls['Date'], merged_calls['Calls Off'], label='Calls Offered')
    ax.plot(merged_calls['Date'], merged_calls['Calls Ans'], label='Calls Answered')

    date_range1 = st.slider(
        'Select Date Range',
        merged_calls['Date'].min().to_pydatetime(),
        merged_calls['Date'].max().to_pydatetime(),
        (merged_calls['Date'].min().to_pydatetime(), merged_calls['Date'].max().to_pydatetime()),
        key='date_range'
    )

    filtered_data = merged_calls[(merged_calls['Date'] >= date_range1[0]) & (merged_calls['Date'] <= date_range1[1])]

    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.plot(filtered_data['Date'], filtered_data['Calls Off'], label='Calls Off')
    ax.plot(filtered_data['Date'], filtered_data['Calls Ans'], label='Calls Ans')
    ax.set_xlabel('Week Of')
    ax.set_ylabel('Call Volume')
    ax.set_title('Weekly Offered vs Accepted Calls')
    ax.legend()
    ax.plot()

    plt.tight_layout()
    ax.legend()

    plt.subplots_adjust(bottom=0.2)
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

    min_date, max_date = date_range_
    selected_data = daily_calls[(daily_calls['Date'] >= min_date) & (daily_calls['Date'] <= max_date)]

    x = selected_data['Date']
    y1 = selected_data['Calls Off']
    y2 = selected_data['Calls Ans']

    bar_width = 0.35
    x_numeric = np.arange(len(x))

    fig_filtered, ax_filtered = plt.subplots(figsize=(10, 6))
    ax_filtered.bar(x_numeric, y1, width=bar_width, align='center', label='Calls Offered')
    ax_filtered.bar(x_numeric+bar_width, y2, width=bar_width, align='edge', label='Calls Answered')

    # Set the x-axis tick locations and labels
    tick_freq = 14
    tick_indices = np.arange(0, len(x_numeric), tick_freq)
    tick_labels = x.iloc[tick_indices].apply(lambda x: x.strftime('%Y-%m-%d'))
    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.set_ylabel('Call Volume')

    plt.tight_layout()

    # Customize the filtered plot
    ax_filtered.legend()

    plt.subplots_adjust(bottom=0.2)

    st.pyplot(fig_filtered)

    #Visual 3
    st.title("Percent Change Week over Week Calls Offered and Calls Answered")

    daily_percent = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    daily_percent = daily_percent['Winsorized Values - Offered'].resample('D').sum().to_frame(name='Winsorized Values - Offered')
    daily_percent['Winsorized Values - Ans'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'Winsorized Values - Ans'].resample('D').sum()

    daily_percent=daily_percent.reset_index()

    filtered_data = daily_percent[(daily_percent['Date'] >= date_range_[0]) & (daily_percent['Date'] <= date_range_[1])]

    x = selected_data['Date']
    y1 = filtered_data['Winsorized Values - Offered']*100
    y2 = filtered_data['Winsorized Values - Ans']*100

    bar_width = 0.35

    x_numeric = np.arange(len(x))

    fig_filtered, ax_filtered = plt.subplots(figsize=(10, 6))
    ax_filtered.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_filtered.bar(x_numeric, y1, width=bar_width, label='Percent Change Calls Offered')
    ax_filtered.bar(x_numeric + bar_width, y2, width=bar_width, label='Percent Change Calls Answered')

    # Set the x-axis tick locations and labels
    tick_freq = 14
    tick_indices = np.arange(0, len(x_numeric), tick_freq)
    tick_labels = x.iloc[tick_indices].apply(lambda x: x.strftime('%Y-%m-%d'))
    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.set_ylabel('Percentage WoW Call Value')

    plt.tight_layout()
    # Customize the filtered plot
    ax_filtered.legend()
    st.pyplot(fig_filtered)

    #Visual 4
    st.title("Service Level vs ABR vs AU")

    df_metric = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    df_metric = df_metric['SL'].resample('D').sum().to_frame(name='SL')
    df_metric['AU'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'AU'].resample('D').sum()
    df_metric['ABR'] = df_main[df_main['Fiscal'].isin(fiscal) & df_main['Country'].isin(country)][
        'ABR'].resample('D').sum()

    df_metric = df_metric.reset_index()

    filtered_data = df_metric[(df_metric['Date'] >= date_range_[0]) & (df_metric['Date'] <= date_range_[1])]
    x = selected_data['Date']
    y1 = filtered_data['SL']*100
    y2 = filtered_data['AU']*100
    y3 = filtered_data['ABR']*100

    bar_width = 0.35

    x_numeric = np.arange(len(x))

    # Set the x-axis tick locations and labels
    tick_freq = 14
    tick_indices = np.arange(0, len(x_numeric), tick_freq)
    tick_labels = x.iloc[tick_indices].apply(lambda x: x.strftime('%Y-%m-%d'))
    fig_filtered, ax_filtered = plt.subplots(figsize=(10, 6))

    ax_filtered.plot(x_numeric, y3, color='blue', label='ABR Line Chart')

    ax_filtered.plot(x_numeric, y1, color='orange', label='SL Line Chart')

    bar_width = 0.4
    ax_filtered.bar(x_numeric, y2, width=bar_width, color='gray', align='center', label='AU Bar Chart')

    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_filtered.set_ylabel('KPI Reading')

    ax_filtered.legend()
    st.pyplot(fig_filtered)

    #Visual 5
    st.title("Service Level vs Agents Logged")
    df_cntrl = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]

    df_cntrl.reset_index(inplace=True)
    filtered_data = df_cntrl[(df_cntrl['Date'] >= date_range_[0]) & (df_cntrl['Date'] <= date_range_[1])]
    x = selected_data['Date']
    y1 = filtered_data['SL'] * 100
    y2 = filtered_data['Agents Log']

    x_numeric = np.arange(len(x))

    x = x[:len(x_numeric)]
    y1 = y1[:len(x_numeric)]
    y2 = y2[:len(x_numeric)]

    tick_freq = 14
    tick_indices = np.arange(0, len(x_numeric), tick_freq)
    tick_labels = x.iloc[tick_indices].apply(lambda x: x.strftime('%Y-%m-%d'))


    offset = 0.2
    fig_filtered, ax_filtered = plt.subplots(figsize=(10, 6))

    ax_filtered.bar(x_numeric, y1, color='blue', label='Service Level')

    ax2 = ax_filtered.twinx()
    ax2.plot(x_numeric + offset, y2, color='orange', label='Agents Logged')
    ax2.set_ylabel('Count of Agents Logged')

    ax_filtered.set_xticks(tick_indices)
    ax_filtered.set_xticklabels(tick_labels, rotation=90)
    ax_filtered.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_filtered.set_ylabel('Service Level Achieved')

    ax_filtered.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    st.pyplot(fig_filtered)

    #Visual 6
    st.title("Linear-Gauge Chart of AHT")
    df_timer = df_main[(df_main['Fiscal'].isin(fiscal)) & (df_main['Country'].isin(country))]
    default_value = 0

    df_timer.reset_index(inplace=True)
    filtered_data = df_timer[(df_timer['Date'] >= date_range_[0]) & (df_timer['Date'] <= date_range_[1])]

    #filtered_data['AHT'] = [int(x) for x in filtered_data['AHT']]

    #print(filtered_data[~filtered_data['AHT'].apply(lambda x: isinstance(x, int))].index)

    average_AHT = np.mean(filtered_data['AHT'][filtered_data['AHT'] != 0])

    average_AHT_seconds = average_AHT

    fig = go.Figure(go.Indicator(domain = {'x': [0, 1], 'y': [0, 1]},
          value = average_AHT_seconds, mode = "gauge+number+delta",
          title = {'text': "Average Handle Time (Over selected Interval) in seconds"},
          delta={'reference': 300},
          gauge = {'axis': {'range': [None, 600]},
          'bar': {'color': "black"},
          'steps' : [{'range': [0, 300], 'color': "green"},
          {'range': [300, 480], 'color': "gold"},
          {'range': [480, 600], 'color': "red"}]}
          ))
    st.plotly_chart(fig)
    # Display the chart value
    st.write(f"Average AHT over period specified: {average_AHT_seconds:.0f} seconds")

create_dashboard(df_main, fiscal=[], country=[], date_range=(df_main['Date'].min(), df_main['Date'].max()))
