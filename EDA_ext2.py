#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import pearsonr
from dash import Input, Output
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from selenium import webdriver
import time


# In[2]:


df = pd.read_csv('Extruder_two_One_year_data.csv')
df


# In[3]:


df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
df = df.sort_values(by='Time')


# In[4]:


df.set_index('Time', inplace = True)


# In[5]:


df.columns


# In[6]:


df.drop(['P02E100M01_IO_CV','P02E100SC01_IO_Setpoint','P02E121Y01_IO_Setpoint',
       'P02E150TT01_IO_PV',
       'P02E161PT01_IO_PV', 'P02E161PT02_IO_PV', 'P02E161TT01_IO_PV',
       'P02E161TT02_IO_PV', 'P02E161TT03_IO_PV', 'P02E161TT04_IO_PV',
       'P02E181XC01_IO_Actual', 'P02E181XC01_IO_Setpoint',
       'P02E182XC01_IO_Actual', 'P02E182XC01_IO_Setpoint',
       'P02E183XC01_IO_Actual', 'P02E183XC01_IO_Setpoint',
       'P02E184XC01_IO_Actual', 'P02E184XC01_IO_Setpoint',
       'P02E185XC01_IO_Actual', 'P02E185XC01_IO_Setpoint',
       'P02E186XC01_IO_Actual', 'P02E186XC01_IO_Setpoint',
       ], inplace = True, axis = 1)


# In[7]:


df.describe().T


# In[8]:


app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    dcc.Graph(id='scatter-plot-origin-df'),
    dcc.Dropdown(
        id='column-selector-origin-df',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=[df.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app.callback(
    Output('scatter-plot-origin-df', 'figure'),
    [Input('column-selector-origin-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = df[col].dropna()
        traces.append(go.Scatter(x=filtered_df.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True, port=8010)


# In[9]:


df.columns


# In[10]:


# Assuming you have a DataFrame `df`
# df = pd.read_csv('your_data.csv') # Load your dataset

# Initialize the Dash app
app10 = dash.Dash(__name__)

# List of variables you want to plot, for example:
variables = ['P02E100M01_IO_PV', 'P02E121EV01_IO_Actual', 'P02E121Y01_IO_Actual', 'P02E400TT01_IO_PV']

app10.layout = html.Div([
    html.H1('Statistical Summary of Variables'),
    dcc.Dropdown(
        id='variable-selector',
        options=[{'label': var, 'value': var} for var in variables],
        value=variables[0]  # Default value
    ),
    dcc.Graph(id='box-plot'),
    dcc.Graph(id='histogram-plot'),
])

@app10.callback(
    [Output('box-plot', 'figure'),
     Output('histogram-plot', 'figure')],
    [Input('variable-selector', 'value')]
)
def update_plots(selected_variable):
    # Box Plot
    box_fig = go.Figure()
    box_fig.add_trace(go.Box(y=df[selected_variable], name=selected_variable))
    box_fig.update_layout(title_text=f'Box Plot of {selected_variable}')
    
    # Histogram
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=df[selected_variable], name=selected_variable))
    hist_fig.update_layout(title_text=f'Histogram of {selected_variable}')
    
    return box_fig, hist_fig
if __name__ == '__main__':
    app10.run_server(debug=True, port = 7088)


# ##  Limiting data to Dec to Feb

# In[11]:


df_recent = pd.read_csv('Ext2_Dec_to_Feb.csv')
df_recent


# In[12]:


df_recent['Time'] = pd.to_datetime(df_recent['Time'], format='%d/%m/%Y %H:%M:%S.%f')
df_recent = df_recent.sort_values(by='Time')


# In[13]:


df_recent.set_index('Time', inplace = True)


# In[14]:


df_recent.columns


# In[15]:


df_recent.describe().T


# In[16]:


df_recent.drop(['col_04_Extrusion_IO_Extruder2_Consumed',
       'col_04_Extrusion_IO_Extruder2_Produced',
       'col_04_Extrusion_IO_Extruder2_Waste', 'Extruder2_IO_BadgeID',
       'Extruder2_IO_PackMLState', 'Extruder2_IO_StepNumber',
       'P02E100M01_IO_CV', 'P02E150PT01_AlarmMostUrgentSeverity',
       'P02E161PT01_IO_PV', 'P02E161PT02_IO_PV', 'P02E161TT01_IO_PV',
       'P02E161TT02_IO_PV', 'P02E161TT03_IO_PV', 'P02E161TT04_IO_PV',
       'P02E185XC01_IO_Actual', 'P02E185XC01_IO_Setpoint',
       'P02E186XC01_IO_Actual', 'P02E186XC01_IO_Setpoint',
       'P02E400TT01_AlarmMostUrgentSeverity'] ,axis =1 , inplace = True)


# In[17]:


df_recent.describe().T


# In[18]:


app1 = dash.Dash(__name__)

app1.layout = html.Div([
    dcc.Graph(id='scatter-plot-recent-df'),
    dcc.Dropdown(
        id='column-selector-recent-df',
        options=[{'label': col, 'value': col} for col in df_recent.columns],
        value=[df_recent.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app1.callback(
    Output('scatter-plot-recent-df', 'figure'),
    [Input('column-selector-recent-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = df_recent[col].dropna()
        traces.append(go.Scatter(x=filtered_df.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app1.run_server(debug=True, port=8012)


# ### Setpoint value tracking analysis

# In[19]:


df_both_actual_setpoint = df_recent.dropna(subset=['P02E181XC01_IO_Actual', 'P02E181XC01_IO_Setpoint'])


# In[20]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[21]:


mae = mean_absolute_error(df_both_actual_setpoint ['P02E181XC01_IO_Actual'], df_both_actual_setpoint ['P02E181XC01_IO_Setpoint'])
rmse = np.sqrt(mean_squared_error(df_both_actual_setpoint ['P02E181XC01_IO_Actual'], df_both_actual_setpoint ['P02E181XC01_IO_Setpoint']))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")


# In[22]:


correlation = df_both_actual_setpoint['P02E181XC01_IO_Actual'].corr(df_both_actual_setpoint['P02E181XC01_IO_Setpoint'])
print(f"Correlation coefficient: {correlation}")


# In[23]:


def analyze_multiple_temperature_relationships(df, actual_cols, setpoint_cols):
    """
    Analyzes the relationship between multiple pairs of actual and setpoint temperatures.

    Parameters:
    - df: DataFrame containing the temperature data.
    - actual_cols: List of strings, each name of a column containing actual temperature data.
    - setpoint_cols: List of strings, each name of a column containing setpoint temperature data.

    Returns:
    - A dictionary of dictionaries, each containing MAE, RMSE, and correlation coefficient for each pair of provided columns.
    """
    
    results = {}
    
    for actual_col, setpoint_col in zip(actual_cols, setpoint_cols):
        df_filtered = df.dropna(subset=[actual_col, setpoint_col])
        mae = mean_absolute_error(df_filtered[actual_col], df_filtered[setpoint_col])
        rmse = np.sqrt(mean_squared_error(df_filtered[actual_col], df_filtered[setpoint_col]))
        correlation = df_filtered[actual_col].corr(df_filtered[setpoint_col])
        
        results[f"{actual_col} vs {setpoint_col}"] = {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation Coefficient': correlation
        }
    
    return results

# Example usage:
# Assuming you have a DataFrame named 'df' and lists of actual and setpoint columns
# actual_cols = ['ActualTemp1', 'ActualTemp2']
# setpoint_cols = ['SetpointTemp1', 'SetpointTemp2']
# results = analyze_multiple_temperature_relationships(df, actual_cols, setpoint_cols)
# for pair, metrics in results.items():
#     print(f"Results for {pair}:")
#     for metric, value in metrics.items():
#         print(f"  {metric}: {value}")



# In[24]:


# Assuming you have a DataFrame named 'df' and lists of actual and setpoint columns
actual_cols = ['P02E181XC01_IO_Actual', 'P02E182XC01_IO_Actual', 'P02E183XC01_IO_Actual', 'P02E184XC01_IO_Actual', 'P02E100SC01_IO_Actual', 'P02E121EV01_IO_Actual','P02E121Y01_IO_Actual']
setpoint_cols = ['P02E181XC01_IO_Setpoint', 'P02E182XC01_IO_Setpoint', 'P02E183XC01_IO_Setpoint','P02E184XC01_IO_Setpoint', 'P02E100SC01_IO_Setpoint','P02E121EV01_IO_Setpoint','P02E121Y01_IO_Setpoint']
results = analyze_multiple_temperature_relationships(df_recent, actual_cols, setpoint_cols)
for pair, metrics in results.items():
    print(f"Results for {pair}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


# ## SME

# In[25]:


sme = df_recent[df_recent['P02E100SME01_IO_PV'].notnull()]['P02E100SME01_IO_PV']


# In[26]:


plt.boxplot(sme)


# In[27]:


plt.hist(sme, bins = 20)


# In[28]:


# Filter outliers and select values smaller than 100
sme_without_outlier = sme[sme< 100]


# In[29]:


sme_df = sme_without_outlier.resample('1H').mean()


# In[30]:


plt.boxplot(sme_without_outlier)


# In[31]:


plt.hist(sme_without_outlier, bins = 5)


# In[32]:


len(sme_without_outlier[sme_without_outlier<40])


# In[33]:


sme_df.describe()


# In[34]:


# Initialize the Dash app
app82 = dash.Dash(__name__)

app82.layout = html.Div([
    html.H1('SME Scatter Plot'),
    dcc.Graph(id='sme-scatter-plot'),
    # Additional components like dcc.Dropdown can be added here for interactive plots
])

@app82.callback(
    Output('sme-scatter-plot', 'figure'),
    [Input('sme-scatter-plot', 'id')]  # This input is just to trigger the callback
)
def update_temperature_scatter_plot(_):
    # Creating a scatter plot of the temperature data
    fig = px.scatter(sme_df, x= sme_df.index, y='P02E100SME01_IO_PV', title='SME Scatter Plot')
    fig.update_layout(xaxis_title='Time', yaxis_title='SME(kJ/kg)', hovermode='closest')
    return fig

if __name__ == '__main__':
    app82.run_server(debug=True, port =  8212)


# ## Screw Speed analysis

# In[35]:


screw_speed = df_recent[df_recent['P02E100M01_IO_PV'].notnull()]['P02E100M01_IO_PV']
screw_speed


# In[36]:


# Initialize the Dash app
app01 = dash.Dash(__name__)

app01.layout = html.Div([
    html.H1('Screw Speed Scatter Plot'),
    dcc.Graph(id='screw speed-scatter-plot'),
    # Additional components like dcc.Dropdown can be added here for interactive plots
])

@app01.callback(
    Output('screw speed-scatter-plot', 'figure'),
    [Input('screw speed-scatter-plot', 'id')]  # This input is just to trigger the callback
)
def update_temperature_scatter_plot(_):
    # Creating a scatter plot of the temperature data
    fig = px.scatter(screw_speed, x=screw_speed.index, y='P02E100M01_IO_PV', title='Screw Speed Scatter Plot')
    fig.update_layout(xaxis_title='Time', yaxis_title='Screw speed (rmp)', hovermode='closest')
    return fig

if __name__ == '__main__':
    app01.run_server(debug=True , port = 6010)


# In[ ]:





# In[37]:


screw_speed_df = screw_speed.resample('1H').mean()
screw_speed_df


# ## Regulator speed

# In[38]:


regulator_speed = df_recent[df_recent['P02E100SC01_IO_Actual'].notnull()]['P02E100SC01_IO_Actual']


# In[39]:


regulator_speed_df = regulator_speed.resample('1H').mean()


# ## Moisture injection

# In[40]:


moisture_injection= df_recent[df_recent['P02E121EV01_IO_Actual'].notnull()]['P02E121EV01_IO_Actual']


# In[41]:


moisture_injection_df = moisture_injection.resample('1H').mean()


# ## Soy dosing

# In[42]:


soy_dosing = df_recent[df_recent['P02E121Y01_IO_Actual'].notnull()]['P02E121Y01_IO_Actual']


# In[43]:


soy_dosing_df = soy_dosing.resample('1H').mean()


# ## Barrels temperature

# In[44]:


temp_barrel_one = df_recent[df_recent['P02E400TT01_IO_PV'].notnull()]['P02E400TT01_IO_PV']


# In[45]:


# Initialize the Dash app
app02 = dash.Dash(__name__)

app02.layout = html.Div([
    html.H1('Temperature Scatter Plot'),
    dcc.Graph(id='temperature-scatter-plot'),
    # Additional components like dcc.Dropdown can be added here for interactive plots
])

@app02.callback(
    Output('temperature-scatter-plot', 'figure'),
    [Input('temperature-scatter-plot', 'id')]  # This input is just to trigger the callback
)
def update_temperature_scatter_plot(_):
    # Creating a scatter plot of the temperature data
    fig = px.scatter(temp_barrel_one, x= temp_barrel_one.index, y='P02E400TT01_IO_PV', title='Temperature Scatter Plot')
    fig.update_layout(xaxis_title='Time', yaxis_title='Temperature (°C)', hovermode='closest')
    return fig

if __name__ == '__main__':
    app02.run_server(debug=True, port =  6012)


# In[46]:


temp_barrel_one_df = temp_barrel_one.resample('1H').mean()


# In[47]:


temp_barrel_one_df.interpolate(method = 'linear', inplace = True)
temp_barrel_one_df


# In[48]:


# Initialize the Dash app
app03 = dash.Dash(__name__)

app03.layout = html.Div([
    html.H1('Interpolated Temperature Scatter Plot'),
    dcc.Graph(id='interpolated-temperature-scatter-plot'),
    # Additional components like dcc.Dropdown can be added here for interactive plots
])

@app03.callback(
    Output('interpolated-temperature-scatter-plot', 'figure'),
    [Input('interpolated-temperature-scatter-plot', 'id')]  # This input is just to trigger the callback
)
def update_temperature_scatter_plot(_):
    # Creating a scatter plot of the temperature data
    fig = px.line(temp_barrel_one_df, x= temp_barrel_one_df.index, y='P02E400TT01_IO_PV', title='Interpolated Temperature Scatter Plot')
    fig.update_layout(xaxis_title='Time', yaxis_title='Temperature (°C)', hovermode='closest')
    return fig

if __name__ == '__main__':
    app03.run_server(debug=True, port =  6014)


# #### second

# In[49]:


temp_barrel_two = df_recent[df_recent['P02E400TT02_IO_PV'].notnull()]['P02E400TT02_IO_PV']


# In[50]:


temp_barrel_two_df = temp_barrel_two.resample('1H').mean()
temp_barrel_two_df.interpolate(method = 'linear' ,inplace = True)
temp_barrel_two_df


# #### Third

# In[51]:


temp_barrel_three = df_recent[df_recent['P02E400TT03_IO_PV'].notnull()]['P02E400TT03_IO_PV']


# In[52]:


temp_barrel_three_df = temp_barrel_three.resample('1H').mean()
temp_barrel_three_df.interpolate(method = 'linear', inplace = True)
temp_barrel_three_df


# #### Fourth

# In[53]:


temp_barrel_four = df_recent[df_recent['P02E400TT04_IO_PV'].notnull()]['P02E400TT04_IO_PV']


# In[54]:


temp_barrel_four_df = temp_barrel_four.resample('1H').mean()
temp_barrel_four_df.interpolate(method = 'linear' ,inplace = True)
temp_barrel_four_df


# #### Fifth

# In[55]:


temp_barrel_five = df_recent[df_recent['P02E400TT05_IO_PV'].notnull()]['P02E400TT05_IO_PV']


# In[56]:


temp_barrel_five_df = temp_barrel_five.resample('1H').mean()
temp_barrel_five_df.interpolate(method = 'linear' ,inplace = True)
temp_barrel_five_df


# #### 6th

# In[57]:


temp_barrel_six = df_recent[df_recent['P02E400TT06_IO_PV'].notnull()]['P02E400TT06_IO_PV']


# In[58]:


temp_barrel_six_df = temp_barrel_six.resample('1H').mean()
temp_barrel_six_df.interpolate(method = 'linear' ,inplace = True)
temp_barrel_six_df


# #### 7th

# In[59]:


temp_barrel_seven = df_recent[df_recent['P02E400TT07_IO_PV'].notnull()]['P02E400TT07_IO_PV']


# In[60]:


temp_barrel_seven_df = temp_barrel_seven.resample('1H').mean()
temp_barrel_seven_df.interpolate(method = 'linear', inplace = True)
temp_barrel_seven_df


# #### 8th

# In[61]:


temp_barrel_eight = df_recent[df_recent['P02E400TT08_IO_PV'].notnull()]['P02E400TT08_IO_PV']


# In[62]:


temp_barrel_eight_df = temp_barrel_eight.resample('1H').mean()
temp_barrel_eight_df.interpolate(method = 'linear', inplace = True)
temp_barrel_eight_df


# #### 9th

# In[63]:


temp_barrel_nine = df_recent[df_recent['P02E400TT09_IO_PV'].notnull()]['P02E400TT09_IO_PV']


# In[64]:


temp_barrel_nine_df = temp_barrel_nine.resample('1H').mean()
temp_barrel_nine_df.interpolate(method = 'linear', inplace = True)
temp_barrel_nine_df


# ## Actual  values of the Heating system

# ### First heating system

# In[65]:


temp_HS_one = df_recent[df_recent['P02E181XC01_IO_Actual'].notnull()]['P02E181XC01_IO_Actual']


# In[66]:


temp_HS_one_df = temp_HS_one.resample('1H').mean()
temp_HS_one_df.interpolate(method = 'linear', inplace = True)
temp_HS_one_df


# ### Second heating system

# In[67]:


temp_HS_two = df_recent[df_recent['P02E182XC01_IO_Actual'].notnull()]['P02E182XC01_IO_Actual']


# In[68]:


temp_HS_two_df = temp_HS_two.resample('1H').mean()
temp_HS_two_df.interpolate(method = 'linear', inplace = True)
temp_HS_two_df


# ### Third heating system

# In[69]:


temp_HS_three = df_recent[df_recent['P02E183XC01_IO_Actual'].notnull()]['P02E183XC01_IO_Actual']


# In[70]:


temp_HS_three_df = temp_HS_three.resample('1H').mean()
temp_HS_three_df.interpolate(method = 'linear', inplace = True)
temp_HS_three_df


# In[ ]:





# ### Fourth heating system

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Merging 

# In[71]:


interpolated_df = pd.concat([screw_speed_df, regulator_speed_df, moisture_injection_df, sme_df, soy_dosing_df, temp_barrel_one_df,temp_barrel_two_df, temp_barrel_three_df, temp_barrel_four_df
                            ,temp_barrel_five_df, temp_barrel_six_df, temp_barrel_seven_df, temp_barrel_eight_df, temp_barrel_nine_df], axis=1)


# In[72]:


interpolated_df


# In[73]:


interpolated_df.describe().T


# In[74]:


app12 = dash.Dash(__name__)

app12.layout = html.Div([
    dcc.Graph(id='scatter-plot-interpolated-df'),
    dcc.Dropdown(
        id='column-selector-interpolated-df',
        options=[{'label': col, 'value': col} for col in interpolated_df.columns],
        value=[interpolated_df.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app12.callback(
    Output('scatter-plot-interpolated-df', 'figure'),
    [Input('column-selector-interpolated-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = interpolated_df[col].dropna()
        traces.append(go.Scatter(x= interpolated_df.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot of interpolated',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app12.run_server(debug=True, port=8060)


# In[75]:


### this is the date when we start to collect soy dosing value
filtered_interpolated_df = interpolated_df[interpolated_df.index >= '2023-12-12']
filtered_interpolated_df.describe().T


# In[76]:


# Calculate the difference between consecutive temperature readings
filtered_interpolated_df['temp_diff'] = filtered_interpolated_df['P02E400TT01_IO_PV'].diff()
filtered_interpolated_df[pd.isnull(filtered_interpolated_df['temp_diff'])]


# In[77]:


# Initialize variables to store the last high value for both columns
last_high_motor_power = 0

# Variable to track the last power value for continuity when slope is near zero
last_power_value = None

# Define a tolerance for considering the slope as zero
slope_tolerance = 2

# Create a copy of the DataFrame to work on
df_modified = filtered_interpolated_df.copy()

# Iterate through each row
for index, row in df_modified.iterrows():
    # Check for NaN in 'P02E100M01_IO_PV' to decide on interpolation
    if pd.isnull(row['P02E100M01_IO_PV']):
        
        # If temperature is below 80 and slope is positive, set to 740
        if row['P02E400TT01_IO_PV'] < 80 and row['temp_diff'] > 2:
            df_modified.at[index, 'P02E100M01_IO_PV'] = 740
            last_power_value = 740  # Update last power value
        
        # If temperature is below 80 and slope is negative, set to 0
        elif row['P02E400TT01_IO_PV'] < 80 and row['temp_diff'] < -2:
            df_modified.at[index, 'P02E100M01_IO_PV'] = 0
            last_power_value = 0  # Update last power value
    
    
        elif row['P02E400TT01_IO_PV'] >= 80 or ( -slope_tolerance <= row['temp_diff'] <= slope_tolerance):
            df_modified.at[index, 'P02E100M01_IO_PV'] = last_power_value   
        
        # If slope is within the tolerance of zero, continue with the last power value
        #elif ( -slope_tolerance <= row['temp_diff'] <= slope_tolerance) and last_power_value is not None:
            #df_modified.at[index, 'P02E100M01_IO_PV'] = last_power_value
            
        
       
            
    # If 'P02E100M01_IO_PV' is not NaN, update last high motor power and last power value
    else:
        last_high_motor_power = max(last_high_motor_power, row['P02E100M01_IO_PV'])
        last_power_value = row['P02E100M01_IO_PV']  # Update last power value to the current one

# The modified DataFrame is ready for use
df_modified


# In[78]:


df_modified.drop(index = '2023-12-12 00:00:00', inplace = True)


# In[79]:


nan_rows = df_modified[pd.isnull(df_modified['P02E100M01_IO_PV'])]
nan_rows


# In[80]:


df_modified.describe().T


# In[81]:


# Drop the row with the specified index
#interpolated_df.drop(pd.Timestamp('2024-01-05 16:00:00'), inplace=True)


# In[ ]:





# In[82]:


app2 = dash.Dash(__name__)

app2.layout = html.Div([
    dcc.Graph(id='scatter-plot-modified-df'),
    dcc.Dropdown(
        id='column-selector-modified-df',
        options=[{'label': col, 'value': col} for col in df_modified.columns],
        value=[df_modified.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app2.callback(
    Output('scatter-plot-modified-df', 'figure'),
    [Input('column-selector-modified-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = df_modified[col].dropna()
        traces.append(go.Scatter(x=df_modified.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot of modified',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app2.run_server(debug=True, port=8016)


# In[83]:


df_modified.columns


# ## SME interpolation

# In[84]:


# Initialize variables to store the last high value for both columns
last_high_value_sme = 0

# Create a copy of the DataFrame to work on
df_modified_sme = df_modified.copy()

# Iterate through each row
for index, row in df_modified_sme.iterrows():
    # Interpolation for 'soy dosing'
    if pd.isnull(row['P02E100SME01_IO_PV']):
        
        if row['P02E100M01_IO_PV'] == 0 :
            df_modified_sme.at[index, 'P02E100SME01_IO_PV'] = 0
            
        elif row['P02E100M01_IO_PV'] != 0 :
            df_modified_sme.at[index, 'P02E100SME01_IO_PV'] = last_high_value_sme
    else:
        last_high_value_sme = max(last_high_value_sme, row['P02E100SME01_IO_PV'])
        
# Return the updated DataFrame
df_modified_sme


# In[85]:


df_modified_sme['P02E100SME01_IO_PV']


# In[86]:


df_modified_sme.describe().T


# ## Soy dosing & water injection interpolation

# In[87]:


# Interpolation function
def interpolate_values(df):
    for index, row in df.iterrows():
        if pd.isna(row['P02E121Y01_IO_Actual']) and pd.isna(row['P02E121EV01_IO_Actual']):
            if row['P02E400TT01_IO_PV'] >= 60:
                if index.month == 12:
                    df.at[index, 'P02E121Y01_IO_Actual'] = 0.4 * 500
                    df.at[index, 'P02E121EV01_IO_Actual'] = 0.6 * 500
                else:
                    df.at[index, 'P02E121Y01_IO_Actual'] = 0.4 * 450
                    df.at[index, 'P02E121EV01_IO_Actual'] = 0.6 * 450
            else:
                df.at[index, 'P02E121Y01_IO_Actual'] = 0
                df.at[index, 'P02E121EV01_IO_Actual'] = 0
        elif pd.isna(row['P02E121Y01_IO_Actual']) and pd.notna(row['P02E121EV01_IO_Actual']):
            if row['P02E121EV01_IO_Actual'] == 0:
                df.at[index, 'P02E121Y01_IO_Actual'] = 0
            else:
                df.at[index, 'P02E121Y01_IO_Actual'] = 2/3 * row['P02E121EV01_IO_Actual']
        elif pd.notna(row['P02E121Y01_IO_Actual']) and pd.isna(row['P02E121EV01_IO_Actual']):
            if row['P02E121Y01_IO_Actual'] == 0:
                df.at[index, 'P02E121EV01_IO_Actual'] = 0
            else:
                df.at[index, 'P02E121EV01_IO_Actual'] = 1.5 * row['P02E121Y01_IO_Actual']
    return df


# In[88]:


# Apply interpolation
df_interpolated = interpolate_values(df_modified_sme.copy())
df_interpolated


# In[89]:


# Initialize variables to store the last high value for both columns
#last_high_value_soy = 0
#last_high_value_water = 240

# Create a copy of the DataFrame to work on
#df_interpolated = df_modified_sme.copy()

# Iterate through each row
#for index, row in df_interpolated.iterrows():
    # Interpolation for 'soy dosing'
    #if pd.isnull(row['P02E121Y01_IO_Actual']):
        
        #if row['P02E400TT01_IO_PV'] > 60 :df_interpolated.at[index, 'P02E121Y01_IO_Actual'] = last_high_value_soy
            
        #elif row['P02E400TT01_IO_PV'] <= 60 :df_interpolated.at[index, 'P02E121Y01_IO_Actual'] = 0
   # else:
       # last_high_value_soy = max(last_high_value_soy, row['P02E121Y01_IO_Actual'])

    # Interpolation for 'water injection' if pd.isnull(row['P02E121EV01_IO_Actual']):if row['P02E400TT01_IO_PV'] > 60 :
           # df_interpolated.at[index, 'P02E121EV01_IO_Actual'] = last_high_value_water
            
      #  elif row['P02E400TT01_IO_PV'] <= 60 :
          #  df_interpolated.at[index, 'P02E121EV01_IO_Actual'] = 0
  #  else:
       # last_high_value_water = max(last_high_value_water, row['P02E121EV01_IO_Actual'])

# Return the updated DataFrame
#df_interpolated


# In[90]:


df_interpolated.describe().T


# In[91]:


# Calculate total of moisture and soy for each row
df_interpolated['total'] = df_interpolated['P02E121EV01_IO_Actual'] + df_interpolated['P02E121Y01_IO_Actual']

# Calculate percentage of moisture and soy
print((df_interpolated['P02E121EV01_IO_Actual'] / df_interpolated['total']) * 100) 
print((df_interpolated['P02E121Y01_IO_Actual'] / df_interpolated['total']) * 100)


# In[92]:


df_soy_water_percent = df.dropna(subset=['P02E121Y01_IO_Actual', 'P02E121EV01_IO_Actual'])


# In[ ]:





# In[93]:


plt.hist(df_interpolated['total'], bins = 10)


# In[94]:


app3 = dash.Dash(__name__)

app3.layout = html.Div([
    dcc.Graph(id='scatter-plot-interpolated-df'),
    dcc.Dropdown(
        id='column-selector-interpolated-df',
        options=[{'label': col, 'value': col} for col in df_interpolated.columns],
        value=[df_interpolated.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app3.callback(
    Output('scatter-plot-interpolated-df', 'figure'),
    [Input('column-selector-interpolated-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = df_interpolated[col].dropna()
        traces.append(go.Scatter(x=filtered_df.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app3.run_server(debug=True, port=8018)


# ## Operating status

# In[95]:


#operating_df = df_interpolated[(df_interpolated['P02E100M01_IO_PV'] >= 680)]
operating_df = df_interpolated[(df_interpolated['P02E100M01_IO_PV'] >= 680) & (df_interpolated['P02E400TT01_IO_PV'] > 80)]


# In[96]:


operating_df.describe().T


# In[97]:


plt.hist(operating_df['total'], bins = 10)


# In[98]:


operating_df[operating_df['P02E400TT02_IO_PV'] < 75]


# In[99]:


app4 = dash.Dash(__name__)

app4.layout = html.Div([
    dcc.Graph(id='scatter-plot-operated-df'),
    dcc.Dropdown(
        id='column-selector-operated-df',
        options=[{'label': col, 'value': col} for col in operating_df.columns],
        value=[operating_df.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app4.callback(
    Output('scatter-plot-operated-df', 'figure'),
    [Input('column-selector-operated-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = operating_df[col].dropna()
        traces.append(go.Scatter(x=filtered_df.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app4.run_server(debug=True, port=8024)


# In[100]:


app4 = dash.Dash(__name__)

app4.layout = html.Div([
    dcc.Graph(id='scatter-plot-operated-df'),
    dcc.Dropdown(
        id='column-selector-operated-df',
        options=[{'label': col, 'value': col} for col in operating_df.columns],
        value=[operating_df.columns[0]],  # Default value as the first column
        multi=True  # Allow multiple selections
    )
])

@app4.callback(
    Output('scatter-plot-operated-df', 'figure'),
    [Input('column-selector-operated-df', 'value')]
)
def update_graph(selected_columns):
    traces = []
    for col in selected_columns:
        filtered_df = operating_df[col].dropna()
        traces.append(go.Scatter(x=filtered_df.index, y=filtered_df, mode='markers', name=col))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Scatter Plot',
            xaxis_title='Time',
            yaxis_title='Value'
        )
    }

if __name__ == '__main__':
    app4.run_server(debug=True, port=8024)


# In[101]:


operating_df.drop('temp_diff', axis = 1, inplace= True)


# In[102]:


operating_df.drop('total', axis =1, inplace = True)


# In[103]:


# Calculate the Pearson correlation matrix
corr_matrix_operated = operating_df.corr()

# Initialize Dash app
app6 = dash.Dash(__name__)

# App layout
app6.layout = html.Div([
    html.H1("Pearson Correlation Heatmap_OPERATED"),
    dcc.Graph(id='correlation-heatmap_OPERATED', figure=px.imshow(
        corr_matrix_operated, 
        x=corr_matrix_operated.columns, 
        y=corr_matrix_operated.columns,
        labels=dict(color="Pearson Correlation"),
        color_continuous_scale='RdBu_r'  # Red to Blue color scale, reversed
    ))
])

# Run the app
if __name__ == '__main__':
    app6.run_server(debug=True, port = 8070)


# In[104]:


df_interpolated.drop('temp_diff', axis=1, inplace = True)


# In[105]:


# Calculate the Pearson correlation matrix
corr_matrix = df_interpolated.corr()

# Initialize Dash app
app7 = dash.Dash(__name__)

# App layout
app7.layout = html.Div([
    html.H1("Pearson Correlation Heatmap_Updated_data"),
    dcc.Graph(id='correlation-heatmap_Updated_data', figure=px.imshow(
        corr_matrix, 
        x=corr_matrix.columns, 
        y=corr_matrix.columns,
        labels=dict(color="Pearson Correlation"),
        color_continuous_scale='RdBu_r'  # Red to Blue color scale, reversed
    ))
])

# Run the app
if __name__ == '__main__':
    app7.run_server(debug=True, port = 8072)


# In[106]:


corr_matrix


# In[107]:


# range of the important variables
# SME 
# how to add protein based information
## Challenges:
#not having protein percentage 
# there is no specific relationship between sme and other parameters, why???
# 


# In[108]:


# Initialize the Dash app
app9 = dash.Dash(__name__)

# List of variables you want to plot, for example:
variables = ['P02E100M01_IO_PV', 'P02E121EV01_IO_Actual', 'P02E121Y01_IO_Actual', 'P02E400TT01_IO_PV']

app9.layout = html.Div([
    html.H1('Statistical Summary of Variables'),
    dcc.Dropdown(
        id='variable-selector',
        options=[{'label': var, 'value': var} for var in variables],
        value=variables[0]  # Default value
    ),
    dcc.Graph(id='box-plot'),
    dcc.Graph(id='histogram-plot'),
])

@app9.callback(
    [Output('box-plot', 'figure'),
     Output('histogram-plot', 'figure')],
    [Input('variable-selector', 'value')]
)
def update_plots(selected_variable):
    # Box Plot
    box_fig = go.Figure()
    box_fig.add_trace(go.Box(y=operating_df[selected_variable], name=selected_variable))
    box_fig.update_layout(title_text=f'Box Plot of {selected_variable}')
    
    # Histogram
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=operating_df[selected_variable], name=selected_variable))
    hist_fig.update_layout(title_text=f'Histogram of {selected_variable}')
    
    return box_fig, hist_fig

if __name__ == '__main__':
    app9.run_server(debug=True, port = 8008)


# In[109]:


# Initialize the Dash app
app10 = dash.Dash(__name__)

# List of variables you want to plot, for example:
variables = ['P02E100M01_IO_PV', 'P02E121EV01_IO_Actual', 'P02E121Y01_IO_Actual', 'P02E400TT01_IO_PV']

app10.layout = html.Div([
    html.H1('Statistical Summary of Variables of the original data'),
    dcc.Dropdown(
        id='variable-selector',
        options=[{'label': var, 'value': var} for var in variables],
        value=variables[0]  # Default value
    ),
    dcc.Graph(id='box-plot'),
    dcc.Graph(id='histogram-plot'),
])

@app10.callback(
    [Output('box-plot', 'figure'),
     Output('histogram-plot', 'figure')],
    [Input('variable-selector', 'value')]
)
def update_plots(selected_variable):
    # Box Plot
    box_fig = go.Figure()
    box_fig.add_trace(go.Box(y=df[selected_variable], name=selected_variable))
    box_fig.update_layout(title_text=f'Box Plot of {selected_variable}')
    
    # Histogram
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=df[selected_variable], name=selected_variable))
    hist_fig.update_layout(title_text=f'Histogram of {selected_variable}')
    
    return box_fig, hist_fig

if __name__ == '__main__':
    app10.run_server(debug=True, port = 8032)


# In[ ]:




