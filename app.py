import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title='Industrial Engineering Dashboard',
    layout='wide'
)

# TITLE
st.title('Industrial Engineering Admin Dashboard')

st.subheader(
    'Manufacturing Production Intelligence System'
)

# LOAD DATA
df = pd.read_csv('hybrid_manufacturing_categorical.csv')

# LOAD MODEL FILES
model_package = joblib.load(
    'models/full_model.pkl'
)

model = model_package['model']

scaler = model_package['scaler']

features = model_package['features']

# KPI SECTION
avg_energy = df['Energy_Consumption'].mean()

avg_processing = df['Processing_Time'].mean()

avg_availability = df['Machine_Availability'].mean()

total_jobs = len(df)

# KPI CARDS
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    'Average Energy',
    f'{avg_energy:.2f}'
)

col2.metric(
    'Avg Processing Time',
    f'{avg_processing:.2f}'
)

col3.metric(
    'Machine Availability',
    f'{avg_availability:.2f}%'
)

col4.metric(
    'Total Jobs',
    total_jobs
)

# ENERGY TREND
fig1 = px.line(
    df,
    y='Energy_Consumption',
    title='Energy Consumption Trend'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# SCATTER PLOT
fig2 = px.scatter(
    df,
    x='Processing_Time',
    y='Energy_Consumption',
    color='Job_Status',
    title='Processing Time vs Energy Consumption'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# OPERATION DISTRIBUTION
fig3 = px.histogram(
    df,
    x='Operation_Type',
    title='Operation Type Distribution'
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# PREDICTION SECTION
st.header('Energy Consumption Prediction')

material_used = st.number_input(
    'Material Used',
    min_value=0.0
)

processing_time = st.number_input(
    'Processing Time',
    min_value=0.0
)

machine_availability = st.number_input(
    'Machine Availability',
    min_value=0.0,
    max_value=100.0
)

if st.button('Predict Energy Consumption'):

    input_df = pd.DataFrame({
        'Material_Used': [material_used],
        'Processing_Time': [processing_time],
        'Machine_Availability': [machine_availability]
    })

    # ADD MISSING COLUMNS
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # REORDER COLUMNS
    input_df = input_df[features]

    # SCALE INPUT
    scaled_input = scaler.transform(input_df)

    # PREDICT
    prediction = model.predict(scaled_input)

    st.success(
        f'Predicted Energy Consumption: {prediction[0]:.2f}'
    )


    # PART 8

# ENCODE
df_encoded = pd.get_dummies(
    df,
    columns=[
        'Job_Status',
        'Optimization_Category',
        'Machine_ID',
        'Operation_Type'
    ],
    drop_first=True
)

# DROP UNUSED COLUMNS
df_encoded = df_encoded.drop([
    'Job_ID',
    'Scheduled_Start',
    'Scheduled_End',
    'Actual_Start',
    'Actual_End'
], axis=1)

# FEATURES
X_dashboard = df_encoded.drop(
    'Energy_Consumption',
    axis=1
)

# MATCH FEATURE ORDER
X_dashboard = X_dashboard[features]

# SCALE
X_scaled_dashboard = scaler.transform(
    X_dashboard
)

# PREDICT
preds_dashboard = model.predict(
    X_scaled_dashboard
)

# RESIDUALS
y_actual = df_encoded['Energy_Consumption']

residuals = y_actual - preds_dashboard

# PART 9

st.header('Feature Importance Analysis')

# CREATE COEFFICIENT DATAFRAME
coefficients = pd.DataFrame({

    'Feature': features,

    'Coefficient': model.coef_

})

# SORT COEFFICIENTS
coefficients = coefficients.sort_values(

    by='Coefficient',

    ascending=False

)

# DISPLAY TABLE
st.dataframe(coefficients)

# BAR CHART
fig_coeff = px.bar(

    coefficients,

    x='Coefficient',

    y='Feature',

    orientation='h',

    title='Feature Coefficient Impact'

)

st.plotly_chart(

    fig_coeff,

    use_container_width=True

)

# PART 10

st.header('Production Optimization Module')

from scipy.optimize import minimize

# OPTIMIZATION FUNCTION
def objective(x):

    # CREATE INPUT DATAFRAME
    sample = pd.DataFrame({

        'Material_Used': [x[0]],

        'Processing_Time': [x[1]],

        'Machine_Availability': [x[2]]

    })

    # ADD MISSING FEATURE COLUMNS
    for col in features:

        if col not in sample.columns:

            sample[col] = 0

    # MATCH FEATURE ORDER
    sample = sample[features]

    # SCALE INPUT
    scaled = scaler.transform(sample)

    # NEGATIVE VALUE FOR MAXIMIZATION
    return -model.predict(scaled)[0]

# VARIABLE BOUNDS
bounds = [

    (50, 150),     # Material Used

    (10, 200),     # Processing Time

    (70, 100)      # Machine Availability
]

# RUN OPTIMIZATION
result = minimize(

    objective,

    x0=[100, 50, 90],

    bounds=bounds,

    method='SLSQP'

)

# DISPLAY RESULTS
st.subheader('Optimal Manufacturing Settings')

st.write(

    f'Material Used: {result.x[0]:.2f}'

)

st.write(

    f'Processing Time: {result.x[1]:.2f}'

)

st.write(

    f'Machine Availability: {result.x[2]:.2f}%'

)

# DISPLAY OPTIMAL PREDICTION
optimal_sample = pd.DataFrame({

    'Material_Used': [result.x[0]],

    'Processing_Time': [result.x[1]],

    'Machine_Availability': [result.x[2]]

})

# ADD MISSING COLUMNS
for col in features:

    if col not in optimal_sample.columns:

        optimal_sample[col] = 0

# FEATURE ORDER
optimal_sample = optimal_sample[features]

# SCALE
optimal_scaled = scaler.transform(optimal_sample)

# PREDICT
optimal_prediction = model.predict(optimal_scaled)

st.success(

    f'Optimal Predicted Energy Consumption: {optimal_prediction[0]:.2f}'

)

# SPC CONTROL CHART

st.header('Residual SPC Control Chart')

fig_control = go.Figure()

# RESIDUAL LINE
fig_control.add_trace(

    go.Scatter(

        y=residuals,

        mode='lines+markers',

        name='Residuals'

    )

)

# SPC CONTROL CHART

st.header('Residual SPC Control Chart')

# CALCULATE CONTROL LIMITS
mean_res = residuals.mean()

std_res = residuals.std()

ucl = mean_res + (3 * std_res)

lcl = mean_res - (3 * std_res)

# CREATE FIGURE
fig_control = go.Figure()

# RESIDUAL LINE
fig_control.add_trace(

    go.Scatter(

        y=residuals,

        mode='lines+markers',

        name='Residuals'

    )

)

# UCL
fig_control.add_hline(

    y=ucl,

    line_dash='dash',

    line_color='red',

    annotation_text='UCL'

)

# LCL
fig_control.add_hline(

    y=lcl,

    line_dash='dash',

    line_color='red',

    annotation_text='LCL'

)

# MEAN
fig_control.add_hline(

    y=mean_res,

    line_dash='dash',

    line_color='green',

    annotation_text='Mean'

)

# DISPLAY
st.plotly_chart(

    fig_control,

    use_container_width=True

)