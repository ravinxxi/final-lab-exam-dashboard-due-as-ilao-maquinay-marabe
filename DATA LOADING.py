# libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

df = pd.read_csv('hybrid_manufacturing_categorical.csv')

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

df.dropna(inplace=True)

df['Scheduled_Start'] = pd.to_datetime(df['Scheduled_Start'])
df['Scheduled_End'] = pd.to_datetime(df['Scheduled_End'])
df['Actual_Start'] = pd.to_datetime(df['Actual_Start'])
df['Actual_End'] = pd.to_datetime(df['Actual_End'])

df['Production_Delay_Minutes'] = (
    df['Actual_End'] - df['Scheduled_End']
).dt.total_seconds() / 60

#figure 1. correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Manufacturing Correlation Matrix')
plt.show()

#figure 2. distribution plot
plt.figure(figsize=(8,5))
sns.histplot(df['Energy_Consumption'], bins=20, kde=True)
plt.title('Energy Consumption Distribution')
plt.show()

#figure 3. scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Processing_Time',
    y='Energy_Consumption',
    data=df
)
plt.title('Processing Time vs Energy Consumption')
plt.show()

# PART 2

df['energy_lag_1'] = df['Energy_Consumption'].shift(1)
df['energy_lag_2'] = df['Energy_Consumption'].shift(2)
df['energy_lag_3'] = df['Energy_Consumption'].shift(3)

df['rolling_mean_7'] = df['Energy_Consumption'].rolling(7).mean()

df.dropna(inplace=True)

df = pd.get_dummies(
    df,
    columns=['Machine_ID', 'Operation_Type'],
    drop_first=True
)

# PART 3
import pandas as pd
from sklearn.preprocessing import StandardScaler

# LOAD CSV
df = pd.read_csv('hybrid_manufacturing_categorical.csv')

# REMOVE NULLS
df.dropna(inplace=True)

# ENCODE CATEGORICAL COLUMNS (ONLY ONCE)
df = pd.get_dummies(
    df,
    columns=[
        'Job_Status',
        'Optimization_Category',
        'Machine_ID',
        'Operation_Type'
    ],
    drop_first=True
)

# REMOVE DATETIME/TEXT COLUMNS
df = df.drop([
    'Job_ID',
    'Scheduled_Start',
    'Scheduled_End',
    'Actual_Start',
    'Actual_End'
], axis=1)

# FEATURES/TARGET
X = df.drop('Energy_Consumption', axis=1)
y = df['Energy_Consumption']

# SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X.head())
print("SUCCESS")

# PART 4

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

# GENERATE PREDICTIONS
preds = model.predict(X_test)

# EVALUATION
r2 = r2_score(y_test, preds)

mae = mean_absolute_error(y_test, preds)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# PART 5

import matplotlib.pyplot as plt
import numpy as np

# RESIDUALS
residuals = y_test - preds

# MEAN AND STANDARD DEVIATION
mean_res = residuals.mean()
std_res = residuals.std()

# 3-SIGMA CONTROL LIMITS
ucl = mean_res + (3 * std_res)
lcl = mean_res - (3 * std_res)

# DETECT OUT-OF-CONTROL POINTS
outliers = np.where(
    (residuals > ucl) |
    (residuals < lcl)
)[0]

print("Out-of-Control Points:", len(outliers))

# SPC CONTROL CHART
plt.figure(figsize=(12,6))

plt.plot(
    residuals.values,
    marker='o',
    linestyle='-',
    label='Residuals'
)

# CENTER LINE
plt.axhline(
    mean_res,
    color='green',
    linestyle='--',
    label='Mean'
)

# UPPER CONTROL LIMIT
plt.axhline(
    ucl,
    color='red',
    linestyle='--',
    label='UCL'
)

# LOWER CONTROL LIMIT
plt.axhline(
    lcl,
    color='red',
    linestyle='--',
    label='LCL'
)

# HIGHLIGHT OUTLIERS
plt.scatter(
    outliers,
    residuals.iloc[outliers],
    color='red',
    s=100,
    label='Out-of-Control'
)

plt.title('Residual SPC Control Chart')

plt.xlabel('Observation')

plt.ylabel('Residual Error')

plt.legend()

plt.show()

# PART 6

import joblib

# SAVE MODEL
joblib.dump(
    model,
    'models/production_model.pkl'
)

# SAVE SCALER
joblib.dump(
    scaler,
    'models/scaler.pkl'
)

# SAVE FEATURES
joblib.dump(
    X.columns.tolist(),
    'models/features.pkl'
)

# SAVE COMPLETE PACKAGE
full_model = {

    'model': model,

    'scaler': scaler,

    'features': X.columns.tolist()

}

joblib.dump(
    full_model,
    'models/full_model.pkl'
)

print("All files saved successfully!")
