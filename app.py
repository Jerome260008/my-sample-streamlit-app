import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("C:/Users/JEROME P. ROMERO/Documents/hec_ras_simulation_results.csv")

print(df.isnull().sum())

X = df[['Rainfall (mm)', 'Flow Rate (cms)', 'River Water Level (m)']]
y = df['Flood Depth (m)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['Flood Depth (m)'] = df['Flood Depth (m)'].fillna(df['Flood Depth (m)'].mean())  # Fill NaN with the mean of the column

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f'MAE: {mae}, RMSE: {rmse}, R² Score: {r2}')

joblib.dump(model, 'rf_flood_model.pkl')
print("Model saved successfully!")