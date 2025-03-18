import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load cleaned data
data = pd.read_csv('data/cleaned_sales_data.csv')

# Features and Target
X = data.drop('Sales', axis=1)
y = data['Sales']

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaled data and scaler
joblib.dump(scaler, 'app/scaler.pkl')
pd.DataFrame(X_scaled).to_csv('data/scaled_sales_data.csv', index=False)
print("✅ Feature Engineering Completed")
