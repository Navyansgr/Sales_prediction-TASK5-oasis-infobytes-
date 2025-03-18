import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load data
X = pd.read_csv('data/scaled_sales_data.csv')
y = pd.read_csv('data/cleaned_sales_data.csv')['Sales']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'app/model.pkl')
print("✅ Model Training Completed")
