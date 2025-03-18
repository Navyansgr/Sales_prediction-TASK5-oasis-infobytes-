import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
X = pd.read_csv('data/scaled_sales_data.csv')
y = pd.read_csv('data/cleaned_sales_data.csv')['Sales']

# Load model
model = joblib.load('app/model.pkl')

# Predictions
y_pred = model.predict(X)

# Evaluation Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"📊 Mean Squared Error: {mse:.2f}")
print(f"📈 R-squared Score: {r2:.2f}")
