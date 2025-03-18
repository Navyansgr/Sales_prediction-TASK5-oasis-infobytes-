import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('data/sales_data.csv')

# Drop unnecessary columns
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Save cleaned data
data.to_csv('data/cleaned_sales_data.csv', index=False)
print("✅ Data Cleaning Completed")
