import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('/mnt/data/Electric Vehicle Sales by State in India.csv')

# Display basic info and check for missing values
print(df.info())
print(df.isnull().sum())

# Fill missing values (example: replacing NaNs with mean)
df.fillna(df.mean(), inplace=True)

# Convert categorical variables into dummy variables (if applicable)
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Define features and target variable
X = df.drop('EV_Sales', axis=1)  # Assuming 'EV_Sales' is the target column
y = df['EV_Sales']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor()

# Train models
lin_reg.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)

# Make predictions
lin_reg_pred = lin_reg.predict(X_test)
rf_reg_pred = rf_reg.predict(X_test)

# Evaluate models
def evaluate_model(model_name, y_true, y_pred):
    print(f"{model_name} Metrics")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.2f}\n")

evaluate_model("Linear Regression", y_test, lin_reg_pred)
evaluate_model("Random Forest Regressor", y_test, rf_reg_pred)

# Simple visualization (Feature importance for Random Forest)
importances = rf_reg.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.show()
