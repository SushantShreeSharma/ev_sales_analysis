Objective

The project aims to analyze and predict Electric Vehicle (EV) sales across different states in India using a dataset containing sales data. By applying data preprocessing, exploratory analysis, and machine learning models, the project seeks to uncover patterns in EV sales and evaluate the predictive performance of two regression models: Linear Regression and Random Forest Regressor.
Dataset

Source: The dataset is loaded from a CSV file named Electric Vehicle Sales by State in India.csv. 
Assumed Columns:

EV_Sales: The target variable representing the number of electric vehicles sold (assumed based on the code). 
State: A categorical variable indicating the state in India.
Other potential numerical or categorical features (not explicitly listed but implied by the use of df.drop('EV_Sales', axis=1)).

Initial Exploration: 
Basic information about the dataset (e.g., data types, number of entries) is displayed using df.info().
Missing values are checked with df.isnull().sum()

Methodology  
Data Preprocessing:

Handling Missing Values: Missing values are filled with the mean of each column using df.fillna(df.mean(), inplace=True). This assumes numerical columns; categorical missing values might need different handling in practice.
Categorical Encoding: The State column is converted into dummy variables using one-hot encoding (pd.get_dummies), with the first category dropped to avoid multicollinearity.

Model Development:
Models Used:
Linear Regression: A simple linear model to predict EV sales based on the features.
Random Forest Regressor: An ensemble model to capture non-linear relationships and feature interactions.
Training: Both models are trained on the standardized training data (X_train, y_train).
Prediction: Predictions are made on the test set (X_test) for evaluation.
Model Evaluation:

Metrics:
Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
Mean Squared Error (MSE): Measures the average squared difference, penalizing larger errors more heavily.
R² Score: Indicates the proportion of variance in the target variable explained by the model (ranges from 0 to 1, with 1 being perfect prediction).
Visualization:

Feature Importance: A bar plot is generated using seaborn to visualize the importance of each feature in the Random Forest model, based on rf_reg.feature_importances_. This helps identify which variables (e.g., specific states or other features) most influence EV sales predictions.
Key Components
Libraries Used:

pandas and numpy: Data manipulation and numerical operations.
seaborn and matplotlib: Data visualization.
sklearn: Machine learning tools (preprocessing, model training, evaluation).

Output:
Printed metrics for model performance (MAE, MSE, R²).
A bar plot showing feature importance for the Random Forest model.
Potential Findings

The project could reveal which states or features (e.g., economic factors, infrastructure) most strongly correlate with EV sales.
Model comparison might show whether a simple linear approach (Linear Regression) suffices or if a more complex model (Random Forest) better captures the data’s patterns.

The feature importance plot could guide policy or business decisions by highlighting key drivers of EV adoption.
Limitations
Assumed Dataset Structure: The code assumes EV_Sales as the target and State as a feature, but the actual dataset structure isn’t provided, so additional features are unclear.

Missing Value Handling: Using the mean might not be optimal for all columns, especially categorical ones.
Single Visualization: Only feature importance is visualized; additional plots (e.g., actual vs. predicted sales) could enhance insights.
