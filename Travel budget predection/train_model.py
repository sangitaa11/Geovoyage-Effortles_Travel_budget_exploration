# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the travel data
travel_data = pd.read_csv('travel.csv')

# Check for missing values
print("Missing values in the dataset:")
print(travel_data.isnull().sum())

# Drop rows with missing values in the target variable 'user_budget'
travel_data = travel_data.dropna(subset=['user_budget'])

# Drop 'user_id' column
travel_data = travel_data.drop(columns=['user_id'])

# Convert currency values to float (remove '₹' symbol)
currency_columns = ['accommodation_cost_per_night', 'transportation_cost', 'daily_meal_expenses', 'user_budget']
for column in currency_columns:
    travel_data[column] = travel_data[column].replace('[\₹,]', '', regex=True).astype(float)

# Convert categorical variables (e.g., gender, travel_preferences, purpose, accommodation_type, transportation_mode)
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'travel_preferences', 'purpose', 'accommodation_type', 'transportation_mode']
for column in categorical_columns:
    travel_data[column] = label_encoder.fit_transform(travel_data[column])

# Save the preprocessed dataset to a new CSV file
travel_data.to_csv('preprocessed_travel.csv', index=False)

# Select relevant features for the budget prediction model
features = ['age', 'duration', 'current_cost_of_living_index', 'destination_cost_of_living_index',
            'accommodation_cost_per_night', 'transportation_cost', 'daily_meal_expenses',
            'gender', 'travel_preferences', 'purpose', 'actual_expenses']

X = travel_data[features]
y = travel_data['user_budget']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Save the trained model to a file
joblib.dump(model, 'trained_model.joblib')
import matplotlib.pyplot as plt

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted User Budget')
plt.xlabel('Actual User Budget')
plt.ylabel('Predicted User Budget')
plt.show()
