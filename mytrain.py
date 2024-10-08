import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load the dataset
df = pd.read_csv('data/hour.csv')
int_columns = df.select_dtypes(include=['int']).columns
df[int_columns] = df[int_columns].astype('float64')
df['day_night'] = df['hr'].apply(lambda x: 'day' if 6 <= x <= 18 else 'night')
df.drop(['instant', 'casual', 'registered', 'dteday'], axis=1, inplace=True)

# Features and target
X = df.drop(columns=['cnt'])
y = df['cnt']

# Numerical features pipeline
numerical_features = ['temp', 'hum', 'windspeed']
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Categorical features pipeline
categorical_features = ['season', 'weathersit', 'day_night']
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, drop='first'))
])

# Preprocessing
X[numerical_features] = numerical_pipeline.fit_transform(X[numerical_features])
X_encoded = categorical_pipeline.fit_transform(X[categorical_features])
X_encoded = pd.DataFrame(X_encoded, columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features))
X = pd.concat([X.drop(columns=categorical_features), X_encoded], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow tracking
mlflow.set_experiment("Bike Sharing Model Training")

# Initialize variables to track the best model
best_model = None
best_mse = float('inf')  # Set to infinity so any MSE will be smaller
best_r2 = 0.0000
results = {}  # To store results for each model

# Function to train and log models
def train_and_log_model(model, model_name):
    global best_model, best_mse, results, best_r2
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate MSE and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model_name} Mean Squared Error: {mse}")
        print(f"{model_name} R-squared: {r2}")
        
        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log the model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        print(f"{model_name} model logged successfully!")
        
        # Store results
        results[model_name] = {'mse': mse, 'r2': r2}

        # Check if this model has the best performance
        if mse < best_mse:
            best_mse = mse
            best_model = model_name
        if r2 > best_r2:
            best_r2 = r2
        
    # End the run explicitly
    mlflow.end_run()

# Train and log Linear Regression
linear_model = LinearRegression()
train_and_log_model(linear_model, "Linear_Regression")

# Train and log Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
train_and_log_model(rf_model, "Random_Forest")

# Print comparison of the models
print("\nModel Comparison:")
for model_name, metrics in results.items():
    print(f"Model: {model_name}, MSE: {metrics['mse']}, R2: {metrics['r2']}")

# Log and register the best-performing model
if best_model:
    print(f"\nRegistering {best_model} as the best model with MSE: {best_mse}")
    with mlflow.start_run(run_name=f"Register Best Model: {best_model}"):
        mlflow.log_param("model_type", best_model)
        mlflow.log_metric("mse", best_mse)
        mlflow.log_metric("r2", best_r2)
        # Get the best model run_id and register the best model in the registry
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{best_model}", "Bike_Sharing_Best_Model")
    
    mlflow.end_run()  # End this run after registering the best model
