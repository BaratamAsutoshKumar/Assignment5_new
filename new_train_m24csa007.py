

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import set_config
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv('data/hour.csv')
df['day_night'] = df['hr'].apply(lambda x: 'day' if 6 <= x <= 18 else 'night')
df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)
df['dteday'] = pd.to_datetime(df.dteday)
df.drop(columns=['dteday'], inplace=True)

# Handle categorical features
categorical_features = ['season', 'holiday', 'weekday', 'weathersit', 'workingday', 'mnth', 'yr', 'hr', 'day_night']
df[categorical_features] = df[categorical_features].astype('category')

# Separating features and target variable
X = df.drop(columns=['cnt']) # Features
y = df['cnt'] # Target

# Numerical and categorical pipelines
numerical_features = ['temp', 'hum', 'windspeed']
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, drop='first'))
])

# Apply pipelines
X[numerical_features] = numerical_pipeline.fit_transform(X[numerical_features])
X_encoded = pd.DataFrame(categorical_pipeline.fit_transform(X[categorical_features]),
                         columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features))

# Combine numerical and encoded categorical features
X = pd.concat([X.drop(columns=categorical_features), X_encoded], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to log the model and metrics
def log_model_to_mlflow(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    with mlflow.start_run(run_name=f"{model_name}_Run"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} logged successfully in MLflow.")
    
    return mse, r2

# MLflow experiment setup
experiment_name = "Bike_Sharing_Model_Experiments"
mlflow.set_experiment(experiment_name)

# Train and log Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
mse_rf, r2_rf = log_model_to_mlflow(random_forest_model, "RandomForest_Model", X_train, X_test, y_train, y_test)

# Train and log Linear Regression model
linear_regression_model = LinearRegression()
mse_lr, r2_lr = log_model_to_mlflow(linear_regression_model, "LinearRegression_Model", X_train, X_test, y_train, y_test)

# Compare the models based on MSE
if mse_rf < mse_lr:
    best_model = random_forest_model
    best_model_name = "RandomForest_Model"
    best_mse = mse_rf
    best_r2 = r2_rf
else:
    best_model = linear_regression_model
    best_model_name = "LinearRegression_Model"
    best_mse = mse_lr
    best_r2 = r2_lr

print(f"Best Model: {best_model_name} with MSE: {best_mse} and R-squared: {best_r2}")

# Log the best-performing model and register it to the Model Registry
with mlflow.start_run(run_name=f"Best_{best_model_name}_Run") as run:
    mlflow.log_param("model_type", best_model_name)
    mlflow.log_metric("mse", best_mse)
    mlflow.log_metric("r2", best_r2)
    mlflow.sklearn.log_model(best_model, best_model_name)
    print(f"Best-performing model ({best_model_name}) logged successfully in MLflow's Model Registry.")
    
    # Register the best model
    model_uri = f"runs:/{run.info.run_id}/{best_model_name}"
    mlflow.register_model(model_uri, best_model_name)

    print(f"Best model registered successfully in MLflow Model Registry under the name '{best_model_name}'.")
