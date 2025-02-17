import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.utils import resample
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Load the historical performance data
data = pd.read_csv("radio_performance_data.csv")

# 1. Data Preprocessing

# Handle missing values (impute or drop based on context)
data.fillna(method="ffill", inplace=True)

# Normalize continuous features
scaler = StandardScaler()
data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']] = scaler.fit_transform(
    data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']]
)

# Outlier detection using IQR (Interquartile Range)
Q1 = data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']].quantile(0.25)
Q3 = data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']].quantile(0.75)
IQR = Q3 - Q1
filtered_data = data[~((data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']] < (Q1 - 1.5 * IQR)) | 
                       (data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Split into features and labels
X = filtered_data[['Signal Strength (dBm)', 'Temperature (°C)', 'Error Rate', 'System Load (%)']]
y = filtered_data['Maintenance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Base Models
logreg_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

logreg_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 3. Bootstrap for Model Validation (Uncertainty Estimation)
def bootstrap_sample(model, X_train, y_train, n_iterations=1000):
    scores = []
    for _ in range(n_iterations):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=42)
        model.fit(X_resampled, y_resampled)
        scores.append(model.score(X_train, y_train))
    return np.mean(scores), np.std(scores)

# Example with Random Forest
mean_score, std_score = bootstrap_sample(rf_model, X_train, y_train)
print(f"Bootstrap mean score: {mean_score:.4f}, Std score: {std_score:.4f}")

# 4. K-S Test for Distribution Comparison (Between training and test data)
stat, p_value = ks_2samp(X_train['Signal Strength (dBm)'], X_test['Signal Strength (dBm)'])
print(f"K-S Test statistic: {stat:.4f}, p-value: {p_value:.4f}")

# 5. ARIMA for Time-Series Forecasting (Example with Signal Strength as a time-series)
# Assuming 'Signal Strength (dBm)' is a time-series and 'date' is a datetime column
# data['date'] = pd.to_datetime(data['date'])  # Uncomment if there is a 'date' column
# model = ARIMA(data['Signal Strength (dBm)'], order=(5, 1, 0))  # ARIMA(p,d,q) model
# model_fit = model.fit()
# forecast = model_fit.forecast(steps=10)
# print("ARIMA forecast:", forecast)

# 6. Combine Models Using Stacking Classifier
base_models = [
    ('logreg', logreg_model),
    ('rf', rf_model),
    ('gb', gb_model),
    ('xgb', xgb_model)
]

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)

# 7. Combine Models Using Voting Classifier (Soft Voting)
voting_clf = VotingClassifier(estimators=[('logreg', logreg_model), ('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model)], voting='soft')
voting_clf.fit(X_train, y_train)

# 8. Model Evaluation
models = {
    "Logistic Regression": logreg_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
    "Stacking Classifier": stacking_clf,
    "Voting Classifier": voting_clf
}

for model_name, model in models.items():
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, pred))

# 9. Save the Best Model (Stacking Classifier in this case)
joblib.dump(stacking_clf, 'predictive_maintenance_model.pkl')

# Load the model for inference
loaded_model = joblib.load('predictive_maintenance_model.pkl')

# Example new data for inference
new_data = np.array([[0.2, 30, 0.05, 85]])  # Example new data point
new_data = scaler.transform(new_data)  # Apply scaling
prediction = loaded_model.predict(new_data)
print("Maintenance prediction:", prediction)
