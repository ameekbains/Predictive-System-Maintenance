# AI Model for Predictive Maintenance in Radio Systems

This project implements a predictive maintenance system for monitoring the performance of radio systems. By analyzing historical performance data, the system predicts when maintenance is needed based on key features such as signal strength, error rate, temperature, and system load. The project employs multiple machine learning algorithms like Logistic Regression, Random Forest, Gradient Boosting, and Stacking Classifiers to predict maintenance needs and assess the system’s health.

## Features:
- Predicts when maintenance is required based on historical performance data.
- Utilizes machine learning algorithms like Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
- Combines models using Stacking and Voting Classifiers.
- Supports uncertainty estimation through the Bootstrap method.
- Evaluates models using classification metrics and statistical tests (K-S Test).
- Implements time-series forecasting using ARIMA for signal strength prediction.

## Requirements

- Python 3.x
- Libraries: Pandas, NumPy, scikit-learn, XGBoost, statsmodels, SciPy, joblib, matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Predictive_Systems_Maintenance.git
    cd Predictive_Systems_Maintenance
    ```

2. Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Preprocess Data

First, preprocess the historical data (e.g., signal strength, temperature, error rate) by cleaning, normalizing, and handling missing values. You can run the preprocessing script located in `src/preprocessing.py`.

Example usage:

```python
from src.preprocessing import load_data, preprocess_data

data = load_data('data/radio_performance_data.csv')
processed_data = preprocess_data(data)
