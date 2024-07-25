import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('models/StudentsPerformance.csv')

def preprocess_data(df):
    # Preprocess data and return X, y
    X = df.drop(columns=['math score'], axis=1)
    y = df['math score']
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    # Train models and return their predictions and evaluation metrics

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(), 

    }

    model_predictions = {}
    model_evaluations = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)  # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate Train and Test dataset
        train_eval = evaluate_model(y_train, y_train_pred)
        test_eval = evaluate_model(y_test, y_test_pred)

        model_predictions[model_name] = {
            'train_predictions': y_train_pred.tolist(),
            'test_predictions': y_test_pred.tolist()
        }

        model_evaluations[model_name] = {
            'train_eval': train_eval,
            'test_eval': test_eval
        }

    return model_predictions, model_evaluations

def evaluate_model(true, predicted):
    # Evaluate the model and return evaluation metrics
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_square': r2_square
    }

def scatter_plot(y_test, y_pred):
    # Create scatter plot and save it as an image
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.savefig('scatter_plot.png')
    plt.close()

# Main function to process data, train models, and generate plots
def process_data_and_models():
    X, y = preprocess_data(df)

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and get predictions and evaluations
    model_predictions, model_evaluations = train_models(X_train, X_test, y_train, y_test)

    # Generate scatter plot
    scatter_plot(y_test, model_predictions['Linear Regression']['test_predictions'])

    return model_predictions, model_evaluations


