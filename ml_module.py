"""Machine Learning Module

This module contains functions for loading data, training a model with cross-validation, and making predictions.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(path)
        df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")

def create_target_and_predictors(
    data: pd.DataFrame, 
    target: str = "estimated_stock_pct"
) -> tuple:
    """Create target and predictors from the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.
        target (str, optional): Target column name. Defaults to "estimated_stock_pct".

    Returns:
        tuple: Tuple containing predictors and target.
    """
    if target not in data.columns:
        raise ValueError(f"Target: {target} is not present in the data")
    x = data.drop(columns=[target])
    y = data[target]
    return x, y

def train_algorithm_with_cross_validation(
    x: pd.DataFrame, 
    y: pd.Series,
    k: int = 10,
    split: float = 0.75
) -> RandomForestRegressor:
    """Train a machine learning algorithm with cross-validation.

    Args:
        x (pd.DataFrame): Features.
        y (pd.Series): Target.
        k (int, optional): Number of folds for cross-validation. Defaults to 10.
        split (float, optional): Train-test split ratio. Defaults to 0.75.

    Returns:
        RandomForestRegressor: Trained model.
    """
    accuarcy = []
    for i in range(k):
        model = RandomForestRegressor()
        scale = StandardScaler()
        
        X_train, x_test, Y_train, y_test = train_test_split(x, y, train_size=split, random_state=42)
        scale.fit(X_train)
        X_train = scale.transform(X_train)
        x_test = scale.transform(x_test)
        
        trained_model = model.fit(X_train, Y_train)
        
        y_pred = trained_model.predict(x_test)
        
        avg = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuarcy.append(avg)
        print(f"Fold {i+1}: Mean Absolute Error: {avg:.3f}")
        
    avg_mean = sum(accuarcy) / len(accuarcy)
    print(f"Average Mean Absolute Error: {avg_mean:.2f}")
    
    return trained_model
