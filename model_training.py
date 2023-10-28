import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def preprocess_data(data):
    """Convert categorical data to numeric format using one-hot encoding."""
    # Identify categorical columns
    cat_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical columns
    for col in cat_columns:
        dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)
    
    return data

def split_data(data, target, test_size):
    """Split the dataset into training and test sets."""
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def train_model(model_type, X_train, y_train):
    """Train a machine learning model."""
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor()
    else:
        st.error("Unsupported model type")
        return None
    model.fit(X_train, y_train)
    return model