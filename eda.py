import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display_data(data):
    """Display the first few rows of the dataset."""
    st.dataframe(data.head())

def generate_summary(data):
    """Generate summary statistics."""
    st.write(data.describe())

def plot_distribution(data, column):
    """Plot the distribution of a numerical variable."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column])
    st.pyplot(plt)

def plot_relationship(data, x, y):
    """Plot the relationship between two variables."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x, y=y)
    st.pyplot(plt)

def handle_missing_data(data):
    """Handle missing data."""
    st.write("Missing values before handling:", data.isnull().sum())
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    non_numeric_data = data.select_dtypes(exclude=['float64', 'int64'])
    numeric_data = numeric_data.fillna(numeric_data.mean())
    non_numeric_data = non_numeric_data.fillna('missing')
    data = pd.concat([numeric_data, non_numeric_data], axis=1)
    st.write("Missing values after handling:", data.isnull().sum())
    return data

def feature_engineering(data, features):
    """Perform feature engineering."""
    data = data[features]
    return data
