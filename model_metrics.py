import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(model, X_test, y_test):
    """Compute and display model metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean Absolute Error:", mae)
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)
