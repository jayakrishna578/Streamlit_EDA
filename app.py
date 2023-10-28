import streamlit as st
from data_upload import upload_data
from eda import display_data, generate_summary, plot_distribution, plot_relationship, handle_missing_data, feature_engineering
from model_training import split_data, train_model, preprocess_data
from model_metrics import compute_metrics

def main():
    """Main function of the Streamlit application."""
    st.title("Explorative Data Analysis and Model Training Platform")

    data = upload_data()
    if data is not None:
        display_data(data)
        generate_summary(data)
        column = st.selectbox("Select a column to plot its distribution", data.columns)
        plot_distribution(data, column)
        x = st.selectbox("Select the x variable for scatter plot", data.columns)
        y = st.selectbox("Select the y variable for scatter plot", data.columns)
        plot_relationship(data, x, y)
        data = handle_missing_data(data)
        features = st.multiselect("Select features for feature engineering", data.columns)
        data = feature_engineering(data, features)

        model_type = st.selectbox("Select a model type", ["Linear Regression", "Decision Tree"])
        target = st.selectbox("Select the target variable", data.columns)
        test_size = st.slider("Select the test size ratio", 0.1, 0.5, 0.2)
        data = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(data, target, test_size)
        model = train_model(model_type, X_train, y_train)
        if model is not None:
            compute_metrics(model, X_test, y_test)

if __name__ == "__main__":
    main()
