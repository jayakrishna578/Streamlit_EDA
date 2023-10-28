import streamlit as st
import pandas as pd

def upload_data():
    """Upload and read a data file."""
    file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
    if file is not None:
        if file.type == "text/csv":
            data = pd.read_csv(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(file)
        else:
            st.error("Unsupported file type")
            return None
        return data
