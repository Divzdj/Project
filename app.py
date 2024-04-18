import streamlit as st
import pandas as pd

from predict_page1 import show_predict_page
from explore_page import show_explore_page

# Define CSS style for the background theme
attrition_theme = """
    <style>
    body {
        background-color: #f0f0f0; /* Set background color */
        font-family: Arial, sans-serif; /* Set font family */
    }
    .stApp {
        max-width: 1000px; /* Set max width for the app */
        margin: 0 auto; /* Center align the app */
    }
    .stSidebar > div:first-child {
        background-color: #ffffff; /* Set sidebar background color */
        border-radius: 10px; /* Add border radius to sidebar */
        padding: 20px; /* Add padding to sidebar */
        box-shadow: 0px 0px 20px 0px rgba(0, 0, 0, 0.1); /* Add box shadow to sidebar */
    }
    </style>
"""

# Apply the custom CSS theme
st.markdown(attrition_theme, unsafe_allow_html=True)

# Function to display the home page content
def show_home_page():
    st.title("Employee Attrition Prediction and Exploration")
    st.write("Welcome to the Employee Attrition App!")
    st.write("Choose an option from the sidebar to predict employee attrition or explore the dataset.")
    st.image("img1.png", use_column_width=True)





# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ("Home", "Predict", "Explore"))

if page == "Home":
    show_home_page()
elif page == "Predict":
    show_predict_page()  # Pass input_data to the show_predict_page function
else:
    show_explore_page()
