import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_data
def load_data():
    df = pd.read_csv("HR-Employee-Attrition.csv")
    df.drop(["Over18", "EmployeeCount", "StandardHours", "EmployeeNumber"], axis=1, inplace=True)
    return df

df = load_data()
def show_explore_page():
    st.write("## Employee Attrition Data Exploration")
    st.write("#### Distribution of Age of Employees")

    sns.histplot(df['Age'], bins=20, kde=True)
    fig, ax = plt.subplots()  # Create a new figure
    ax = sns.histplot(df['Age'], bins=20, kde=True)  # Plot on the new figure
    st.pyplot(fig)


    st.write("#### Count of Attrition")

    attrition_count = df['Attrition'].value_counts()
    fig, ax = plt.subplots()
    ax = sns.barplot(x=attrition_count.index, y=attrition_count.values)
    st.pyplot(fig)

    st.write("#### Attrition by Department")
    attrition_by_department = df.groupby('Department')['Attrition'].value_counts().unstack()
    fig, ax = plt.subplots(figsize=(8, 6))
    attrition_by_department.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

    st.write("## Attrition by Job Role")
    attrition_by_job_role = df.groupby('JobRole')['Attrition'].value_counts().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    attrition_by_job_role.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

    st.write("## Attrition by Education Field")
    attrition_by_education_field = df.groupby('EducationField')['Attrition'].value_counts().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    attrition_by_education_field.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)



