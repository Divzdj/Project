import streamlit as st
import pickle
import numpy as np
import xgboost as xgb
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_model():
    with open('XGBClassifier_model.pkl', 'rb') as file:
        datas = pickle.load(file)
    return datas


datas = load_model()

XGB_loaded = datas["XGBClassifier"]

Business, Department, Education, Gender, Job, Marital, Over = (
    datas[key] for key in ["Business", "Department", "Education", "Gender", "Job", "Marital", "Over"]
)
smt = SMOTE()  # Load SMOTE object
scaler = MinMaxScaler()  # Load MinMaxScaler object

label_encoders = {
    "BusinessTravel": Business,
    "Department": Department,
    "EducationField": Education,
    "Gender": Gender,
    "JobRole": Job,
    "MaritalStatus": Marital,
    "OverTime": Over
}


def encode_input(input_data):
    encoded_data = []
    for key, encoder in label_encoders.items():
        encoded_value = encoder.transform([input_data[key]])[0]
        encoded_data.append(encoded_value)
    return encoded_data


def show_predict_page():
    st.title("Employee Attrition Prediction")
    st.write("""### We need some information to predict employee attrition""")

    Business_Travel = ("Travel_Rarely", "Travel_Frequently", "Non-Travel")
    Department_ = ("Sales", "Research & Development", "Human Resources")
    Education_Field = ("Life Sciences", "Medical", "Marketing",
                       "Technical Degree", "Human Resources", "Other")
    Gender_ = ("Female", "Male")
    Job_Role = ("Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources")
    Marital_Status = ("Single", "Married", "Divorced")
    Over_Time = ("Yes", "No")

    Business_Travel = st.selectbox("Business Travel", Business_Travel)
    Department_ = st.selectbox("Department", Department_)
    Education_Field = st.selectbox("Education Field", Education_Field)
    Gender_ = st.selectbox("Gender", Gender_)
    Job_Role = st.selectbox("Job Role", Job_Role)
    Marital_Status = st.selectbox("Marital Status", Marital_Status)
    Over_Time = st.selectbox("Over Time", Over_Time)

    age = st.slider("Age", 18, 60, 20)
    DailyRate = st.slider("DailyRate", 100, 1500, 120)
    DistanceFromHome = st.slider("Distance From Home(Kms)", 0, 30, 3)
    Education_ = st.slider("Level of Education", 1, 5, 2)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction Level", 1, 4, 2)
    HourlyRate = st.slider("Hourly Rate", 0, 100, 30)
    JobInvolvement = st.slider("Job Involvement Level", 1, 4, 2)
    JobLevel = st.slider("Job Level", 1, 5, 2)
    JobSatisfaction = st.slider("Job Satisfaction Level", 1, 4, 2)
    MonthlyIncome = st.slider("Monthly Income", 1000, 20000, 1500)
    MonthlyRate = st.slider("Monthly Rate", 2000, 30000, 2500)
    NumCompaniesWorked = st.slider("No.of companies previously worked", 0, 10, 2)
    PercentSalaryHike = st.slider("Percent of salary Hike", 10, 25, 12)
    PerformanceRating = st.slider("Performance Rating", 1, 5, 2)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction Level", 1, 4, 2)
    StockOptionLevel = st.slider("Stock Option Level", 0, 3, 2)
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 3)
    TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 2)
    WorkLifeBalance = st.slider("Work Life Balance Level", 1, 4, 2)
    YearsAtCompany = st.slider("Years At Current Company", 0, 40, 3)
    YearsInCurrentRole = st.slider("Years In Current Role", 0, 40, 4)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 3)
    YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20, 3)

    ok = st.button("Employee Attrition Prediction")
    if ok:
        # Encode categorical variables
        encoded_data = encode_input({
            "BusinessTravel": Business_Travel,
            "Department": Department_,
            "EducationField": Education_Field,
            "Gender": Gender_,
            "JobRole": Job_Role,
            "MaritalStatus": Marital_Status,
            "OverTime": Over_Time
        })

        print("Encoded data:", encoded_data)  # Debug print

        # Construct input array
        x = np.array([[*encoded_data,
                       age, DailyRate, DistanceFromHome, Education_, EnvironmentSatisfaction,
                       HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome,
                       MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating,
                       RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
                       WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
                       YearsWithCurrManager]])

        print("Input array shape:", x.shape)  # Debug print

        # Apply MinMax scaling
        x_scaled = scaler.fit_transform(x)

        print("Scaled input array:", x_scaled)  # Debug print



        def convert_prediction(attrition):
            return "Yes" if attrition == 1 else "No"

        # Usage:
        attrition_result = XGB_loaded.predict(x_scaled)
        attrition_prediction = convert_prediction(attrition_result[0])
        st.subheader(f"The employee attrition prediction is {attrition_prediction}")


