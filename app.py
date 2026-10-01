import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv(
    "dataset/student-mat.csv",
    sep=";",
    encoding="latin1"
)

# Remove duplicate rows
data = data.drop_duplicates()


# ==============================
# FEATURES AND TARGET
# ==============================

X = data.drop(columns=["G3", "G1", "G2"])
y = data["G3"]


# ==============================
# PREPROCESSING
# ==============================

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==============================
# TRAIN MODEL
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

model.fit(X_train, y_train)


# ==============================
# STREAMLIT INTERFACE
# ==============================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter student information below to predict the final grade."
)

st.divider()


# ==============================
# USER INPUTS
# ==============================

school = st.selectbox(
    "School",
    ["GP", "MS"]
)

sex = st.selectbox(
    "Sex",
    ["F", "M"]
)

age = st.slider(
    "Age",
    15,
    22,
    17
)

address = st.selectbox(
    "Address",
    ["U", "R"]
)

famsize = st.selectbox(
    "Family Size",
    ["GT3", "LE3"]
)

Pstatus = st.selectbox(
    "Parent Status",
    ["T", "A"]
)

Medu = st.slider(
    "Mother's Education",
    0,
    4,
    2
)

Fedu = st.slider(
    "Father's Education",
    0,
    4,
    2
)

Mjob = st.selectbox(
    "Mother's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

Fjob = st.selectbox(
    "Father's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

reason = st.selectbox(
    "Reason for Choosing School",
    ["home", "reputation", "course", "other"]
)

guardian = st.selectbox(
    "Guardian",
    ["mother", "father", "other"]
)

traveltime = st.slider(
    "Travel Time",
    1,
    4,
    1
)

studytime = st.slider(
    "Study Time",
    1,
    4,
    2
)

failures = st.slider(
    "Past Class Failures",
    0,
    3,
    0
)

schoolsup = st.selectbox(
    "Extra School Support",
    ["yes", "no"]
)

famsup = st.selectbox(
    "Family Educational Support",
    ["yes", "no"]
)

paid = st.selectbox(
    "Extra Paid Classes",
    ["yes", "no"]
)

activities = st.selectbox(
    "Extra-Curricular Activities",
    ["yes", "no"]
)

nursery = st.selectbox(
    "Attended Nursery School",
    ["yes", "no"]
)

higher = st.selectbox(
    "Wants Higher Education",
    ["yes", "no"]
)

internet = st.selectbox(
    "Internet Access",
    ["yes", "no"]
)

romantic = st.selectbox(
    "In a Romantic Relationship",
    ["yes", "no"]
)

famrel = st.slider(
    "Family Relationship Quality",
    1,
    5,
    4
)

freetime = st.slider(
    "Free Time",
    1,
    5,
    3
)

goout = st.slider(
    "Going Out",
    1,
    5,
    3
)

Dalc = st.slider(
    "Weekday Alcohol Consumption",
    1,
    5,
    1
)

Walc = st.slider(
    "Weekend Alcohol Consumption",
    1,
    5,
    1
)

health = st.slider(
    "Current Health",
    1,
    5,
    3
)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=75,
    value=4
)


# ==============================
# CREATE INPUT DATA
# ==============================

student = pd.DataFrame({
    "school": [school],
    "sex": [sex],
    "age": [age],
    "address": [address],
    "famsize": [famsize],
    "Pstatus": [Pstatus],
    "Medu": [Medu],
    "Fedu": [Fedu],
    "Mjob": [Mjob],
    "Fjob": [Fjob],
    "reason": [reason],
    "guardian": [guardian],
    "traveltime": [traveltime],
    "studytime": [studytime],
    "failures": [failures],
    "schoolsup": [schoolsup],
    "famsup": [famsup],
    "paid": [paid],
    "activities": [activities],
    "nursery": [nursery],
    "higher": [higher],
    "internet": [internet],
    "romantic": [romantic],
    "famrel": [famrel],
    "freetime": [freetime],
    "goout": [goout],
    "Dalc": [Dalc],
    "Walc": [Walc],
    "health": [health],
    "absences": [absences]
})


# ==============================
# PREDICTION
# ==============================

if st.button("🔮 Predict Final Grade"):

    prediction = model.predict(student)[0]

    st.success(
        f"Predicted Final Grade (G3): {prediction:.2f}"
    )

    st.info(
        "The prediction is generated using the trained Random Forest model."
    )