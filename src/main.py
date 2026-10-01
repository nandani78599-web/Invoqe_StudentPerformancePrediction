import pandas as pd

# Load the dataset
data = pd.read_csv(
    "dataset/student-mat.csv",
    sep=";",
    encoding="latin1"
)

# Display first 5 rows
print("========== FIRST 5 ROWS ==========")
print(data.head())

# Dataset shape
print("\n========== DATASET SHAPE ==========")
print(data.shape)

# Column names
print("\n========== COLUMN NAMES ==========")
print(data.columns.tolist())

# Dataset information
print("\n========== DATASET INFO ==========")
print(data.info())

# Missing values
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

# ==============================
# DATA CLEANING
# ==============================

# Check for duplicate rows
print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Check data types
print("\n========== DATA TYPES ==========")
print(data.dtypes)

# Check missing values again
print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(data.isnull().sum().sum())

# Final dataset shape
print("\n========== FINAL DATASET SHAPE ==========")
print(data.shape)


import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# EXPLORATORY DATA ANALYSIS
# ==============================

# 1. Final Grade Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data["G3"], bins=10, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.show()


# 2. Study Time vs Final Grade
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=data)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.show()


# 3. Absences vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(x="absences", y="G3", data=data)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.show()

# ==============================
# FEATURE SELECTION
# ==============================

# Target variable
target = "G3"

# Remove target and previous grades
X = data.drop(columns=["G3", "G1", "G2"])
y = data["G3"]

print("\n========== FEATURES ==========")
print(X.columns.tolist())

print("\n========== TARGET ==========")
print(target)

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])



# ==============================
# DATA PREPROCESSING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object", "str"]).columns
numerical_columns = X.select_dtypes(exclude=["object", "str"]).columns

print("\n========== CATEGORICAL FEATURES ==========")
print(categorical_columns.tolist())

print("\n========== NUMERICAL FEATURES ==========")
print(numerical_columns.tolist())

# Preprocessing
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==============================
# MODEL TRAINING - LINEAR REGRESSION
# ==============================

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create model pipeline
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== LINEAR REGRESSION RESULTS ==========")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 2))


# ==============================
# MODEL 2 - RANDOM FOREST
# ==============================

from sklearn.ensemble import RandomForestRegressor

# Create Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions
rf_pred = random_forest_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n========== RANDOM FOREST RESULTS ==========")
print("Mean Absolute Error (MAE):", round(rf_mae, 2))
print("Root Mean Squared Error (RMSE):", round(rf_rmse, 2))
print("R² Score:", round(rf_r2, 2))


# ==============================
# MODEL COMPARISON
# ==============================

print("\n========== MODEL COMPARISON ==========")

print("\nLinear Regression:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))

print("\nRandom Forest:")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R²  :", round(rf_r2, 2))


# ==============================
# STUDENT PERFORMANCE PREDICTION
# ==============================

print("\n========== STUDENT PERFORMANCE PREDICTION ==========")

# Example student data
student = pd.DataFrame({
    "school": ["GP"],
    "sex": ["F"],
    "age": [17],
    "address": ["U"],
    "famsize": ["GT3"],
    "Pstatus": ["T"],
    "Medu": [3],
    "Fedu": [3],
    "Mjob": ["teacher"],
    "Fjob": ["services"],
    "reason": ["course"],
    "guardian": ["mother"],
    "traveltime": [1],
    "studytime": [3],
    "failures": [0],
    "schoolsup": ["yes"],
    "famsup": ["yes"],
    "paid": ["no"],
    "activities": ["yes"],
    "nursery": ["yes"],
    "higher": ["yes"],
    "internet": ["yes"],
    "romantic": ["no"],
    "famrel": [4],
    "freetime": [3],
    "goout": [3],
    "Dalc": [1],
    "Walc": [1],
    "health": [4],
    "absences": [4]
})

# Predict using Random Forest
prediction = random_forest_model.predict(student)

print("Predicted Final Grade (G3):", round(prediction[0], 2))

