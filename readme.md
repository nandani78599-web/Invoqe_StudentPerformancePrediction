# 🎓 Student Performance Prediction System

A Machine Learning project that predicts a student's final academic grade using demographic, social, family, and academic-related factors.

## 📌 Project Overview

This project was developed as part of the **Invoqe Artificial Intelligence & Machine Learning Internship — Task 1: Machine Learning Prediction System**.

The system performs data preprocessing, exploratory data analysis, feature selection, machine learning model training, model evaluation, and final-grade prediction.

An interactive **Streamlit web application** is also provided for making predictions using student information.

## 🎯 Objective

The main objective of this project is to build a machine learning system that predicts a student's final grade (`G3`) based on available student-related features.

## 📊 Dataset

The project uses the **Student Performance Dataset**.

Dataset file:

```text
student-mat.csv
```

The dataset contains:

- 395 student records
- 33 features
- Demographic information
- Family information
- Study-related information
- Social information
- Absence information
- Previous and final grades

The target variable is:

```text
G3
```

where `G3` represents the student's final grade.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset structure.
3. Checked for missing values.
4. Checked for duplicate records.
5. Removed duplicate records where applicable.
6. Separated categorical and numerical features.
7. Applied One-Hot Encoding to categorical features.
8. Split the dataset into training and testing sets.

The dataset was divided into:

- 80% training data
- 20% testing data

## 🔍 Exploratory Data Analysis

EDA was performed to understand the relationship between student characteristics and final grades.

The project includes visualizations for:

- Final grade distribution
- Study time vs final grade
- Absences vs final grade

## 🎯 Feature Selection

The prediction target is:

```text
G3
```

`G1` and `G2` were excluded from the input features so that the model predicts the final grade without directly relying on the previous grade values.

## 🤖 Machine Learning Models

Two regression models were implemented:

### 1. Linear Regression

Linear Regression was used as a baseline machine learning model.

### 2. Random Forest Regression

Random Forest Regression was implemented as the second model for comparison.

## 📈 Model Evaluation

The models were evaluated on the test dataset using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 3.40 | 4.20 | 0.14 |
| Random Forest | 2.97 | 3.75 | 0.31 |

These values represent the performance obtained on the project's 20% test split.

The Random Forest model produced lower MAE and RMSE and a higher R² score than the Linear Regression baseline on this test split.

## 🔮 Prediction System

The project includes an interactive Streamlit application.

Users can enter student information such as:

- School
- Age
- Gender
- Family information
- Education information
- Study time
- Previous failures
- Support information
- Social factors
- Absences

After clicking **Predict Final Grade**, the trained Random Forest model generates the predicted final grade.

## 🖥️ Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
python -m streamlit run app.py
```

### 3. Open the local application

Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open this URL in your browser.

## 📁 Project Structure

```text
Invoqe_StudentPerformancePrediction/
│
├── dataset/
│   └── student-mat.csv
│
├── src/
│   └── main.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## ✅ Task 1 Requirements Covered

- [x] Public dataset
- [x] Data cleaning and preprocessing
- [x] Exploratory Data Analysis
- [x] Feature selection
- [x] Machine learning model
- [x] Model evaluation
- [x] Prediction results
- [x] Multiple model comparison
- [x] Interactive prediction interface

## 👩‍💻 Project Status

**Completed — Ready for GitHub submission and internship documentation.**