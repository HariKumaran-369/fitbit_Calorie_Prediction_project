import joblib
import pandas as pd
import streamlit as st

st.title("Fitbit Calorie Burn Predictor")
st.write("Fill in your workout details below to predict calories burned.")

# Load the saved model, scaler, and feature column list
model = joblib.load("models/best_regression_model.pkl")
scaler = joblib.load("models/regression_scaler.pkl")
feature_columns = joblib.load("models/regression_feature_columns.pkl")

# Split page left and right
left_col, right_col = st.columns(2)

# Left column - profile and body details
with left_col:
    st.subheader("Profile")
    age = st.number_input("Age", 15, 80, 28)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 35.0, 150.0, 70.0)
    height = st.number_input("Height (m)", 1.3, 2.2, 1.70)
    fat_percentage = st.number_input("Fat Percentage", 5.0, 50.0, 22.0)
    max_bpm = st.number_input("Max BPM", 120, 220, 185)
    avg_bpm = st.number_input("Avg BPM", 80, 200, 140)
    resting_bpm = st.number_input("Resting BPM", 40, 100, 65)

# Right column - workout details
with right_col:
    st.subheader("Workout")
    duration = st.number_input("Session Duration (hours)", 0.1, 3.0, 1.0)
    workout_type = st.selectbox(
        "Workout Type", ["Cardio", "Strength", "HIIT", "Yoga", "Mixed"]
    )
    water_intake = st.number_input("Water Intake (liters)", 0.5, 6.0, 2.5)
    workout_freq = st.slider("Workout Frequency (days/week)", 1, 7, 3)
    experience = st.selectbox(
        "Experience Level", ["Beginner", "Intermediate", "Advanced"]
    )

if st.button("Predict Calories Burned"):
    bmi = round(weight / (height**2), 1)
    experience_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

    # Build one row of input data matching the training format
    row = {
        "Age": age,
        "Weight (kg)": weight,
        "Height (m)": height,
        "BMI": bmi,
        "Fat_Percentage": fat_percentage,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": duration,
        "Water_Intake (liters)": water_intake,
        "Workout_Frequency (days/week)": workout_freq,
        "Experience_Level": experience_map[
            experience if experience else "Beginner"
        ],
        "Gender_Male": 1 if gender == "Male" else 0,
        "Workout_Type_HIIT": 1 if workout_type == "HIIT" else 0,
        "Workout_Type_Mixed": 1 if workout_type == "Mixed" else 0,
        "Workout_Type_Strength": 1 if workout_type == "Strength" else 0,
        "Workout_Type_Yoga": 1 if workout_type == "Yoga" else 0,
    }

    input_df = pd.DataFrame([row])

    # Make sure all expected columns are present, in the right order
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Scale the numeric columns the same way already did during training
    numeric_cols = [
        "Age",
        "Weight (kg)",
        "Height (m)",
        "BMI",
        "Fat_Percentage",
        "Max_BPM",
        "Avg_BPM",
        "Resting_BPM",
        "Session_Duration (hours)",
        "Water_Intake (liters)",
        "Workout_Frequency (days/week)",
    ]
    numeric_cols = [c for c in numeric_cols if c in input_df.columns]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    st.success(
        "Estimated Calories Burned: " + str(round(prediction, 1)) + " kcal"
    )
