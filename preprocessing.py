# prepare Both the regression task and the clustering task use these functions.
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Path to dataset file
DATA_PATH = "raw_data/Fitbit_dataset.csv"

# The columns that will use as numeric features
NUMERIC_COLS = [
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

TARGET_COL = "Calories_Burned"


def load_data():
    # Read the CSV file
    df = pd.read_csv(DATA_PATH)

    # Drop the extra index. that pandas sometimes adds
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # The dataset already has Base_MET, HR_Intensity and Effective_MET
    # These were used to CALCULATE Calories_Burned in the first
    # place (Calories_Burned = Effective_MET * Weight * Duration), so if
    # we keep them the model would just learn the formula instead of
    # actually learning from the workout data. We drop them here.
    leak_cols = ["Base_MET", "HR_Intensity", "Effective_MET"]
    for col in leak_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Rename the target column to a simpler name
    if "Calories_Burned (kcal)" in df.columns:
        df = df.rename(columns={"Calories_Burned (kcal)": TARGET_COL})

    return df


def clean_data(df):
    df = df.copy()

    # fill missing values with the median
    for col in NUMERIC_COLS + [TARGET_COL]:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # fill missing text columns with most common value
    for col in ["Gender", "Workout_Type", "Experience_Level"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove extreme outliers using the IQR method
    for col in NUMERIC_COLS:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower_limit, upper_limit)

    return df


def encode_data(df, drop_col=None):
    # drop_col is used in clustering to remove Workout_Type before
    # encoding, so it is not used as an input feature.
    df = df.copy()

    if drop_col is not None and drop_col in df.columns:
        df = df.drop(columns=[drop_col])

    # Convert Gender and Workout_Type into 0/1 columns
    text_cols = [c for c in ["Gender", "Workout_Type"] if c in df.columns]
    df = pd.get_dummies(df, columns=text_cols, drop_first=True)

    # convert text labels to numbers if needed
    if "Experience_Level" in df.columns:
        if df["Experience_Level"].dtype == object or str(
            df["Experience_Level"].dtype
        ).startswith("str"):
            level_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
            df["Experience_Level"] = df["Experience_Level"].map(level_map)

    return df


def scale_data(df, cols, scaler=None):
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


def get_clean_data():
    # Convenience function: load + clean in one step
    df = load_data()
    df = clean_data(df)
    return df
