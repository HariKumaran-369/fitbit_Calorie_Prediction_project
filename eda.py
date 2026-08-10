import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import get_clean_data, NUMERIC_COLS, TARGET_COL
matplotlib.use("Agg")

FIG_DIR = "reports/figures"


def run_eda():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = get_clean_data()

    print("Shape of data:", df.shape)
    print()
    print("Missing values:")
    print(df.isnull().sum())
    print()
    print("Summary statistics:")
    print(df.describe())

    # target column
    plt.figure(figsize=(7, 5))
    sns.histplot(df[TARGET_COL], kde=True, color="steelblue")
    plt.title("Distribution of Calories Burned")
    plt.savefig(FIG_DIR + "/target_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[NUMERIC_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(FIG_DIR + "/correlation_heatmap.png")
    plt.close()

    # Boxplots outliers check
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, col in zip(axes.flatten(), NUMERIC_COLS):
        sns.boxplot(y=df[col], ax=ax, color="lightgreen")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR + "/boxplots_numeric.png")
    plt.close()

    # Count categorical columns
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    cat_cols = ["Gender", "Workout_Type", "Experience_Level"]
    for ax, col in zip(axes, cat_cols):
        sns.countplot(x=df[col], ax=ax)
        ax.set_title("Count of " + col)
    plt.tight_layout()
    plt.savefig(FIG_DIR + "/categorical_counts.png")
    plt.close()

    # Calories burned
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Workout_Type", y=TARGET_COL, data=df)
    plt.title("Calories Burned Workout Type")
    plt.savefig(FIG_DIR + "/calories_by_workout_type.png")
    plt.close()

    print("charts saved in", FIG_DIR)


if __name__ == "__main__":
    run_eda()
