import os
import eda
import train_regression
import train_clustering

DATA_PATH = "raw_data/Fitbit_dataset.csv"


def main():
    print("STEP 0: Checking dataset")
    if not os.path.exists(DATA_PATH):
        print("ERROR: Data not found", DATA_PATH)
        return
    print("Dataset found. Continuing...")

    print("=" * 50)
    print("STEP 1: Exploratory Data Analysis")
    print("=" * 50)
    eda.run_eda()

    print("\n" + "=" * 50)
    print("STEP 2: Regression - Predicting Calories Burned")
    print("=" * 50)
    train_regression.main()

    print("\n" + "=" * 50)
    print("STEP 3: Clustering - Workout Patterns")
    print("=" * 50)
    train_clustering.main()

    print("all done Check the 'reports' and 'models' folders for output")


if __name__ == "__main__":
    main()
