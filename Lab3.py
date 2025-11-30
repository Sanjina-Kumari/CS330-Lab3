import argparse
import pandas as pd
import numpy as np

# --------------------------------------------------
# Sequential 80/20 split
# --------------------------------------------------
def splitData(df):
    n = int(0.8 * len(df))
    train = df.iloc[:n]
    test = df.iloc[n:]
    return train, test

# --------------------------------------------------
# Random 80/20 split
# --------------------------------------------------
def splitDataRandom(df):
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = int(0.8 * len(df_shuffled))
    train = df_shuffled.iloc[:n]
    test = df_shuffled.iloc[n:]
    return train, test

# --------------------------------------------------
# Optional: 70/15/15 split (Part 6)
# --------------------------------------------------
def splitThreeWays(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(0.70 * len(df))
    n_val = int(0.85 * len(df))

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_val]
    test = df.iloc[n_val:]
    return train, val, test

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to dataset file")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    print("Loaded dataset:", args.file)
    print(df.head())

    # 80/20 sequential
    train_seq, test_seq = splitData(df)
    print("\nSequential Split:")
    print("Train size:", len(train_seq), "Test size:", len(test_seq))

    # 80/20 random
    train_rand, test_rand = splitDataRandom(df)
    print("\nRandom Split:")
    print("Train size:", len(train_rand), "Test size:", len(test_rand))

    # 70/15/15 (extra)
    train3, val3, test3 = splitThreeWays(df)
    print("\nThree-way Split (70/15/15):")
    print("Train:", len(train3), "Val:", len(val3), "Test:", len(test3))
