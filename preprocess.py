import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", default="/opt/ml/processing/input/application_train.csv"
    )
    parser.add_argument("--train-output", default="/opt/ml/processing/train/train.csv")
    parser.add_argument(
        "--val-output", default="/opt/ml/processing/validation/validation.csv"
    )
    parser.add_argument("--test-output", default="/opt/ml/processing/test/test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    df = df.drop(columns=["SK_ID_CURR"])
    target = df.pop("TARGET")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("Missing")

    df = pd.get_dummies(df, columns=categorical_cols)
    df.insert(0, "TARGET", target)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["TARGET"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["TARGET"], random_state=42
    )

    train_df.to_csv(args.train_output, header=False, index=False)
    val_df.to_csv(args.val_output, header=False, index=False)
    test_df.to_csv(args.test_output, header=False, index=False)

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
