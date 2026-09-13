import pandas as pd


TRAIN_CSV = "data/raw/train.csv"


df = pd.read_csv(TRAIN_CSV)

print("Original shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget names:")
print(df["target_name"].unique())


# Convert long format → wide format

wide_df = (
    df.pivot_table(
        index=[
            "sample_id",
            "image_path",
            "Sampling_Date",
            "State",
            "Species",
            "Pre_GSHH_NDVI",
            "Height_Ave_cm"
        ],
        columns="target_name",
        values="target",
        aggfunc="first"
    )
    .reset_index()
)


# Remove the columns index name
wide_df.columns.name = None


print("\nConverted shape:")
print(wide_df.shape)

print("\nConverted columns:")
print(wide_df.columns.tolist())


print("\nFirst rows:")
print(wide_df.head())


# Save
wide_df.to_csv(
    "data/processed/train_wide.csv",
    index=False
)

print("\nSaved:")
print("data/processed/train_wide.csv")