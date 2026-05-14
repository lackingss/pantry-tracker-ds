import pandas as pd
import sqlite3


# columns to load from parquet
columns_we_need = [
    "code",
    "brands",
    "categories",
    "lang",
    "nutriscore_grade",
    "nova_group",
    "completeness",
    "no_nutrition_data",
]

# load parquet with only needed columns
df = pd.read_parquet(
    "data/raw/food.parquet",
    columns=columns_we_need
)

print(f"Before filtering: {df.shape[0]:,} rows, {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# filter 1: 
df = df[df["lang"] == "en"] # only keep products with English language, inner computes True for rows where lang is "en" and False otherwise, and then we keep only the True rows
print(f"After filter 1: {df.shape[0]:,} rows")

# filter 2: 
df = df[df["completeness"] >= 0.4] # only keep products with at least 40% of the fields filled in, inner computes True for rows where completeness is >= 0.4 and False otherwise, and then we keep only the True rows
print(f"After filter 2: {df.shape[0]:,} rows")

# filter 3: 
df = df[df["no_nutrition_data"] != True] # only keep products without missing nutrition data, inner computes True for rows where no_nutrition_data is not True and False otherwise, and then we keep only the True rows
print(f"After filter 3: {df.shape[0]:,} rows")

# filter 4: 
df = df[df["code"].notna()] # only keep products with a valid bar code
print(f"After filter 4: {df.shape[0]:,} rows")

# fix string "null" values
df["categories"] = df["categories"].replace("null", pd.NA) # replace string "null" with actual NA values, so that they are treated as missing values in pandas and can be handled accordingly (e.g., dropped or filled)

# extract first category only
df["category"] = df["categories"].str.split(",").str[0] # split the categories string by comma and take the first element as the main category, this creates a new column "category" with just the first category for each product

# drop columns you no longer need
df = df.drop(columns=["lang", "no_nutrition_data", "categories"]) 

# drop rows where category is null
df = df[df["category"].notna()]

# reset index
df = df.reset_index(drop=True)

# summary
print(f"Rows: {df.shape[0]:,}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(df.head(10))
print(df.dtypes)
print(df.isnull().sum())

# Save to sqlite
conn = sqlite3.connect("data/processed/pantry_tracker.db")
df.to_sql("products", conn, if_exists="replace", index=False)
conn.close()
print("Data saved to SQLite database.")