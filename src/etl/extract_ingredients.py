# This script extracts the ingredients text for each product from the raw Parquet file and merges it into the existing products table in the SQLite database.
# pyarrow reads parquet files efficiently — we use it here instead of pandas
# because it supports batch/streaming reads which prevents memory crashes when processing large files.      

import pyarrow.parquet as pq
import pandas as pd
import sqlite3
import re

def strip_html(text):
    # removes anything between < and > tags like <span class="allergen">
    if text is None or not isinstance(text, str):
        return None
    return re.sub(r'<[^>]+>', '', text).strip()


def extract_ingredients_text(row):
    """
    Each product has a list of ingredient entries like:
    [{'lang': 'main', 'text': 'Sugar, water...'}, {'lang': 'en', 'text': 'Sugar, water...'}]
    
    We want the 'en' entry first, fall back to 'main' if no english version exists.
    Returns None if neither exists.
    """
    # if the list is empty, return None bcuz there is nothing to extract
    if row is None or len(row) == 0:
        return None
    # placeholders, will fill in when looping through the entries
    en_text = None
    main_text = None
    # each entry in the list is a dict with 'lang' and 'text' keys
    # loop through all entries and grab the english or main version
    for entry in row:
        if entry['lang'] == 'en':
            en_text = entry['text']
        elif entry['lang'] == 'main':
            main_text = entry['text']
    # return english text if it exists, otherwise fall back to main 
    # if neither exists return None
    return en_text if en_text else main_text


# load in batches to avoid memory crash, and how many rows to process at a time
BATCH_SIZE = 100_000
parquet_file = pq.ParquetFile("data/raw/food.parquet") # open the parquet file for reading, this does not load it into memory yet

results = [] # empty list, collect each batch's results here and combine at the end

batch_num = 0

# iter_batches streams through the file in chunks of BATCH_SIZE rows
# columns= means we only load code and ingredients_text, nothing else
for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["code", "ingredients_text"]):
    batch_num += 1
    print(f"Processing batch {batch_num}...")
    
    df_batch = batch.to_pandas() # convert the pyarrow batch to a pandas dataframe so we can work with it normally

    # apply our function to every row in the ingredients_text column
    # this extracts the english/main text from the nested list structure
    df_batch["ingredients_text_clean"] = df_batch["ingredients_text"].apply(extract_ingredients_text)

    # strip any HTML tags leftover from OpenFoodFacts contributors pasting formatted text
    df_batch["ingredients_text_clean"] = df_batch["ingredients_text_clean"].apply(strip_html)
    
    # drop rows where we couldn't extract any text — they're useless
    # only keep the two columns we need going forward
    df_batch = df_batch[df_batch["ingredients_text_clean"].notna()][["code", "ingredients_text_clean"]]
    results.append(df_batch) # add this batch's results to my collection

# stack all the batch dataframes into one single dataframe
# ignore_index=True resets row numbers cleanly from 0
print("Combining batches...")
df_ingredients = pd.concat(results, ignore_index=True)
print(f"Products with ingredients text: {df_ingredients.shape[0]:,}")

# open a connection to the existing SQLite database
conn = sqlite3.connect("data/processed/pantry_tracker.db")

# load existing products
df_products = pd.read_sql("SELECT * FROM products", conn)

# drop old ingredients column if it exists so we can replace it cleanly
if "ingredients_text_clean" in df_products.columns:
    df_products = df_products.drop(columns=["ingredients_text_clean"])

# deduplicate ingredients — keep first occurrence of each barcode
df_ingredients = df_ingredients.drop_duplicates(subset="code", keep="first")

print(f"Products in DB before merge: {df_products.shape[0]:,}")

# join the ingredients text onto the products table using 'code' (barcode) as the key
# how="left" means keep all products even if they don't have ingredients text
df_merged = df_products.merge(df_ingredients, on="code", how="left")
print(f"Products after merge: {df_merged.shape[0]:,}")
print(f"Products with ingredients: {df_merged['ingredients_text_clean'].notna().sum():,}")

# overwrite the products table in SQLite with the new merged version
df_merged.to_sql("products", conn, if_exists="replace", index=False)
conn.close() # always close !
print("Done. Products table updated with ingredients text.")