# This script reads the RecipeNLG dataset, cleans it, and loads it into
# the recipes table in our SQLite database.

import pandas as pd # reading adn cleaning the CSV 
import sqlite3 # connecting to the database
import ast # converting string lists to real Python Lists
import json # for saving lists as JSON strings in the database
import os # for building file paths

# PATHS
# Where the raw datta lives
RAW_PATH = "data/raw/RecipeNLG_dataset.csv"

# Where our SQlite database lives (same one as the product table)
DB_PATH = "data/processed/pantry_tracker.db"

# STEP 1: LOAD
print("Loading raw data...")

#read the CSV in chunks of 100,000 rows at a time
chunk_size = 100_000
chunks = pd.read_csv(RAW_PATH, chunksize=chunk_size)

# STEP 2: connect to the database
print("Connecting to database...")
# Connect to the ame SQLite database that holds products table
conn = sqlite3.connect(DB_PATH)

# STEP 3: Create table
#create recipe table if it doesn't exist yet
conn.execute("""
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id INTEGER PRIMARY KEY,
    title TEXT,
    ingredients TEXT,
    directions TEXT,
    ner TEXT
)
""")
conn.commit() # commit the table creation to the database

# STEP 4: Clean each chunk and load it into the database
total_loaded = 0 #keep track of how many recipes we have loaded so far

for i, chunk in enumerate(chunks):
    print(f"processing chunk {i+1} ({total_loaded} recipes loaded so far)...")
    
    #drop columns we dont need
    chunk = chunk.drop(columns=["link", "source"])

    #rename the unamed index column to recipe_id 
    chunk = chunk.rename(columns={"Unnamed: 0": "recipe_id", "NER": "ner"})
    
    #drop any rows where title, ingredients, or NER is missing
    chunk = chunk.dropna(subset=["title", "ingredients", "ner"])

    # The ingredients, directions, and NER columns are stored as strings
    # that look like Python lists e.g. "['butter', 'milk']"
    # ast.literal_eval converts them to real Python lists
    # then json.dumns converts them to proper JSON strings for storage
    def safe_convert(val):
        # try to convert the string to a list, return None if it fails
        try:
            return json.dumps(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            return None
    
    chunk["ingredients"] = chunk["ingredients"].apply(safe_convert)
    chunk["directions"] = chunk["directions"].apply(safe_convert)
    chunk["ner"] = chunk["ner"].apply(safe_convert)

    #drop any rows where conversion failed
    chunk = chunk.dropna(subset=["ingredients", "directions", "ner"])

    # Load this chunk into the database
    # if_exists="append" menas we add to the table without overwriting to it
    chunk[["recipe_id", "title", "ingredients", "directions", "ner"]].to_sql("recipes", conn, if_exists="append", index=False)
    
    total_loaded += len(chunk) # update our count of how many recipes we have loaded


# STEP 5: ADD INDEX
print ("creating index...")
conn.execute("CREATE INDEX IF NOT EXISTS idx_recipe_id ON recipes (recipe_id)")
conn.commit() # commit the index creation to the database

#STEP 6: close connection
conn.close()
print(f"Done! Total recipes loaded: {total_loaded:,}")




