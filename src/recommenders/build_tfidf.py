# This script builds and saves the TF-IDF model for recipe recommendation.

import sqlite3       # 
import pandas as pd  # 
import json          # 
from sklearn.feature_extraction.text import TfidfVectorizer  # 
import scipy.sparse  # 
import pickle        # 
import os            # 

# PATHS
DB_PATH = "data/processed/pantry_tracker.db"  # same database that holds products and recipes
MODEL_DIR = "models"  # where i am saving the TF-IDF model

# Load recipes from the database

conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("SELECT recipe_id, title, ner FROM recipes", conn)

conn.close()
print(f"Loaded {len(df):,} recipes")

# Parse ner column to extract ingredient names and create a new column with the parsed ingredient strings.
# The TF-IDF vectorizer will be trained on this new column. Also drop any rows where parsing fails or results in an empty string, since those won't be useful for the model. 
# This step is crucial because the quality of the TF-IDF model depends heavily on having clean and meaningful ingredient data to work with.

def ner_to_string(ner_json):
    try:
        ingredients = json.loads(ner_json)  # parse the JSON string into a Python list
        return " ".join(ingredients)  # join the list into a space-separated string
    except json.JSONDecodeError:
        return None  # return None if parsing fails

df["ner_string"] = df["ner"].apply(ner_to_string)

# drop rows where parsing failed
df = df.dropna(subset=["ner_string"])
df = df[df["ner_string"].str.strip() != ""]

print(f"Recipes after parsing: {len(df):,}")

# Fit the TF-IDF vectorizer on the ner_string column.
# This will create a sparse matrix where each row corresponds to a recipe and each column corresponds to an ingredient (or ingredient combination) in the vocabulary.
# The values in the matrix are the TF-IDF scores, which reflect how important each ingredient is to each recipe relative to the entire corpus of recipes.

vectorizer = TfidfVectorizer(
    min_df=2, # ignore ingredients that are in fewer than 2 recipes (probably something niche or a typo)
    max_df=0.95, # ignore ingredients that are in more than 95% of recipes (probably something generic like "salt" or "water")
    ngram_range=(1, 2) # considering both single and double word combinations
)



tfidf_matrix = vectorizer.fit_transform(df["ner_string"]) # call fit_transform on the ner_string column
               # this returns a sparse matrix where each row corresponds to a recipe and each column corresponds to an ingredient (or ingredient combination) in the vocabulary. The values are the TF-IDF scores.

print(f"Matrix shape: {tfidf_matrix.shape}") # the shape of the matrix is (number of recipes, number of unique ingredients in the vocabulary).
print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}") # the size of the vocabulary (number of unique ingredients)

# Recipe Index: I need to keep track of which row in the TF-IDF matrix corresponds to which recipe_id and title, so I create a simple DataFrame that maps from the row index back to the recipe_id and title. 
# This will be useful later when I want to retrieve the original recipe information based on the TF-IDF scores
recipe_index = df[["recipe_id", "title"]].reset_index(drop=True) # this is a simple DataFrame that maps from the row index of the TF-IDF matrix back to the recipe_id and title


# Save the model files

os.makedirs(MODEL_DIR, exist_ok=True) # create the models directory if it doesn't exist

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f: # save the fitted TF-IDF vectorizer to a pickle file so that it can be loaded later when I want to transform new ingredient lists into the same feature space.
    pickle.dump(vectorizer, f)

scipy.sparse.save_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"), tfidf_matrix)

with open(os.path.join(MODEL_DIR, "recipe_index.pkl"), "wb") as f:
    pickle.dump(recipe_index, f)
