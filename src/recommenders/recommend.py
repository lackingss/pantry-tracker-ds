# Inference script: load saved TF-IDF artifacts, score a pantry against
# all recipes using cosine similarity, return top N matches.

import pickle
import scipy.sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


MODEL_DIR = "models"

def load_artifacts():
    """Load the three saved files from models/. Called once at startup."""
    
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = scipy.sparse.load_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"))

    with open(os.path.join(MODEL_DIR, "recipe_index.pkl"), "rb") as f:
        recipe_index = pickle.load(f)

    return vectorizer, tfidf_matrix, recipe_index


def recommend(pantry_items: list[str], vectorizer, tfidf_matrix, recipe_index, top_n: int = 10):
    """
    Given a list of pantry ingredient strings, return the top N recipes.

    Args:
        pantry_items:  e.g. ["chicken", "garlic", "lemon juice"]
        vectorizer:    fitted TfidfVectorizer loaded from pickle
        tfidf_matrix:  sparse matrix (n_recipes × vocab) loaded from .npz
        recipe_index:  DataFrame with columns [recipe_id, title]
        top_n:         how many results to return

    Returns:
        List of dicts: [{"recipe_id": ..., "title": ..., "score": ...}, ...]
    """

    #join pantry items into a single string, same format as ner_string
    pantry_string = " ".join(pantry_items)

    # transform the pantry string into the TF-IDF feature space
    pantry_vector = vectorizer.transform([pantry_string])  # returns a (1 × vocab) sparse matrix

  
    # cosine_similarity returns a (1 × n_recipes) array — flatten it to 1D
    scores = cosine_similarity(pantry_vector, tfidf_matrix).flatten()  # shape (n_recipes,)

    # get the indices of the top N scores (highest first)
    top_indices = np.argsort(scores)[::-1][:top_n]  # argsort ascending, reverse, slice

    results = []
    for idx in top_indices:
        row = recipe_index.iloc[idx]
        results.append({
            "recipe_id": int(row["recipe_id"]),
            "title": row["title"],
            "score": float(scores[idx])  # cast numpy float → Python float for JSON safety
        })

    return results
