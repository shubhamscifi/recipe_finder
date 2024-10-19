from pathlib import Path
import torch
import pandas as pd
import os

ROOT_DIR = Path(__file__).absolute().parent

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

retrieval_model_id = os.getenv("RETRIEVAL_MODEL_ID" ,"msmarco-distilbert-base-v4")
reranker_model_id = os.getenv("RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

retrieval_top_k = 50
reranker_top_k = 5

retrieval_model_cache_path = ROOT_DIR / 'model'

vector_db_path = str(ROOT_DIR / "vector-store")
recipe_collection = "recipe_collection"

recipe_file = str(ROOT_DIR / "recipe.parquet")

