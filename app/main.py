from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import constants
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import chromadb
import pandas as pd
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retrieval_model = SentenceTransformer(model_name_or_path=constants.retrieval_model_id, 
                                      cache_folder=constants.retrieval_model_cache_path,
                                      device=constants.device)

reranker_model = CrossEncoder(model_name=constants.reranker_model_id,
                              default_activation_function=torch.nn.Sigmoid(),
                              device=constants.device)

chroma_client = chromadb.PersistentClient(path=constants.vector_db_path)
# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(
    name=constants.recipe_collection, 
    metadata={"hnsw:space": "cosine"}   # use cosine similarity
    )

recipe_df = pd.read_parquet(constants.recipe_file)


@app.post("/get-recipe")
async def get_recipe(query:str = Body(embed=True)):

    query_embeddings: list[list[float]] = retrieval_model.encode([query])
    # retrieve to 50 recipes
    results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=constants.retrieval_top_k,
            # include=['distances'],
        )
    
    top_recipe_ids: list[str] = results['ids'][0]

    top_recipe_df = recipe_df.loc[lambda df_: df_['id'].isin(top_recipe_ids),:]
    # print(top_recipe_df)

    ranks = reranker_model.rank(query,
                                top_recipe_df['document'].to_list(), 
                                # return_documents=True, 
                                top_k=constants.reranker_top_k,
                                # convert_to_numpy=False
                                )

    ranks_df = pd.DataFrame(ranks).loc[lambda df_: df_['score'] >= 0.1,:]

    top_recipe_ranked_df = (
        top_recipe_df
        .iloc
        [ranks_df['corpus_id'].to_list(),]
        .assign(score=ranks_df['score'].to_list())
    )
    # top_recipe_ranked_df['score'] = ranks_df['score'].to_list()
    # print(top_recipe_ranked_df)

    return json.loads(top_recipe_ranked_df.to_json(orient='records'))


@app.get("/get-recipe-data")
async def get_recipe_html():
    return FileResponse("./app/response.html")