Try now: https://me-shubham-hf-test.hf.space/
# Recipe Finder
<img src="../image.png" alt="Alt text" width="400"/>
<!-- ![Alt text](image.png) -->

Project structure:
-
- `get_data.ipynb`: Fetches recipe data from upliance.ai. and saves it into json files:
    - `recipe_category.json`
    - `recipe_ingredients.json`
- `prepare_data.ipynb`: Prepares recipe data into tabular format. Creates document for each recipe. And, saves final table in `recipe.parquet`
- `vector_db.ipynb`: Create chromaDB vector store in `vector-store` folder and saves the recipe document embeddings in it.
- `app`: FastAPI webserver.

How to run: (**Be patient it might take time when you run it for the first time, as it downloads the model.**)
--
- Install python 3.10
- create virtual environment: `python -m venv venv`
- activate virtual environment:
    - windows: `source ./venv/Scripts/activate`
    - linux/mac: `source ./venv/bin/activate`
- `pip install -r requirements.txt`
- [optional]: run prepare data and vector store bash script.
    - Not required if `recipe.parquet` and `vector-store` already present.
    - `chmod +x script.sh`
    - `./script.sh`
- run FastAPI server: `uvicorn app.main:app --port 8080`
- visit http://127.0.0.1:8080/get-recipe-data

Resources Used:
--
- `app/response.html` for frontend was entirely generated via Claude and Mistral AI.
- Libraries used: pandas, numpy, pyarrow, requests, chromadb, sentence-transformers, torch, fastapi, jupyter, ipykernel


Scope of Improvement:
-
- Spelling mistakes in query might result not getting relevant results.
- This solution might not work well for hindi or Hinglish text, because the model used was trained on English language data.


<iframe
	src="https://me-shubham-hf-test.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
