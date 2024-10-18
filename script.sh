#!/bin/bash

# source ./venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace get_data.ipynb
jupyter nbconvert --to notebook --execute --inplace prepare_data.ipynb
jupyter nbconvert --to notebook --execute --inplace vector_db.ipynb
