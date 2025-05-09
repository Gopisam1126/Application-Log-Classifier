## Overview

This repository implements a modular log‐classification service using FastAPI. Uploaded CSV files with `source` and `log_message` columns are processed through three classifiers—regex rules, a BERT embedding model, and an LLM—to produce a `predicted_label` for each entry. The service exposes a single `/classify/` endpoint that returns a new CSV with the added classification column.

## Features

- **Regex Classifier**  
  Fast pattern‐matching for common error formats using predefined regular expressions.

- **BERT Embedding Classifier**  
  Leverages a SentenceTransformer (`all-MiniLM-L6-v2`) and a scikit‐learn model for nuanced labeling when regex yields “Other.”

- **LLM Classifier**  
  Routes logs from designated sources (e.g., `LegacyCRM`) through an LLM for contexts where rules or embeddings underperform.

- **FastAPI Endpoint**  
  - **Route**: `POST /classify/`  
  - **Validates**: CSV format and presence of `source` & `log_message` columns  
  - **Pipeline**: Regex → BERT → LLM  
  - **Response**: Downloadable CSV (`classified.csv`) with `predicted_label`

- **Concurrency-Safe Output**  
  Saves results under `resources/` with unique filenames to avoid collisions.

## Requirements

- Python 3.8+  
- FastAPI  
- Uvicorn (or other ASGI server)  
- pandas  
- sentence-transformers  
- scikit-learn  
- joblib  
- Pretrained models:  
  - `model/log_model.joblib` 
