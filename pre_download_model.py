from sentence_transformers import CrossEncoder
from core.config import RERANKER_MODEL_NAME

print(f"Downloading and caching model: {RERANKER_MODEL_NAME}")
try:
    model = CrossEncoder(RERANKER_MODEL_NAME)
    print("Model downloaded and cached successfully.")
except Exception as e:
    print(f"An error occurred while downloading the model: {e}")
