"""Pre-download the BGE cross-encoder so the first request (or MOCK mode) can use it offline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from app.config import get_settings  # noqa: E402

if __name__ == "__main__":
    name = get_settings().reranker_model
    print(f"Downloading {name} ...")
    AutoTokenizer.from_pretrained(name)
    AutoModelForSequenceClassification.from_pretrained(name)
    print("Done.")
