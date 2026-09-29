"""Build (or refresh) the vector index for the current mode and print a corpus summary.

    python scripts/build_index.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pipeline = RAGPipeline(get_settings())
    for key, value in pipeline.describe().items():
        print(f"{key:<20} {value}")
