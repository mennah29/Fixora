"""Build or refresh the Chroma index before deploying the API.

Run once before starting the API service:
    python scripts/build_index.py

Or via Docker:
    docker compose run --rm -e AUTO_INDEX=true --entrypoint python fixora-api scripts/build_index.py
"""

import sys
from pathlib import Path

# Support the documented `python scripts/build_index.py` command by placing the
# project root on the import path before importing the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.service import RagService


if __name__ == "__main__":
    # The build operation may run before the embedding model is cached; it is
    # the one intentional path that can download the configured model.
    settings = get_settings().model_copy(update={"auto_index": True, "embedding_local_files_only": False})
    service = RagService(settings)
    service.load()
    if not service.ready:
        raise SystemExit(service.startup_error or "Index build failed.")
    print(f"Index ready: {service.document_count} documents at {settings.chroma_path}")
