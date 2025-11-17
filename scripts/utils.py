import os, json
from pathlib import Path
import yaml
from typing import Union
import torch
from sentence_transformers import SentenceTransformer


def read_config(config_path: Union[str, Path] = "scripts/config.yaml") -> dict:
    """
    Load a YAML configuration file safely, compatible with all Python versions.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# keys utility
def _hit_key(hit: dict) -> str:
    """
    Compute a stable key for a retrieval hit (used for deduplication).

    Each hit is a dict that typically includes keys like:
      - "source" or "dataset"
      - "id" or "doc_id"
      - "text" or "content"
    The function builds a unique string key from those.
    """
    if not isinstance(hit, dict):
        return str(hit)

    source = str(hit.get("source") or hit.get("dataset") or "unknown")
    hid = str(hit.get("id") or hit.get("doc_id") or hit.get("uuid") or "")
    text = str(hit.get("text") or hit.get("content") or "")[:120]  # short snippet for uniqueness
    return f"{source}:{hid}:{text}"


def resolve_project_path(p: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the project root (rag_app/), regardless of CWD.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[1]  # rag_app/
    return (project_root / p).resolve()


def load_embedder(model_dir: Union[str, Path]) -> SentenceTransformer:
    """
    Load a local SentenceTransformer model from a directory.
    Enforces offline mode and validates local artifacts.
    """
    # Enforce offline / no-proxy
    for key in [
        "HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy",
        "ALL_PROXY","all_proxy","NO_PROXY","no_proxy"
    ]:
        os.environ.pop(key, None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    p = resolve_project_path(model_dir)

    if not p.exists():
        raise FileNotFoundError(
            f"Embedding model path does not exist: {p}\n"
            "Check config.embedding.model_name and your project layout."
        )

    modules_file = p / "modules.json"
    if not modules_file.exists():
        raise RuntimeError(
            f"Local SentenceTransformer seems incomplete at: {p}\n"
            f"Missing file: {modules_file}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(p), device=device)
    # Prevent remote card fetch (sometimes triggers network)
    model._model_card = None
    return model