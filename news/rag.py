# news/rag.py — local embedding RAG (sentence-transformers + numpy cosine similarity)

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from config import (
    EMBEDDING_MODEL,
    NEWS_INDEX_DIR,
    NEWS_RAG_TOP_K,
    NEWS_LOOKBACK_DAYS,
)
from news.ingest import fetch_theme_news

_MODEL = None


def _get_embedder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def _doc_text(doc: dict) -> str:
    """Text passed to the embedder."""
    parts = [doc.get("title", "")]
    if doc.get("ticker"):
        parts.append(f"({doc['ticker']})")
    if doc.get("theme"):
        parts.append(f"[{doc['theme']}]")
    return " ".join(p for p in parts if p)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def build_theme_index(
    theme_docs: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Embed all documents per theme.

    Returns {theme: {"docs": [...], "embeddings": ndarray}}.
    """
    index: dict[str, dict] = {}
    for theme, docs in theme_docs.items():
        if not docs:
            index[theme] = {"docs": [], "embeddings": np.zeros((0, 384), dtype=np.float32)}
            continue
        texts = [_doc_text(d) for d in docs]
        index[theme] = {"docs": docs, "embeddings": embed_texts(texts)}
    return index


def retrieve(
    theme: str,
    query: str,
    index: dict[str, dict],
    top_k: int = NEWS_RAG_TOP_K,
) -> list[dict]:
    """Return top-k news docs for a theme by cosine similarity (embeddings normalized)."""
    entry = index.get(theme, {})
    docs = entry.get("docs", [])
    embeddings = entry.get("embeddings")
    if not docs or embeddings is None or len(docs) == 0:
        return []

    q = embed_texts([query])[0]
    scores = embeddings @ q
    k = min(top_k, len(docs))
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for i in top_idx:
        hit = dict(docs[i])
        hit["similarity"] = float(scores[i])
        results.append(hit)
    return results


def theme_query(theme: str) -> str:
    return f"{theme} sector investment outlook stock market news"


def retrieve_all_themes(
    index: dict[str, dict],
    themes: list[str] | None = None,
    top_k: int = NEWS_RAG_TOP_K,
) -> dict[str, list[dict]]:
    themes = themes or sorted(index.keys())
    return {t: retrieve(t, theme_query(t), index, top_k=top_k) for t in themes}


def save_index(index: dict[str, dict], as_of: str, out_dir: str | Path = NEWS_INDEX_DIR) -> Path:
    """Persist docs + embeddings for an as-of date."""
    base = Path(out_dir) / as_of
    base.mkdir(parents=True, exist_ok=True)

    meta = {"as_of": as_of, "embedding_model": EMBEDDING_MODEL, "themes": {}}
    for theme, entry in index.items():
        docs = entry["docs"]
        emb = entry["embeddings"]
        (base / f"{theme}.json").write_text(json.dumps(docs, indent=2))
        if len(emb):
            np.save(base / f"{theme}.npy", emb)
        meta["themes"][theme] = {"n_docs": len(docs), "dim": int(emb.shape[1]) if len(emb) else 0}

    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    return base


def load_index(as_of: str, out_dir: str | Path = NEWS_INDEX_DIR) -> dict[str, dict]:
    base = Path(out_dir) / as_of
    if not base.is_dir():
        raise FileNotFoundError(f"No news index at {base}")

    index: dict[str, dict] = {}
    for path in sorted(base.glob("*.json")):
        if path.name == "meta.json":
            continue
        theme = path.stem
        docs = json.loads(path.read_text())
        npy = base / f"{theme}.npy"
        embeddings = np.load(npy) if npy.exists() else np.zeros((0, 384), dtype=np.float32)
        index[theme] = {"docs": docs, "embeddings": embeddings}
    return index


def build_and_save_index(
    as_of: str | None = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> tuple[dict[str, dict], Path]:
    """End-to-end: ingest news → embed → save index."""
    from datetime import date

    as_of_str = (as_of or date.today().isoformat()) if not isinstance(as_of, str) else as_of
    theme_docs = fetch_theme_news(as_of=as_of_str, lookback_days=lookback_days)
    index = build_theme_index(theme_docs)
    path = save_index(index, as_of_str)
    return index, path


if __name__ == "__main__":
    from datetime import date

    from theme_agent import list_themes

    as_of = date.today().isoformat()
    print(f"Building local RAG index for {as_of}...")
    index, path = build_and_save_index(as_of=as_of)
    print(f"Saved to {path}")

    hits = retrieve_all_themes(index, themes=list_themes())
    for theme, docs in hits.items():
        print(f"\n{theme} — top {len(docs)} hits")
        for d in docs:
            sim = d.get("similarity", 0)
            print(f"  [{sim:.3f}] [{d['ticker']}] {d['title'][:70]}")
