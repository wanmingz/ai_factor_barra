# multi_asset/rag.py — local embedding RAG for macro / multi-asset news

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
    MULTI_ASSET_NEWS_INDEX_DIR,
    NEWS_LOOKBACK_DAYS,
    NEWS_RAG_TOP_K,
)
from multi_asset.ingest import fetch_asset_class_news
from multi_asset.universe import list_asset_classes
from news.rag import build_theme_index, embed_texts, retrieve


def _doc_text(doc: dict) -> str:
    parts = [doc.get("title", "")]
    if doc.get("proxy"):
        parts.append(f"({doc['proxy']})")
    if doc.get("asset_class"):
        parts.append(f"[{doc['asset_class']}]")
    return " ".join(p for p in parts if p)


def build_asset_class_index(
    class_docs: dict[str, list[dict]],
) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for ac, docs in class_docs.items():
        if not docs:
            index[ac] = {"docs": [], "embeddings": np.zeros((0, 384), dtype=np.float32)}
            continue
        texts = [_doc_text(d) for d in docs]
        index[ac] = {"docs": docs, "embeddings": embed_texts(texts)}
    return index


def asset_class_query(asset_class: str, rag_queries: dict[str, str]) -> str:
    return rag_queries.get(asset_class, f"{asset_class} asset class investment outlook")


def retrieve_all_asset_classes(
    index: dict[str, dict],
    asset_classes: list[str] | None = None,
    rag_queries: dict[str, str] | None = None,
    top_k: int = NEWS_RAG_TOP_K,
) -> dict[str, list[dict]]:
    asset_classes = asset_classes or sorted(index.keys())
    rag_queries = rag_queries or {}
    return {
        ac: retrieve(ac, asset_class_query(ac, rag_queries), index, top_k=top_k)
        for ac in asset_classes
    }


def save_index(
    index: dict[str, dict],
    as_of: str,
    out_dir: str | Path = MULTI_ASSET_NEWS_INDEX_DIR,
) -> Path:
    base = Path(out_dir) / as_of
    base.mkdir(parents=True, exist_ok=True)

    meta = {"as_of": as_of, "embedding_model": EMBEDDING_MODEL, "asset_classes": {}}
    for ac, entry in index.items():
        docs = entry["docs"]
        emb = entry["embeddings"]
        (base / f"{ac}.json").write_text(json.dumps(docs, indent=2))
        if len(emb):
            np.save(base / f"{ac}.npy", emb)
        meta["asset_classes"][ac] = {"n_docs": len(docs), "dim": int(emb.shape[1]) if len(emb) else 0}

    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    return base


def load_index(
    as_of: str,
    out_dir: str | Path = MULTI_ASSET_NEWS_INDEX_DIR,
) -> dict[str, dict]:
    base = Path(out_dir) / as_of
    if not base.is_dir():
        raise FileNotFoundError(f"No multi-asset news index at {base}")

    index: dict[str, dict] = {}
    for path in sorted(base.glob("*.json")):
        if path.name == "meta.json":
            continue
        ac = path.stem
        docs = json.loads(path.read_text())
        npy = base / f"{ac}.npy"
        embeddings = np.load(npy) if npy.exists() else np.zeros((0, 384), dtype=np.float32)
        index[ac] = {"docs": docs, "embeddings": embeddings}
    return index


def build_and_save_index(
    as_of: str | None = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> tuple[dict[str, dict], Path]:
    from datetime import date

    as_of_str = (as_of or date.today().isoformat()) if not isinstance(as_of, str) else as_of
    class_docs = fetch_asset_class_news(as_of=as_of_str, lookback_days=lookback_days)
    index = build_asset_class_index(class_docs)
    path = save_index(index, as_of_str)
    return index, path


if __name__ == "__main__":
    from datetime import date

    from multi_asset.universe import load_asset_classes, proxy_map

    as_of = date.today().isoformat()
    print(f"Building multi-asset RAG index for {as_of}...")
    index, path = build_and_save_index(as_of=as_of)
    print(f"Saved to {path}")

    queries = {k: v["rag_query"] for k, v in proxy_map().items()}
    hits = retrieve_all_asset_classes(index, asset_classes=list_asset_classes(), rag_queries=queries)
    for ac, docs in hits.items():
        if docs:
            print(f"\n{ac} — top {len(docs)} hits")
            for d in docs[:2]:
                sim = d.get("similarity", 0)
                print(f"  [{sim:.3f}] [{d.get('proxy', '?')}] {d['title'][:70]}")
