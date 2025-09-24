# backend/rag.py
from __future__ import annotations

import time
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import faiss  # pip install faiss-cpu

from openai import OpenAI
from openai import RateLimitError, APIError

# ---- Tartós tárolási helyek az indexnek és a metaadatnak ----
RAG_DIR = pathlib.Path("data") / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = RAG_DIR / "faiss.index"
META_PATH = RAG_DIR / "meta.parquet"

# ---- Globális progress (ha nincs callback, ide írunk) ----
# stage: 'start' | 'prepare_texts' | 'embedding' | 'indexing' | 'done' | 'error: ...'
RAG_PROGRESS: Dict[str, Any] = {"stage": "", "pct": 0, "rows": 0, "built": 0}


# ---- Konfiguráció az embedding/hívásokhoz ----
@dataclass
class RagConfig:
    # OpenAI embedding modell – text-embedding-3-large → 3072 dim
    model: str = "text-embedding-3-large"
    dim: int = 3072
    batch: int = 256
    max_retries: int = 6
    max_backoff_s: float = 8.0


def _mk_client() -> OpenAI:
    """OpenAI kliens inicializálása (OPENAI_API_KEY az env-ben)."""
    return OpenAI()


# --------------------------- EMBEDDING ---------------------------

def _embed_texts(texts: List[str], cfg: RagConfig, on_progress=None) -> np.ndarray:
    """
    OpenAI embeddings hívás batch-ekben; 429/5xx esetén exponenciális backoff.
    Visszatérés: (N, dim) float32 mátrix. A dimenziót ellenőrzi (3072).
    """
    client = _mk_client()
    vecs: List[List[float]] = []
    N = len(texts)
    B = cfg.batch
    done = 0

    for i in range(0, N, B):
        chunk = texts[i:i + B]
        backoff = 1.0
        for _attempt in range(cfg.max_retries):
            try:
                resp = client.embeddings.create(model=cfg.model, input=chunk)
                for d in resp.data:
                    v = d.embedding
                    if len(v) != cfg.dim:
                        raise ValueError(f"Embedding dim mismatch: {len(v)} != {cfg.dim}")
                    vecs.append(v)
                break  # sikeres batch
            except (RateLimitError, APIError):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, cfg.max_backoff_s)
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, cfg.max_backoff_s)
        else:
            raise RuntimeError("Embedding API többszöri próbálkozás után is sikertelen.")

        done += len(chunk)
        pct = int(done * 100 / max(1, N))
        if on_progress:
            on_progress("embedding", pct, total_rows=N, built_rows=done)
        else:
            RAG_PROGRESS.update({"stage": "embedding", "pct": pct, "rows": N, "built": done})

    return np.asarray(vecs, dtype="float32")


# ---------------------- SZÖVEG SOROK ÉPÍTÉSE --------------------

def build_rows_from_tables(
    tables: Dict[str, pd.DataFrame],
    on_progress=None
) -> Tuple[List[str], pd.DataFrame]:
    """
    A betöltött táblákból (STATE.tables) RAG dokumentum-sorokat készítünk:
    "Időszak, Megye, Település + facts" szöveg + meta (period, Megye, TelepulesNev, JOIN_KEY, table, text).
    """
    rows: List[Dict[str, Any]] = []
    tnames = list(tables.keys())
    total_rows = sum(len(df) for df in tables.values())
    seen = 0

    for tname in tnames:
        df = tables[tname]
        period = tname.split("_", 1)[-1] if "_" in tname else ""
        for _, r in df.iterrows():
            megye = str(r.get("Megye", "")).strip()
            telep = str(r.get("TelepulesNev", "")).strip()
            join = str(r.get("JOIN_KEY", "")).strip()

            facts = []
            for c, v in r.items():
                if c in ("Megye", "TelepulesNev", "JOIN_KEY"):
                    continue
                sv = str(v)
                if sv and sv != "nan":
                    facts.append(f"{c}: {sv}")
            # limitáljuk a chunk hosszát
            body = "; ".join(facts[:40])

            text = f"Időszak: {period}. Megye: {megye}. Település: {telep}. {body}"
            rows.append({
                "period": period, "Megye": megye, "TelepulesNev": telep,
                "JOIN_KEY": join, "table": tname, "text": text
            })

            seen += 1
            pct = int(seen * 100 / max(1, total_rows))
            if on_progress:
                on_progress("prepare_texts", pct, total_rows=total_rows, built_rows=seen)
            else:
                RAG_PROGRESS.update({"stage": "prepare_texts", "pct": pct, "rows": total_rows, "built": seen})

    meta_df = pd.DataFrame(rows)
    texts = meta_df["text"].tolist() if not meta_df.empty else []
    return texts, meta_df


# ----------------------- INDEX ÉPÍTÉS / TÖLTÉS -------------------

def build_or_load_index(
    tables: Dict[str, pd.DataFrame],
    cfg: RagConfig = RagConfig(),
    on_progress=None,
    force_rebuild: bool = False,
) -> faiss.IndexFlatIP:
    """
    FAISS index koszinusszal: IndexFlatIP + L2 normalizálás (adat + query).
    - force_rebuild=True esetén törli a régi index/meta fájlokat és újraépít.
    - ha meglévő indexet talál, CSAK akkor tölti be, ha az tényleg értelmes (ntotal>0 és meta_rows>0),
      különben újraépít.
    """
    # kényszerített törlés
    if force_rebuild:
        try:
            if INDEX_PATH.exists():
                INDEX_PATH.unlink()
            if META_PATH.exists():
                META_PATH.unlink()
        except Exception:
            pass

    # betöltés csak ha valid (nem üres)
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            idx = faiss.read_index(str(INDEX_PATH))
            ntotal = int(getattr(idx, "ntotal", 0))
            meta_ok = False
            meta_rows = 0
            if META_PATH.exists():
                try:
                    meta_rows = int(pd.read_parquet(META_PATH).shape[0])
                    meta_ok = meta_rows > 0
                except Exception:
                    meta_ok = False
            if ntotal > 0 and meta_ok:
                if on_progress:
                    on_progress("done", 100, total_rows=meta_rows, built_rows=ntotal)
                else:
                    RAG_PROGRESS.update({"stage": "done", "pct": 100, "rows": meta_rows, "built": ntotal})
                return idx
            # különben építünk
        except Exception:
            # sérült index → építünk
            pass

    # 1) szöveg-előkészítés
    texts, meta_df = build_rows_from_tables(tables, on_progress=on_progress)
    if not texts:
        raise RuntimeError("Nincs betöltött tábla a RAG index építéséhez.")

    # 2) embedding
    X = _embed_texts(texts, cfg, on_progress=on_progress)
    if X.shape[1] != cfg.dim:
        raise ValueError(f"Index dim mismatch: {X.shape[1]} != {cfg.dim}")

    # 3) indexelés
    if on_progress:
        on_progress("indexing", 95, total_rows=len(texts), built_rows=len(texts))
    else:
        RAG_PROGRESS.update({"stage": "indexing", "pct": 95, "rows": len(texts), "built": len(texts)})

    faiss.normalize_L2(X)                  # koszinusz normalizálás
    index = faiss.IndexFlatIP(cfg.dim)     # belső szorzat → koszinusz L2 norm után
    index.add(X)

    faiss.write_index(index, str(INDEX_PATH))
    meta_df.to_parquet(META_PATH, index=False)

    if on_progress:
        on_progress("done", 100, total_rows=len(texts), built_rows=len(texts))
    else:
        RAG_PROGRESS.update({"stage": "done", "pct": 100, "rows": len(texts), "built": len(texts)})
    return index


# ----------------------- GYORSDIAGNOSZTIKA -----------------------

def index_stats() -> Dict[str, int]:
    """Hány sor van a meta-ban és hány vektor az indexben (ntotal)."""
    out = {"meta_rows": 0, "index_rows": 0}
    try:
        if META_PATH.exists():
            out["meta_rows"] = int(pd.read_parquet(META_PATH).shape[0])
        if INDEX_PATH.exists():
            idx = faiss.read_index(str(INDEX_PATH))
            out["index_rows"] = int(getattr(idx, "ntotal", 0))
    except Exception:
        pass
    return out


# ----------------------------- KERESÉS ----------------------------

def search(query: str, k: int = 8, cfg: RagConfig = RagConfig()) -> Dict[str, Any]:
    """
    Keresés a helyi FAISS indexben koszinusszal. Visszatérés:
      { "query": str, "k": int, "hits": [ {meta..., "score": float}, ... ] }
    """
    if not (INDEX_PATH.exists() and META_PATH.exists()):
        raise RuntimeError("RAG index hiányzik. /rag/ingest (vagy /refresh után auto build) szükséges.")

    meta_df = pd.read_parquet(META_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    qv = _embed_texts([query], cfg).astype("float32")
    faiss.normalize_L2(qv)

    D, I = index.search(qv, k)
    hits: List[Dict[str, Any]] = []
    scores = D[0].tolist() if len(D) else []
    idxs = I[0].tolist() if len(I) else []

    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue
        row = meta_df.iloc[idx].to_dict()
        row["score"] = float(score)
        hits.append(row)

    return {"query": query, "k": k, "hits": hits}
