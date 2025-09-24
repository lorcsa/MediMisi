# backend/main.py
from __future__ import annotations

# --- Windows asyncio loop policy fix (subprocess/HTTP kompatibilitás) ---
import sys, asyncio
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass
# ------------------------------------------------------------------------

import os
import tempfile
import logging
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from backend.state import STATE
from backend.hakka import fetch_and_prepare_latest
from backend import rag  # FAISS-alapú RAG (backend/rag.py)

# --- Logging ---
logger = logging.getLogger("hakka")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# --- OpenAI init ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY hiányzik az .env-ből.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- FastAPI app ---
app = FastAPI(title="MediMISI – HAKKA (RAG backend)", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ----------------------------- Schemas -----------------------------
class ChatRequest(BaseModel):
    message: str

# ----------------------------- Helpers -----------------------------
def _ensure_client() -> OpenAI:
    if client is None:
        raise HTTPException(500, "OPENAI_API_KEY nincs beállítva az .env-ben.")
    return client

def _register_tables(prepared: List[tuple[str, "pandas.DataFrame"]]) -> List[str]:
    """
    A fetch_and_prepare_latest() [(period, df), ...] eredményét betölti STATE-be
    és ideiglenes DuckDB-be regisztrálja (csak /sql debughoz).
    """
    import pandas as pd  # csak típus miatt
    con = duckdb.connect(database=":memory:")
    names: List[str] = []
    for idx, (period, df) in enumerate(prepared, start=1):
        safe = period.replace(".", "").replace(" ", "")
        tname = f"t{idx}_{safe}"
        STATE.set_table(tname, df)
        con.register(tname, df)
        names.append(tname)
    STATE.set_meta(periods=[p for p, _ in prepared], tables=names)
    return names

# ----------------------------- Endpoints -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "tables": STATE.list_tables(), "meta": STATE.get_meta()}

@app.post("/refresh")
async def refresh() -> Dict[str, Any]:
    """
    Letöltés + beolvasás + normalizálás (4 CSV a legfrissebb sor 1,3,5,7 oszlopából),
    betöltés memóriába.
    """
    logger.info("REFRESH: start")
    try:
        prepared = await asyncio.to_thread(fetch_and_prepare_latest)  # SZINKRON függvény szálban
    except Exception as e:
        logger.exception("Refresh error")
        raise HTTPException(500, f"Letöltési/parse hiba: {type(e).__name__}: {e}")
    names = _register_tables(prepared)
    logger.info("REFRESH: done -> %d tables", len(names))
    return {"ok": True, "tables": names, "periods": STATE.get_meta().get("periods", [])}

@app.post("/rag/ingest")
def rag_ingest() -> Dict[str, Any]:
    """
    FAISS RAG index építése a memóriában lévő táblákból (STATE.tables) a backend/rag.py-vel.
    """
    if not STATE.tables:
        raise HTTPException(400, "Nincsenek betöltött táblák. Előbb hívd a /refresh-et.")
    try:
        idx = rag.build_or_load_index(STATE.tables)
        return {"ok": True, "index_ntotal": int(getattr(idx, "ntotal", 0))}
    except Exception as e:
        logger.exception("RAG ingest error")
        raise HTTPException(500, f"RAG index építés sikertelen: {type(e).__name__}: {e}")

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    """
    RAG-os kérdezés (NEM SQL):
      1) top-k keresés FAISS-szal (rag.search),
      2) kontextus összeállítása a talált 'text' mezőkből,
      3) válasz generálása OpenAI-val (Responses API).
    """
    cli = _ensure_client()

    # 1) Keresés a helyi FAISS indexben
    try:
        hits = rag.search(req.message, k=6)  # top-6 kontextus
    except Exception as e:
        raise HTTPException(500, f"RAG keresés hiba: {type(e).__name__}: {e}")
    contexts = [h["text"] for h in hits.get("hits", [])][:6]
    context_block = "\n\n".join(contexts) if contexts else "(nincs releváns kontextus)"

    # 2) Prompt összeállítás
    sys_prompt = (
        "Te egy adat-asszisztens vagy. Magyarul válaszolj tömören és logikusan. "
        "A válaszaidat a megadott KONEXTUSBAN szereplő információkra alapozd. "
        "Ha a kontextus nem elég, jelezd röviden, és kérj pontosítást."
    )
    user_prompt = f"KONEXTUS:\n{context_block}\n\nKÉRDÉS:\n{req.message}"

    # 3) Modellhívás (Responses API – egyszerű output_text kiolvasás)
    try:
        resp = cli.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = getattr(resp, "output_text", None)
        if not answer:
            # fallback – ha az SDK verzió más struktúrát adna
            try:
                chunks = []
                for item in getattr(resp, "output", []) or []:
                    if hasattr(item, "content"):
                        content = item.content
                        if isinstance(content, list):
                            for part in content:
                                if getattr(part, "type", "") == "output_text" and getattr(part, "text", None):
                                    chunks.append(part.text)
                        elif isinstance(content, str):
                            chunks.append(content)
                answer = "\n".join(chunks) if chunks else None
            except Exception:
                answer = None
        return {
            "answer": answer or "Nincs szöveges válasz.",
            "hits": hits.get("hits", []),  # opcionális: visszaadjuk a talált sorok meta adatait
        }
    except Exception as e:
        logger.exception("OpenAI hívás hiba")
        raise HTTPException(503, f"OpenAI hívás sikertelen: {type(e).__name__}: {e}")

# ----------------------------- (opcionális) SQL debug -----------------------------
@app.post("/sql")
def sql_debug(body: Dict[str, str] = Body(...)) -> Dict[str, Any]:
    """
    Fejlesztői/diagnosztikai célra: tetszőleges SQL futtatása a memóriában lévő táblákon (DuckDB).
    Nem része a chat/RAG folyamatnak.
    """
    sql = body.get("sql", "")
    con = duckdb.connect(database=":memory:")
    for name, df in STATE.tables.items():
        con.register(name, df)
    try:
        out = con.execute(sql).df()
        return {"rows": out.to_dict(orient="records"), "columns": list(out.columns)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/rag/diagnose")
def rag_diagnose() -> Dict[str, Any]:
    info: Dict[str, Any] = {"openai_key": bool(os.getenv("OPENAI_API_KEY"))}
    # fájlok állapota
    info["rag_files"] = {
        "index_exists": os.path.exists(rag.INDEX_PATH),
        "meta_exists": os.path.exists(rag.META_PATH),
        "stats": rag.index_stats()
    }
    # kis embedding próba
    try:
        cli = _ensure_client()
        emb = cli.embeddings.create(model="text-embedding-3-large", input=["ping"])
        info["embed_ok"] = (len(emb.data[0].embedding) == 3072)
    except Exception as e:
        info["embed_ok"] = False
        info["embed_error"] = f"{type(e).__name__}: {e}"
    return info
