import os
import json
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="MediMISI – HAKKA CSV Chat", layout="wide")
st.title("Törvény MISI – HAKKA CSV / Államkincstár Chat")

# ----------------- ACTIONS -----------------

def call_refresh():
    with st.spinner("Letöltés és feldolgozás..."):
        r = requests.post(f"{BACKEND}/refresh", timeout=300)
        r.raise_for_status()
        st.session_state["meta"] = r.json()
        st.success("Siker! Adatok betöltve.")

def call_rag_load():
    with st.spinner("RAG betöltés..."):
        r = requests.post(f"{BACKEND}/rag/ingest", timeout=300)
        r.raise_for_status()
        st.session_state["rag"] = r.json()
        st.success("RAG készen áll.")

def call_health():
    try:
        return requests.get(f"{BACKEND}/health", timeout=20).json()
    except Exception as e:
        st.warning(f"Health endpoint nem elérhető: {e}")
        return {}

def call_chat(msg: str):
    payload = {"message": msg}
    r = requests.post(f"{BACKEND}/chat", json=payload, timeout=300)
    if not r.ok:
        try:
            data = r.json()
            detail = data.get("detail") if isinstance(data, dict) else None
        except Exception:
            detail = None
        raise RuntimeError(detail or f"{r.status_code} {r.reason}")
    return r.json()

# ----------------- UI -----------------

btn_col, rag_col = st.columns([1,1], gap="large")
with btn_col:
    if st.button("Legutolsó 4 CSV letöltése és betöltése", use_container_width=True):
        try:
            call_refresh()
        except Exception as e:
            st.error(f"Frissítési hiba: {e}")

with rag_col:
    if st.button("Adatok feltöltése RAG-ba", use_container_width=True):
        try:
            call_rag_load()
        except Exception as e:
            st.error(f"RAG hiba: {e}")

meta = call_health()
st.subheader("Állapot")
st.json(meta)

st.divider()
st.subheader("Chat")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_q = st.text_input("Kérdezz az adatokból (pl. 'Mely megyékben a legmagasabb az IPA?')")

if st.button("Küldés") and user_q:
    try:
        data = call_chat(user_q)
        st.session_state["history"].append((user_q, data.get("answer", ""), data.get("tool_result")))
    except Exception as e:
        st.error(f"Hiba a chat hívásnál: {e}")

for q, a, tr in reversed(st.session_state["history"]):
    st.markdown(f"**Te:** {q}")
    st.markdown(f"**Válasz:** {a}")
    if tr and isinstance(tr, dict) and tr.get("rows"):
        with st.expander("SQL eredmény (első 100 sor)"):
            rows = tr.get("rows", [])[:100]
            st.dataframe(rows, use_container_width=True)
    st.markdown("---")
