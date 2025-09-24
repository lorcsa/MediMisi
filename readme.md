############################
# README.md
############################
# Törvény MISI – HAKKA CSV Chat (FastAPI + Streamlit)
Pécsi verseny 2025.09.23 (dR. Tóth Judit Lenke & Donkó Loránd & Misi)


Ez a kis app automatikusan **a HAKKA Letöltés** oldalról a **legutolsó 4 CSV** fájlt letölti a
**PM-es tételes települési adó adatok szöveges formátumban** oszlopból, betölti memóriába,
**megye kódot** valós **megye névvé** alakítja, és `JOIN_KEY = Megye|Település` kulcsot hoz létre.


> Források/tények:
> * HAKKA Letöltés táblázat (időszakok és oszlopok): lásd a MÁK oldalát. A legfrissebb hónapok legfelül vannak.
> * Megyekód → megye-név mapping stabil KSH/OKFŐ nómenklatúra szerint (Budapest=01 .. Zala=20).
> Lásd: OKFŐ kódrendszerek oldalán a megyekód táblázatot.


## Telepítés


```bash
python -m venv .venv
. .venv/Scripts/activate # Windows
pip install -r requirements.txt
# Playwright böngésző letöltése (egyszer):
python -m playwright install chromium
```


`.env` fájl:
```
OPENAI_API_KEY=sk-...
# opcionális:
BACKEND_URL=http://127.0.0.1:8000
```


## Indítás


Két külön terminálban:


```bash
# Backend
uvicorn backend.main:app --reload --port 8000


# Frontend
streamlit run frontend/app.py
```


## Használat
1. Frontend gomb: **„Legutolsó 4 CSV letöltése és betöltése”** – a backend Playwright‑tal letölti az első 4 hónap CSV‑jét.
2. A betöltött táblák: `t1_YYYYMM`, `t2_YYYYMM`, ...
3. **Chat** mezőben magyar kérdések. A modellek a `run_sql` eszközzel **DuckDB**‑n SQL‑t futtathatnak, és válaszolnak.
4. Több táblát **JOIN_KEY** alapján kapcsolj ("Megye|Település").


## Megjegyzések (robusztusság)
- A HAKKA oldal ASP.NET postbacket használ; ezért **Playwright**‑ot alkalmazunk a linkek kattintásához és a letöltésekhez, így nem kell URL‑mintákat találgatni.
- CSV beolvasásnál több kódlapot és elválasztót próbálunk (`cp1250`, `utf-8-sig`; `;`, `\t`, `,`).
- A megye kód → megye név mapping stabil, a KSH/OKFŐ nomenklatúra alapján van bedrótozva.
- Biztonság: API‑kulcs az `.env`‑ben, a backend nem logol adatot.

