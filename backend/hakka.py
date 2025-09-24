# backend/hakka.py
from __future__ import annotations

import logging, re, unicodedata
from io import BytesIO
from typing import List, Tuple, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, FeatureNotFound

logger = logging.getLogger("hakka")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

DOWNLOAD_BASE = "https://hakka.allamkincstar.gov.hu/Letoltes.aspx"
TARGET_COL_LABEL = "PM-es tételes települési adó adatok szöveges formátumban"

# A legfelső sorból letöltendő oszlopok indexei (Időszak=0):
# 1: Országos szöveges, 3: PM tételes szöveges, 5: PM települési szöveges, 7: PM kedvezmények szöveges
ROW_COL_INDEXES: List[int] = [1, 3, 5, 7]

MEGYE_MAP: Dict[str, str] = {
    "01":"Budapest","02":"Baranya","03":"Bács-Kiskun","04":"Békés","05":"Borsod-Abaúj-Zemplén",
    "06":"Csongrád-Csanád","07":"Fejér","08":"Győr-Moson-Sopron","09":"Hajdú-Bihar","10":"Heves",
    "11":"Komárom-Esztergom","12":"Nógrád","13":"Pest","14":"Somogy","15":"Szabolcs-Szatmár-Bereg",
    "16":"Jász-Nagykun-Szolnok","17":"Tolna","18":"Vas","19":"Veszprém","20":"Zala",
}

TELEPULES_NAME_CANDIDATES = [
    "Település","település","Telepules","telepules","Település neve",
    "Önkormányzat","önkormányzat","Onkormanyzat",
]
COUNTY_NAME_CANDIDATES = [
    "Megye","megye","Megye neve","megye neve","Vármegye","vármegye","Vármegye neve","vármegye neve",
]
SETTLEMENT_CODE_CANDIDATES = [
    "Település kód","település kód","Település azonosító","település azonosító",
    "KSH kód","KSH szám","KSH azonosító","Település KSH kód",
    "PIR település","PIR telepules","PIR kód","PIR kod","PIR azonosító",
    "Településazonosító","Település azonosító kód",
]

def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")

def _collect_form_fields(form: BeautifulSoup) -> dict:
    data: Dict[str, str] = {}
    for inp in form.select("input[name]"):
        name = inp.get("name"); typ = (inp.get("type") or "").lower()
        if not name: continue
        if typ in ("checkbox","radio"):
            if inp.has_attr("checked"): data[name] = inp.get("value","on")
        else:
            data[name] = inp.get("value","")
    for sel in form.select("select[name]"):
        name = sel.get("name")
        if not name: continue
        opt = sel.select_one("option[selected]") or sel.select_one("option")
        data[name] = opt.get("value","") if opt else ""
    for ta in form.select("textarea[name]"):
        data[ta["name"]] = ta.text
    return data

def _extract_target_arg(val: str) -> tuple[Optional[str], Optional[str]]:
    m = re.search(r"__doPostBack\('([^']+)','([^']*)'\)", val or "")
    return (m.group(1), m.group(2)) if m else (None, None)

def _extract_target_from_cell(cell: BeautifulSoup) -> Optional[str]:
    a = cell.find("a")
    if a:
        for attr in ("href","onclick"):
            val = a.get(attr, "") or ""
            m = re.search(r"__doPostBack\('([^']+)'", val)
            if m: return m.group(1)
    cand = cell.find(["button","input","span"], onclick=True)
    if cand:
        m = re.search(r"__doPostBack\('([^']+)'", cand.get("onclick",""))
        if m: return m.group(1)
    return None

def _norm(s: str) -> str:
    if s is None: return ""
    s = ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c))
    return re.sub(r"\s+"," ", s.strip().lower())

_NAME_TO_CODE = { _norm(v): k for k, v in MEGYE_MAP.items() }
_NAME_TO_CODE.update({
    _norm("Főváros"): "01", _norm("Budapest főváros"): "01",
    _norm("Csongrád"): "06", _norm("Komarom-Esztergom"): "11",
    _norm("Gyor-Moson-Sopron"): "08", _norm("Szabolcs-Szatmar-Bereg"): "15",
})

# --- CÉLTÁBLA KERESŐ (csak ez az új rész a robusztussághoz) ---
def _find_download_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    # 1) ID alapján (renderelt: ctl00_tartalom_gdvLetoltesIdoszakok)
    t = soup.select_one("table[id*='gdvLetoltesIdoszakok']")
    if t: return t
    # 2) Bármelyik table, ahol van libTelepulesiCsv postback link
    for cand in soup.find_all("table"):
        if cand.find(lambda tag: tag.name in ("a","button","input") and
                                 ("__doPostBack" in (tag.get("href","")+tag.get("onclick","")) and
                                  "libTelepulesiCsv" in (tag.get("href","")+tag.get("onclick","")))):
            return cand
    return None

# ---------- 4) A legfrissebb SOR 4 egymás melletti oszlopának letöltése ----------
def download_latest_4_csv_bytes() -> List[Tuple[str, bytes]]:
    """
    A Letoltes.aspx legfelső (legfrissebb) sorából tölti le az egymás melletti 4 oszlop
    szöveges CSV-jeit (ROW_COL_INDEXES = [1,3,5,7]). Csak attachment választ fogad el.
    """
    logger.info("DL: GET %s", DOWNLOAD_BASE)
    sess = requests.Session()
    sess.headers.update({"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    r = sess.get(DOWNLOAD_BASE, timeout=60); r.raise_for_status()
    soup = _soup(r.text)

    form = soup.find("form")
    table = _find_download_table(soup)
    if not form or not table:
        raise RuntimeError("Form vagy a letöltési táblázat nem található a HAKKA oldalon.")

    # Legfelső adat-sor kiválasztása: az első TR, amelyikben legalább egy __doPostBack-es 'Letöltés' van
    top = None
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        has_link = any(
            re.search(r"__doPostBack\('", (a.get("href","")+a.get("onclick","")))
            for a in tr.find_all(["a","button","input"])
        )
        if has_link:
            top = tr
            break
    if top is None:
        raise RuntimeError("Nem találtam 'Letöltés' linket tartalmazó adat sort.")

    tds = top.find_all("td")
    period = tds[0].get_text(strip=True) if tds else "ISMERETLEN"
    logger.info("ROW mode (top row) period=%s", period)

    # Cél cellák -> EVENTTARGET
    event_targets: List[str] = []
    for ci in ROW_COL_INDEXES:
        if ci >= len(tds):
            raise RuntimeError(f"A sor cellái rövidebbek a várt oszlopindexnél (kért={ci}, elérhető={len(tds)-1}).")
        et = _extract_target_from_cell(tds[ci])
        if not et:
            raise RuntimeError(f"Nem találtam __doPostBack targetet a(z) {ci}. oszlopban.")
        logger.info("ROW target -> col=%d, EVENTTARGET=%s", ci, et)
        event_targets.append(et)

    base_fields = _collect_form_fields(form)
    sm_name_el = form.find("input", {"name": re.compile(r".*ScriptManager.*")})
    sm_name = sm_name_el.get("name") if sm_name_el else None

    results: List[Tuple[str, bytes]] = []
    for et in event_targets:
        # Szinkron postback
        form_data = dict(base_fields)
        form_data["__EVENTTARGET"]   = et
        form_data["__EVENTARGUMENT"] = ""
        logger.info("POST sync -> EVENTTARGET=%s", et)
        r1 = sess.post(DOWNLOAD_BASE, data=form_data, timeout=180, headers={"Referer": DOWNLOAD_BASE})
        r1.raise_for_status()

        dispo = r1.headers.get("Content-Disposition") or ""
        payload: Optional[bytes] = None

        if "attachment" not in dispo:
            # UpdatePanel/AJAX fallback
            soup2 = _soup(r1.text)
            form2 = soup2.find("form") or soup
            fields2 = _collect_form_fields(form2)
            ajax = dict(fields2)
            ajax["__EVENTTARGET"]   = et
            ajax["__EVENTARGUMENT"] = ""
            ajax["__ASYNCPOST"]     = "true"
            if sm_name:
                ajax[sm_name] = f"{sm_name}|{et}"
            logger.info("POST ajax -> EVENTTARGET=%s", et)
            r2 = sess.post(
                DOWNLOAD_BASE,
                data=ajax,
                timeout=180,
                headers={"Referer": DOWNLOAD_BASE, "X-MicrosoftAjax":"Delta=true"},
            )
            r2.raise_for_status()
            if "attachment" in (r2.headers.get("Content-Disposition") or ""):
                payload = r2.content
        else:
            payload = r1.content

        if not payload:
            logger.info("SKIP non-attachment for EVENTTARGET=%s", et)
            continue

        results.append((period, payload))
        logger.info("ACCEPT CSV -> period=%s, size=%d", period, len(payload))

    if not results:
        raise RuntimeError("Nem találtam letölthető CSV-t (postback).")
    return results[:4]

# ---------- 5) Kódolás-detektálás + CSV olvasás ----------
def _detect_encoding(data: bytes) -> Optional[str]:
    if data.startswith(b"\xef\xbb\xbf"): return "utf-8-sig"
    for enc in ("cp1250","latin2","iso-8859-2","windows-1250"):
        try:
            _ = data[:2048].decode(enc); return enc
        except Exception: pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding: return best.encoding
    except Exception: pass
    try:
        import chardet
        det = chardet.detect(data[:10000])
        if det and det.get("encoding"): return det["encoding"]
    except Exception: pass
    return None

def read_csv_bytes(data: bytes) -> pd.DataFrame:
    enc_guess = _detect_encoding(data)
    encs = [e for e in (enc_guess,"utf-8-sig","cp1250","latin2","iso-8859-2","windows-1250","utf-8") if e]
    for enc in encs:
        for sep in (";","\t",","):
            try:
                df = pd.read_csv(BytesIO(data), sep=sep, encoding=enc, dtype=str)
                if df.shape[1] >= 3: return df
            except Exception: continue
    return pd.read_csv(BytesIO(data), sep=None, engine="python", dtype=str)

# ---------- 6) Normalizálás + JOIN_KEY (KSH fallback) ----------
def normalize_and_key(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    name_col = next((c for c in df.columns if c in TELEPULES_NAME_CANDIDATES), None)
    if name_col is None:
        name_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())

    county_name_col = next((c for c in df.columns if c in COUNTY_NAME_CANDIDATES), None)

    settle_code_col = next((c for c in df.columns if c in SETTLEMENT_CODE_CANDIDATES), None)
    if settle_code_col is None:
        for c in df.columns:
            s = df[c].astype(str).str.replace(r"\D","", regex=True)
            if (s.str.len() >= 4).mean() > 0.6:
                settle_code_col = c; break

    megye_kod_series: Optional[pd.Series] = None
    megye_name_series: Optional[pd.Series] = None

    if county_name_col is not None:
        megye_name_series = df[county_name_col].astype(str).str.strip()
        megye_kod_series  = megye_name_series.map(lambda s: _NAME_TO_CODE.get(_norm(s)))
    elif settle_code_col is not None:
        code2 = df[settle_code_col].astype(str).str.replace(r"\D","", regex=True).str[:2]
        code2 = code2.where(code2.str.len() == 2, None)
        megye_kod_series  = code2
        megye_name_series = code2.map(lambda k: MEGYE_MAP.get(k, k))
    else:
        for c in df.columns:
            s = df[c].astype(str).str.replace(r"\D","", regex=True).str[:2]
            if (s.str.len() == 2).mean() > 0.5:
                megye_kod_series  = s
                megye_name_series = s.map(lambda k: MEGYE_MAP.get(k, k))
                break

    if megye_kod_series is None and megye_name_series is None:
        raise ValueError("Megye kód/név nem található – sem Megye/Vármegye oszlop, sem település-kód nincs.")

    if megye_name_series is None and megye_kod_series is not None:
        megye_name_series = megye_kod_series.map(lambda k: MEGYE_MAP.get(str(k).zfill(2), str(k).zfill(2)))
    if megye_kod_series is None and megye_name_series is not None:
        megye_kod_series = megye_name_series.map(lambda s: _NAME_TO_CODE.get(_norm(s), None))

    df["MegyeKod"]     = megye_kod_series.astype(str).str.zfill(2)
    df["Megye"]        = megye_name_series.astype(str).str.strip()
    df["TelepulesNev"] = df[name_col].astype(str).str.strip()
    df["JOIN_KEY"]     = df["Megye"].astype(str).str.strip() + "|" + df["TelepulesNev"]
    return df

# ---------- 7) Backend belépő ----------
def fetch_and_prepare_latest() -> List[Tuple[str, pd.DataFrame]]:
    logger.info("REFRESH: start")
    pairs = download_latest_4_csv_bytes()
    prepared: List[Tuple[str, pd.DataFrame]] = []
    for period, blob in pairs:
        df = read_csv_bytes(blob)
        df = normalize_and_key(df)
        prepared.append((period, df))
    logger.info("REFRESH: done -> %d tables", len(prepared))
    return prepared
