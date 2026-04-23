import os
import re
import json
import math
import time
import sqlite3
import shutil
import threading
import zipfile
import hashlib
import functools
import uuid
import sys
import bisect
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Set, List, Tuple

import tkinter as tk
from tkinter import filedialog
import requests
import numpy as np
import faiss
from flask import Flask, request, render_template_string, redirect, jsonify, flash, send_from_directory
from huggingface_hub import snapshot_download

# =========================
# CONFIG
# =========================
HF_REPO = "ArieLLL123/otzaria-embeddings"

def _find_default_db() -> str:
    paths = [
        r"C:\אוצריא\אוצריא\seforim.db",
        r"C:\אוצריא\seforim.db",
        os.path.join(os.environ.get('APPDATA', ''), 'io.github.kdroidfilter.seforimapp', 'databases', 'seforim.db'),
        r"C:\Users\daniely\AppData\Roaming\io.github.kdroidfilter.seforimapp\databases\seforim.db",
        os.path.join(os.environ.get('APPDATA', ''), 'Otzaria', 'books', 'seforim.db'),
        r"C:\Users\daniely\AppData\Roaming\Otzaria\books\seforim.db"
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0]

DEFAULT_DB_PATH = _find_default_db()
DB_DOWNLOAD_URL = "https://github.com/Otzaria/otzaria-library/releases/download/library-db-1/seforim.zip"

EDITION_PATHS = {
    "v1": "editions/otzaria_embeddings_v1",
    "v2": "editions/otzaria_embeddings_v2",
    "v3": "editions/otzaria_embeddings_v3",
}

if getattr(sys, 'frozen', False):
    # במצב EXE:
    # EXE_DIR = התיקייה שבה נמצא קובץ ה-EXE (לשמירת הגדרות ו-DB)
    # BUNDLE_DIR = התיקייה הזמנית שבה נפתח ה-EXE (לקריאת המודל הארוז)
    EXE_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = sys._MEIPASS
else:
    # במצב פיתוח רגיל
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = EXE_DIR

BASE_DIR = EXE_DIR  # תמיכה לאחור בקוד שמשתמש ב-BASE_DIR

def _user_data_dir() -> str:
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Otzaria AI")
    return EXE_DIR


DATA_DIR = _user_data_dir() if getattr(sys, 'frozen', False) else EXE_DIR

CACHE_DIR = os.path.join(DATA_DIR, "hf_cache")
RUNTIME_DIR = os.path.join(DATA_DIR, "runtime")
DB_DIR = os.path.join(DATA_DIR, "db")

# בדיקה אם המודלים ארוזים בתוך ה-EXE או נמצאים בחוץ
# הגדרה קבועה לתיקייה מחוץ ל-EXE כדי שהעלאות דרך הממשק יישמרו לתמיד
MODELS_ZIPS_DIR = os.path.join(DATA_DIR, "models_zips")

if os.path.exists(os.path.join(BUNDLE_DIR, "static")):
    STATIC_DIR = os.path.join(BUNDLE_DIR, "static")
else:
    STATIC_DIR = os.path.join(DATA_DIR, "static")

LOCAL_MODELS_DIR = os.path.join(DATA_DIR, "local_models")

SETTINGS_PATH = os.path.join(RUNTIME_DIR, "settings.json")

DEFAULT_TOP_K = 20
DEFAULT_MIN_SCORE = 0.0
SEARCH_MIN_RESULTS = 12
SEARCH_TARGET_RESULTS = 40
SEARCH_MAX_RESULTS = 250
SEARCH_INITIAL_CANDIDATES = 160
SEARCH_MAX_CANDIDATES = 5000
SEARCH_CACHE_SIZE = 24

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(MODELS_ZIPS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

try:
    from werkzeug.utils import secure_filename
except ImportError:
    def secure_filename(filename): return filename

# הגדרות חלון מילים חכם (Smart Chunking)
IDEAL_CHUNK_WORDS = 50   # המספר שבו מתחילים לחפש סימן פיסוק
MAX_CHUNK_WORDS = 60     # הגבול העליון לחיתוך
DEFAULT_OVERLAP_WORDS = 10 # חפיפה בסיסית בין מקטעים

# =========================
# TEXT TOOLS & HEBREW NLP
# =========================
NIQQUD_RE   = re.compile(r"[\u0591-\u05C7]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z\u0590-\u05FF\"']+")
HEB_LETTERS = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ"

def clean_text(s: str) -> str:
    if not s: return ""
    s = HTML_TAG_RE.sub(" ", s)
    s = NIQQUD_RE.sub("", s)
    s = s.replace('״', '"').replace('׳', "'")
    s = NON_WORD_RE.sub(" ", s)
    return " ".join(s.split())

def strip_niqqud(s: str) -> str:
    if not s: return ""
    return NIQQUD_RE.sub("", s)

@functools.lru_cache(maxsize=10000)
def hebrew_stem(word: str) -> str:
    if len(word) < 4: return word
    prefixes = ['וכש', 'וש', 'וה', 'וב', 'ול', 'ומ', 'כש', 'שב', 'שה', 'מש', 'מה', 'ו', 'ה', 'ב', 'ל', 'מ', 'ש', 'כ']
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 2:
            return word[len(p):]
    return word

def get_tokens(text: str) -> Set[str]:
    words = clean_text(text).split()
    return {hebrew_stem(w) for w in words if w}

def fts_query_from_text(q_clean: str) -> str:
    toks = [t for t in clean_text(q_clean).split() if len(t) > 1]
    return " ".join(toks) if toks else ""

# =========================
# SETTINGS PERSISTENCE
# =========================
def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH): return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except: return {}

def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except: pass

# =========================
# ZIP MODEL SUPPORT
# =========================
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_zip_extracted(zip_path: str) -> str:
    if not os.path.exists(zip_path): raise FileNotFoundError(f"ZIP לא נמצא: {zip_path}")
    zhash = sha256_file(zip_path)[:16]
    target_dir = os.path.join(LOCAL_MODELS_DIR, zhash)
    marker = os.path.join(target_dir, ".extracted_ok")
    if os.path.exists(marker): return target_dir
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            member_path = os.path.join(target_dir, member.filename)
            abs_target = os.path.abspath(target_dir)
            abs_member = os.path.abspath(member_path)
            if not abs_member.startswith(abs_target + os.sep) and abs_member != abs_target:
                raise RuntimeError("ZIP לא תקין (path traversal).")
        z.extractall(target_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    return target_dir

def find_model_files(root_dir: str, edition: str) -> tuple[str, str]:
    candidates_vocab = list(Path(root_dir).rglob("vocab.json"))
    candidates_emb   = list(Path(root_dir).rglob("embeddings_last.npy"))
    if not candidates_vocab or not candidates_emb:
        raise FileNotFoundError("לא מצאתי בתוך ה-ZIP את vocab.json ו/או embeddings_last.npy.")
    prefer_key = f"otzaria_embeddings_{edition}".lower()
    def pick(cands):
        for p in cands:
            if prefer_key in str(p).lower(): return str(p)
        return str(cands[0])
    return pick(candidates_vocab), pick(candidates_emb)

def resolve_zip_model_path(edition: str, zip_path: str = "") -> str:
    if zip_path and os.path.exists(zip_path):
        return zip_path
    local_zip_path = os.path.join(MODELS_ZIPS_DIR, f"otzaria_embeddings_{edition}.zip")
    if os.path.exists(local_zip_path):
        return local_zip_path
    bundled_zip_path = os.path.join(BUNDLE_DIR, "models_zips", f"otzaria_embeddings_{edition}.zip")
    if os.path.exists(bundled_zip_path):
        return bundled_zip_path
    return zip_path or local_zip_path

def has_model_source_available(cfg: dict) -> bool:
    model_source = cfg.get("model_source", "zip")
    edition = cfg.get("edition", "v3")
    if model_source == "zip":
        return os.path.exists(resolve_zip_model_path(edition, cfg.get("zip_path", "")))
    return True

# =========================
# DATABASE & STREAMING
# =========================
def get_book_titles(db_path: str) -> Dict[int, str]:
    titles = {}
    if not os.path.exists(db_path): return titles
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute("SELECT id, title FROM book")
        for r in cur: titles[r[0]] = r[1]
        con.close()
    except Exception as e: print(f"שגיאה בטעינת שמות ספרים: {e}")
    return titles

def iter_rows_ordered(db_path: str, chunk_rows: int = 20000):
    if not os.path.exists(db_path): raise FileNotFoundError(f"קובץ מסד הנתונים לא נמצא: {db_path}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=OFF;")
    table_name = "line"
    try:
        con.execute("SELECT 1 FROM lines LIMIT 1")
        table_name = "lines"
    except: pass
    try: con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
    except:
        con.close(); return
    q = f"SELECT id, bookId, lineIndex, content FROM {table_name} WHERE content IS NOT NULL AND content != '' ORDER BY bookId, lineIndex"
    cur = con.execute(q)
    while True:
        rows = cur.fetchmany(chunk_rows)
        if not rows: break
        yield rows
    con.close()

def iter_rows_ordered_filtered(db_path: str, book_ids: Optional[List[int]] = None, chunk_rows: int = 20000):
    if not book_ids:
        yield from iter_rows_ordered(db_path, chunk_rows=chunk_rows)
        return

    if not os.path.exists(db_path): raise FileNotFoundError(f"קובץ מסד הנתונים לא נמצא: {db_path}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=OFF;")
    table_name = "line"
    try:
        con.execute("SELECT 1 FROM lines LIMIT 1")
        table_name = "lines"
    except: pass
    try: con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
    except:
        con.close(); return

    placeholders = ",".join(["?"] * len(book_ids))
    q = (
        f"SELECT id, bookId, lineIndex, content FROM {table_name} "
        f"WHERE content IS NOT NULL AND content != '' AND bookId IN ({placeholders}) "
        "ORDER BY bookId, lineIndex"
    )
    cur = con.execute(q, list(book_ids))
    while True:
        rows = cur.fetchmany(chunk_rows)
        if not rows: break
        yield rows
    con.close()

def iter_chunks(db_path: str, max_chunks: int, ideal_words: int = IDEAL_CHUNK_WORDS, max_words: int = MAX_CHUNK_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS, book_ids: Optional[List[int]] = None):
    rows_iter = iter_rows_ordered_filtered(db_path, book_ids=book_ids)
    buf = []
    cur_book = None
    produced = 0
    # סימני פיסוק שאנחנו מחשיבים כסוף משפט / רעיון
    punctuation = ('.', ':', ';', '?', '!')

    def flush_chunk(buffer_slice, b_id):
        chunk_text = " ".join([w for _, w in buffer_slice])
        cln_text = clean_text(chunk_text)
        if len(cln_text) > 30:
            return {"bookId": b_id, "startLine": buffer_slice[0][0], "endLine": buffer_slice[-1][0], "text": chunk_text, "clean": cln_text}
        return None

    for batch in rows_iter:
        for r in batch:
            b_id = r["bookId"]

            # אם עברנו לספר חדש, נרוקן את החוצץ
            if cur_book is not None and b_id != cur_book:
                if buf and len(buf) > 15:
                    chunk_data = flush_chunk(buf, cur_book)
                    if chunk_data:
                        yield chunk_data
                        produced += 1
                        if produced >= max_chunks: return
                buf = []

            cur_book = b_id
            txt = str(r["content"]).strip()
            if not txt: continue

            # מוסיפים מילים לחוצץ תוך שמירה על סימני הפיסוק המקוריים
            for w in txt.split():
                buf.append((r["lineIndex"], w))

            # כל עוד יש לנו מספיק מילים לחפש חיתוך חכם
            while len(buf) >= ideal_words:
                split_idx = -1
                
                # מחפשים סימן פיסוק בטווח שבין המינימום למקסימום
                for i in range(ideal_words - 1, min(len(buf), max_words)):
                    if buf[i][1].endswith(punctuation):
                        split_idx = i
                        break
                
                # אם לא מצאנו סימן פיסוק (למשל בספרות תורנית ישנה), נחתוך במקסימום
                if split_idx == -1:
                    split_idx = min(len(buf) - 1, max_words - 1)

                # יצירת המקטע ושליחתו
                chunk_slice = buf[:split_idx + 1]
                chunk_data = flush_chunk(chunk_slice, cur_book)
                if chunk_data:
                    yield chunk_data
                    produced += 1
                    if produced >= max_chunks: return

                # חישוב החפיפה (Overlap) - ננסה להתחיל את המקטע הבא מתחילת משפט
                stride_start = (split_idx + 1) - overlap_words
                if stride_start > 0:
                    # סריקה לאחור/קדימה כדי למצוא נקודה להתחיל ממנה את החפיפה (אחרי סימן פיסוק)
                    adjusted_start = stride_start
                    for i in range(max(1, stride_start - 15), min(len(buf), stride_start + 15)):
                        if buf[i-1][1].endswith(punctuation):
                            adjusted_start = i
                            break
                    stride_start = adjusted_start
                else:
                    stride_start = 0

                # חיתוך החוצץ להמשך העבודה
                buf = buf[stride_start:]

    # שאריות אחרונות
    if buf and len(buf) > 15:
        chunk_data = flush_chunk(buf, cur_book)
        if chunk_data:
            yield chunk_data

# =========================
# ENGINE CORE
# =========================
@dataclass
class LoadedModel:
    edition: str
    vocab: Dict[str, int]
    emb_norm: np.ndarray
    idf: np.ndarray
    idx_to_word: Dict[int, str]
    word_freqs: Dict[str, float]
    sorted_vocab: List[str]

@dataclass
class BuiltIndex:
    faiss_index: faiss.Index
    meta_db_path: str
    count: int

class Engine:
    def __init__(self):
        self.model: Optional[LoadedModel] = None
        self.built: Optional[BuiltIndex] = None
        self.book_map: Dict[int, str] = {}
        self.clean_book_titles: List[Tuple[str, str]] = []
        self.library_tree: List[Dict] = []
        self.status = {"state": "idle", "msg": "המערכת מוכנה", "progress": 0}
        self._lock = threading.RLock()
        self._log_seq = 0
        self.log_entries: deque = deque(maxlen=500)
        self.search_cache: Dict[str, List[Dict]] = {}
        self.last_cfg = load_settings()
        self._append_log("idle", self.status["msg"], self.status["progress"])

    def _append_log(self, state: str, msg: str, progress: Optional[int] = None):
        self._log_seq += 1
        self.log_entries.append({
            "id": self._log_seq,
            "ts": time.strftime("%H:%M:%S"),
            "state": state,
            "msg": msg,
            "progress": int(progress if progress is not None else self.status.get("progress", 0)),
        })

    def _update(self, state, msg, progress):
        with self._lock:
            self.status = {"state": state, "msg": msg, "progress": int(progress)}
            self._append_log(state, msg, progress)
        print(f"[{state}] {msg} ({progress}%)")

    def log(self, msg: str, state: str = "info", progress: Optional[int] = None):
        with self._lock:
            self._append_log(state, msg, progress)
        print(f"[{state}] {msg}")

    def get_logs(self, limit: int = 200) -> List[Dict]:
        with self._lock:
            return list(self.log_entries)[-max(1, int(limit)):]

    def update_book_map(self, book_map: Dict[int, str]):
        self.book_map = book_map
        # Pre-calculate cleaned titles for fast autocomplete
        temp = []
        for title in book_map.values():
            temp.append((clean_text(title), title))
        # Sort by length (shortest match first)
        temp.sort(key=lambda x: len(x[0]))
        self.clean_book_titles = temp

    def _hf_snapshot_offline_first(self, allow_patterns: List[str]) -> str:
        try:
            return snapshot_download(repo_id=HF_REPO, repo_type="model", cache_dir=CACHE_DIR, allow_patterns=allow_patterns, local_files_only=True)
        except Exception:
            return snapshot_download(repo_id=HF_REPO, repo_type="model", cache_dir=CACHE_DIR, allow_patterns=allow_patterns, local_files_only=False)

    def load_resources(self, db_path: str, edition: str = "v3", model_source: str = "hf", zip_path: str = ""):
        if db_path and os.path.exists(db_path):
            self.update_book_map(get_book_titles(db_path))
            self.library_tree = get_library_tree(db_path)
        try:
            self._update("downloading", f"טוען מודל {edition} ({model_source})...", 5)
            if model_source == "zip":
                zip_path = resolve_zip_model_path(edition, zip_path)
                extracted_root = ensure_zip_extracted(zip_path)
                vocab_path, emb_path = find_model_files(extracted_root, edition)
            else:
                path = EDITION_PATHS.get(edition, EDITION_PATHS["v3"])
                local_dir = self._hf_snapshot_offline_first([f"{path}/vocab.json", f"{path}/embeddings_last.npy"])
                base = os.path.join(local_dir, path)
                vocab_path = os.path.join(base, "vocab.json")
                emb_path = os.path.join(base, "embeddings_last.npy")

            with open(vocab_path, "r", encoding="utf-8") as f: meta = json.load(f)
            # Use mmap_mode to avoid loading the raw file entirely into RAM before normalization
            emb = np.load(emb_path, mmap_mode='r')
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            # This division creates a new in-memory array, but we saved the RAM of the raw 'emb'
            emb_norm = emb / norms
            vocab = meta["vocab"]
            freqs = np.array(meta.get("freqs", []), dtype=np.float64)
            
            if len(freqs) == len(vocab):
                idf = np.log((np.sum(freqs) + 1) / (freqs + 1)) + 1
                word_freqs = {w: float(freqs[idx]) for w, idx in vocab.items()}
            else:
                idf = np.ones(len(vocab), dtype=np.float32)
                word_freqs = {w: 1.0 for w in vocab.keys()}
                
            idx_to_word = {idx: w for w, idx in vocab.items()}
            sorted_vocab = sorted(vocab.keys())
            self.model = LoadedModel(edition, vocab, emb_norm, idf.astype(np.float32), idx_to_word, word_freqs, sorted_vocab)
            self._update("idle", "המודל נטען בהצלחה", 100)
        except Exception as e:
            self._update("error", f"שגיאה בטעינת מודל: {e}", 0)
            raise

    def _stamp(self, edition: str, max_chunks: int, ideal: int, max_w: int, overlap: int, book_ids: Optional[List[int]] = None) -> str:
        if book_ids:
            normalized = ",".join(str(bid) for bid in sorted(set(book_ids)))
            scope_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
            scope = f"B{len(set(book_ids))}_{scope_hash}"
        else:
            scope = "ALL"
        return f"{edition}_{scope}_N{max_chunks}_Ideal{ideal}_Max{max_w}_Overlap{overlap}"
    
    def build_index(self, db_path: str, max_chunks: int, ideal: int = IDEAL_CHUNK_WORDS, max_w: int = MAX_CHUNK_WORDS, overlap: int = DEFAULT_OVERLAP_WORDS, book_ids: Optional[List[int]] = None):
        if not self.model or self.status["state"] == "indexing": return
        normalized_book_ids = sorted({int(bid) for bid in (book_ids or []) if str(bid).isdigit()})
        stamp = self._stamp(self.model.edition, max_chunks, ideal, max_w, overlap, normalized_book_ids)
        idx_path = os.path.join(RUNTIME_DIR, f"{stamp}.index")
        meta_db_path = os.path.join(RUNTIME_DIR, f"{stamp}.sqlite")

        if os.path.exists(idx_path) and os.path.exists(meta_db_path):
            self._update("loading", "טוען אינדקס קיים...", 50)
            # שימוש ב-Python open כדי לתמוך בנתיבי עברית ב-Windows
            with open(idx_path, "rb") as f:
                idx_data = np.frombuffer(f.read(), dtype=np.uint8)
            idx = faiss.deserialize_index(idx_data)
            self.built = BuiltIndex(idx, meta_db_path, idx.ntotal)
            if not self.book_map: self.update_book_map(get_book_titles(db_path))
            scope_msg = f", {len(normalized_book_ids):,} ספרים" if normalized_book_ids else ""
            self._update("ready", f"מוכן לחיפוש ({idx.ntotal:,} רשומות{scope_msg})", 100)
            return

        if normalized_book_ids:
            self._update("indexing", f"מתחיל בבניית אינדקס עבור {len(normalized_book_ids):,} ספרים...", 0)
        else:
            self._update("indexing", "מתחיל בבניית אינדקס (זה יקח זמן)...", 0)
        
        # שימוש בקובץ זמני ייחודי כדי למנוע התנגשויות בין תהליכים/ת'רדים
        temp_db_path = meta_db_path + f".{uuid.uuid4().hex}.tmp"
        try:
            if os.path.exists(temp_db_path): os.remove(temp_db_path)
        except OSError: pass

        con = sqlite3.connect(temp_db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("DROP TABLE IF EXISTS chunks")
        con.execute("DROP TABLE IF EXISTS chunks_fts")
        con.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, bookId INTEGER, startLine INTEGER, endLine INTEGER, text TEXT)")
        con.execute("CREATE INDEX idx_book ON chunks(bookId)")
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='');")

        d = self.model.emb_norm.shape[1]
        
        # 🔹 OPTIMIZATION: Use IVF Index for large datasets (>20k chunks)
        # This changes complexity from O(N) to O(log N) roughly.
        use_ivf = max_chunks > 20000
        train_size = 0
        if use_ivf:
            # FAISS requires at least one training vector per centroid.
            # Keep the training sample bounded, but always large enough for nlist.
            target_nlist = int(4 * math.sqrt(max_chunks))
            train_size = min(max_chunks, max(10000, min(100000, target_nlist * 8)))
            nlist = max(1, min(target_nlist, train_size))
            quantizer = faiss.IndexFlatIP(d)
            # IndexIVFFlat requires training
            ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            ivf_index.nprobe = 10  # Search 10 nearest clusters (Balance speed/accuracy)
            index = faiss.IndexIDMap(ivf_index)
            is_trained = False
        else:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
            is_trained = True

        vectors, ids, db_buffer, fts_buffer = [], [], [], []
        batch_size = 5000
        total_processed = 0
        start_time = time.time()

        for chunk in iter_chunks(db_path, max_chunks, ideal, max_w, overlap, book_ids=normalized_book_ids or None):
            vec = self._text_to_vec(chunk["clean"])
            if vec is None: continue
            current_id = total_processed
            vectors.append(vec)
            ids.append(current_id)
            db_buffer.append((current_id, chunk["bookId"], chunk["startLine"], chunk["endLine"], chunk["text"]))
            fts_buffer.append((current_id, chunk["clean"]))
            total_processed += 1

            if len(vectors) >= batch_size:
                if use_ivf and not is_trained and total_processed < train_size:
                    pct = min(20, int((total_processed / max(train_size, 1)) * 20))
                    self._update("indexing", f"צובר דגימות לאימון אינדקס IVF ({total_processed:,}/{train_size:,})", pct)
                    continue

                # Train IVF index once enough samples were buffered
                if use_ivf and not is_trained:
                    self._update("indexing", "מאמן אינדקס וקטורי (IVF)...", 5)
                    # We need to access the sub-index to train
                    index.index.train(np.vstack(vectors))
                    is_trained = True

                index.add_with_ids(np.vstack(vectors), np.array(ids).astype("int64"))
                con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", db_buffer)
                con.executemany("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", fts_buffer)
                con.commit()
                vectors, ids, db_buffer, fts_buffer = [], [], [], []
                elapsed = time.time() - start_time
                rate = total_processed / (elapsed + 0.1)
                pct = min(95, int((total_processed / max_chunks) * 100))
                self._update("indexing", f"עובדו {total_processed:,} רשומות ({int(rate)} לשנייה)", pct)

        if vectors:
            if use_ivf and not is_trained:
                 # Handle the final buffered training sample before the first add
                 index.index.train(np.vstack(vectors))
                 is_trained = True
            
            index.add_with_ids(np.vstack(vectors), np.array(ids).astype("int64"))
            con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", db_buffer)
            con.executemany("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", fts_buffer)
            con.commit()

        con.close()
        # שימוש ב-Python open כדי לעקוף בעיות קידוד ב-Faiss C++ IO
        os.makedirs(RUNTIME_DIR, exist_ok=True)
        idx_data = faiss.serialize_index(index)
        with open(idx_path, "wb") as f:
            f.write(idx_data)
        
        # החלפת הקובץ המקורי בקובץ הזמני
        final_db_path = meta_db_path
        try:
            if os.path.exists(meta_db_path): os.remove(meta_db_path)
            os.rename(temp_db_path, meta_db_path)
        except OSError:
            # במקרה של כישלון (קובץ נעול), נשתמש בקובץ הזמני לריצה הנוכחית
            final_db_path = temp_db_path
            
        self.built = BuiltIndex(index, final_db_path, total_processed)
        scope_msg = f" עבור {len(normalized_book_ids):,} ספרים" if normalized_book_ids else ""
        self._update("ready", f"הבנייה הושלמה בהצלחה{scope_msg}!", 100)

    def _text_to_vec(self, text: str):
        if not self.model: return None
        words = text.split()
        if not words: return None
        indices = [self.model.vocab[w] for w in words if w in self.model.vocab]
        if not indices: return None
        idfs = self.model.idf[indices]
        vecs = self.model.emb_norm[indices]
        weighted = vecs * idfs[:, None]
        avg_vec = np.sum(weighted, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm < 1e-9: return None
        return avg_vec / norm

    # 🔹 SPELL CHECK ALGORITHM (NORVIG)
    def check_spelling(self, query: str) -> Optional[str]:
        if not self.model or not query: return None
        words = clean_text(query).split()
        corrected = []
        changed = False
        for w in words:
            if w in self.model.word_freqs or len(w) <= 2:
                corrected.append(w)
            else:
                c = self._correct_word(w)
                corrected.append(c)
                if c != w: changed = True
        return " ".join(corrected) if changed else None

    def _correct_word(self, word: str) -> str:
        candidates = (self._known([word]) or self._known(self._edits1(word)) or [word])
        return max(candidates, key=lambda w: self.model.word_freqs.get(w, 0))

    def _known(self, words):
        return set(w for w in words if w in self.model.word_freqs)

    def _edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in HEB_LETTERS]
        inserts = [L + c + R for L, R in splits for c in HEB_LETTERS]
        return set(deletes + transposes + replaces + inserts)

    # 🔹 QUERY EXPANSION ALGORITHM
    def _build_expanded_fts_query(self, q_clean: str, top_synonyms: int = 2, threshold: float = 0.7) -> str:
        if not self.model or not self.model.idx_to_word:
            return fts_query_from_text(q_clean)
        tokens = [t for t in q_clean.split() if len(t) > 1]
        if not tokens: return ""
        expanded_parts = []
        for t in tokens:
            synonyms = [t]
            if t in self.model.vocab:
                idx = self.model.vocab[t]
                vec = self.model.emb_norm[idx]
                sims = np.dot(self.model.emb_norm, vec)
                best_indices = np.argsort(sims)[-(top_synonyms + 2):][::-1]
                for bi in best_indices:
                    if bi != idx and sims[bi] > threshold:
                        synonyms.append(self.model.idx_to_word[bi])
            part = "(" + " OR ".join(f'"{s}"' for s in synonyms) + ")"
            expanded_parts.append(part)
        return " ".join(expanded_parts)

    def get_expanded_terms(self, q_clean: str, top_synonyms: int = 2, threshold: float = 0.7) -> list[str]:
            if not self.model or not self.model.idx_to_word:
                return [t for t in q_clean.split() if len(t) > 1]
            
            tokens = [t for t in q_clean.split() if len(t) > 1]
            expanded = set(tokens) # נשמור את המילים המקוריות
            
            for t in tokens:
                if t in self.model.vocab:
                    idx = self.model.vocab[t]
                    vec = self.model.emb_norm[idx]
                    sims = np.dot(self.model.emb_norm, vec)
                    # מציאת המילים הקרובות ביותר (וקטורית)
                    best_indices = np.argsort(sims)[-(top_synonyms + 2):][::-1]
                    for bi in best_indices:
                        if bi != idx and sims[bi] > threshold:
                            expanded.add(self.model.idx_to_word[bi])
                            
            return list(expanded)

    def _fts_candidates(self, q_clean: str, limit: int, book_filter: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if not self.built: return []
        fts_q = self._build_expanded_fts_query(q_clean)
        if not fts_q: return []
        con = sqlite3.connect(self.built.meta_db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            if book_filter:
                # סינון ברמת ה-SQL: חיפוש רק בתוך הספר הרלוונטי
                placeholders = ",".join(["?"] * len(book_filter))
                sql = f"SELECT f.rowid, bm25(f) AS bm FROM chunks_fts f JOIN chunks c ON f.rowid = c.rowid WHERE c.bookId IN ({placeholders}) AND f.chunks_fts MATCH ? LIMIT ?"
                params = list(book_filter) + [fts_q, int(limit)]
            else:
                sql = "SELECT rowid, bm25(chunks_fts) AS bm FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?"
                params = (fts_q, int(limit))
            
            rows = con.execute(sql, params).fetchall()
            return [(int(r["rowid"]), float(r["bm"])) for r in rows]
        except: return []
        finally: con.close()

    def search(self, query: str, book_filter: Optional[List[int]] = None, top_k: Optional[int] = 20):
        if not self.model or not self.built: return []
        q_clean = clean_text(query)
        q_vec = self._text_to_vec(q_clean)
        if q_vec is None: return []
        requested_k = None if not top_k or top_k <= 0 else int(top_k)
        index_count = int(self.built.count or 0)

        # 1) מועמדים וקטוריים עם סינון מוקדם (Pre-filtering)
        search_params = None
        if book_filter:
            con = sqlite3.connect(self.built.meta_db_path, timeout=30)
            placeholders = ",".join(["?"] * len(book_filter))
            # שליפת כל ה-rowids השייכים לספרים שנבחרו (בסדר ממוין לביצועים אופטימליים ב-FAISS)
            res = con.execute(f"SELECT rowid FROM chunks WHERE bookId IN ({placeholders}) ORDER BY rowid", book_filter).fetchall()
            con.close()
            
            target_ids = np.array([r[0] for r in res], dtype=np.int64)
            search_space_count = int(len(target_ids))
            if len(target_ids) == 0:
                return [] # אין מקטעים מאונדקסים עבור הספרים שנבחרו
                
            selector = faiss.IDSelectorArray(target_ids)
            # שימוש ב-downcast_index הכרחי כדי ש-Python יזהה שהאינדקס הפנימי הוא מסוג IVF
            # ויאפשר יצירת SearchParametersIVF תקינים.
            underlying_index = faiss.downcast_index(self.built.faiss_index.index)
            if isinstance(underlying_index, faiss.IndexIVF):
                search_params = faiss.SearchParametersIVF(sel=selector, nprobe=underlying_index.nprobe)
            else:
                search_params = faiss.SearchParameters(sel=selector)
            
            # כשמסננים מראש, אין צורך ב-K ענקי כי כל התוצאות שיחזרו הן מהספרים הנכונים
            vec_candidates_k = search_space_count if requested_k is None else max(requested_k * 4, 100)
        else:
            search_space_count = index_count
            vec_candidates_k = index_count if requested_k is None else max(requested_k * 20, 200)

        if search_space_count <= 0:
            return []

        scores, ids = self.built.faiss_index.search(np.array([q_vec]), vec_candidates_k, params=search_params)
        vec_found_ids = [int(i) for i in ids[0] if i >= 0]

        fts_candidates_k = search_space_count if requested_k is None else max(requested_k * 20, 200)
        fts_rows = self._fts_candidates(q_clean, fts_candidates_k, book_filter=book_filter)
        fts_found_ids = [rid for rid, _ in fts_rows]

        union_ids = list(set(vec_found_ids + fts_found_ids))
        if not union_ids: return []

        con = sqlite3.connect(self.built.meta_db_path, timeout=30)
        con.row_factory = sqlite3.Row
        placeholders = ",".join(["?"] * len(union_ids))
        sql = f"SELECT rowid, bookId, startLine, endLine, text FROM chunks WHERE rowid IN ({placeholders})"
        params: List = list(union_ids)
        if book_filter:
            placeholders_books = ",".join(["?"] * len(book_filter))
            sql += f" AND bookId IN ({placeholders_books})"
            params.extend(book_filter)
        rows = con.execute(sql, params).fetchall()
        con.close()

        vec_scores = {int(fid): float(scr) for fid, scr in zip(ids[0], scores[0]) if int(fid) >= 0}
        fts_bm = {rid: bm for rid, bm in fts_rows}
        def bm_to_rel(bm: Optional[float]) -> float:
            return 1.0 / (1.0 + max(0.0, bm)) if bm is not None else 0.0

        q_tokens = get_tokens(q_clean)
        cfg = self.last_cfg or {}
        w_vec       = float(cfg.get("w_vec", 0.35))
        w_bm        = float(cfg.get("w_bm", 0.25))
        w_overlap   = float(cfg.get("w_overlap", 0.25))
        w_phrase    = float(cfg.get("w_phrase", 0.10))
        w_proximity = float(cfg.get("w_proximity", 0.05))
        total_weight = w_vec + w_bm + w_overlap + w_phrase + w_proximity
        if total_weight == 0: total_weight = 1

        results = []
        for r in rows:
            rid = int(r["rowid"])
            chunk_txt = r["text"]
            chunk_clean = clean_text(chunk_txt)
            chunk_tokens = get_tokens(chunk_clean)
            chunk_words = chunk_clean.split()

            base_vec = vec_scores.get(rid, 0.0)
            bm_rel = bm_to_rel(fts_bm.get(rid))
            intersection = len(q_tokens & chunk_tokens)
            overlap = (intersection / len(q_tokens)) if q_tokens else 0.0
            phrase = 1.0 if (q_clean and q_clean in chunk_clean) else 0.0
            proximity = 0.0
            if intersection > 1 and q_tokens:
                # חישוב קרבה עם דעיכה לחזרות (Decay for repetitions)
                # 1st: 100%, others from config
                d2 = float(cfg.get("decay_2", 0.75))
                d3 = float(cfg.get("decay_3", 0.35))
                d4 = float(cfg.get("decay_4", 0.05))
                decay_factors = [1.0, d2, d3, d4]
                found_indices = []
                effective_count = 0.0
                
                for qw in q_tokens:
                    qw_indices = [i for i, cw in enumerate(chunk_words) if hebrew_stem(cw) == qw]
                    if qw_indices:
                        found_indices.extend(qw_indices)
                        for k in range(len(qw_indices)):
                            if k < len(decay_factors):
                                effective_count += decay_factors[k]

                if found_indices:
                    span = max(found_indices) - min(found_indices)
                    density = effective_count / (span + 1)
                    proximity = min(density, 1.0)

            final_score = ((base_vec * w_vec) + (bm_rel * w_bm) + (overlap * w_overlap) + (phrase * w_phrase) + (proximity * w_proximity)) / total_weight
            book_title = self.book_map.get(int(r["bookId"]), f"ספר {int(r['bookId'])}")

            results.append({
                "score": float(final_score),
                "text": chunk_txt,
                "source": f"{book_title}, שורה {int(r['startLine'])}",
                "book_id": int(r["bookId"]),
            "line_index": int(r["startLine"]),
            "end_line": int(r["endLine"]),
                "book_title": book_title,
                "features": {"vec": float(base_vec), "bm": float(bm_rel), "overlap": float(overlap), "phrase": float(phrase), "prox": float(proximity)}
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results if requested_k is None else results[:requested_k]

    def get_indexed_book_ids(self) -> Set[int]:
        """מחזיר את רשימת ה-IDs של הספרים שקיימים באינדקס בפועל"""
        if not self.built or not os.path.exists(self.built.meta_db_path): return set()
        try:
            con = sqlite3.connect(self.built.meta_db_path)
            # בדיקה מהירה בטבלת chunks (יש עליה אינדקס)
            cur = con.execute("SELECT DISTINCT bookId FROM chunks")
            ids = {r[0] for r in cur}
            con.close()
            return ids
        except: return set()

ENGINE = Engine()

# =========================
# FLASK WEB APP
# =========================
app = Flask(__name__)
app.secret_key = "otzaria_ai_secret_v5"

BASE_DIR = Path(__file__).parent
HTML_TEMPLATE = (BASE_DIR / "app_ai.html").read_text(encoding="utf-8")

# =========================
# HELPER FILTERS
# =========================
def close_html_tags(html: str) -> str:
    """סוגר תגיות HTML פתוחות ומסיר תגיות חתוכות בסוף המחרוזת למניעת שיבוש בעיצוב הדף."""
    # הסרת תגית שמתחילה בסוף המחרוזת אך לא נסגרה (למשל "טקסט <a")
    html = re.sub(r'<[^>]*$', '', html)
    
    # מציאת כל התגיות (פתיחה, סגירה, וסגירה עצמית)
    tag_regex = re.compile(r'<(/?)([a-zA-Z0-9]+)[^>]*(/?)>')
    stack = []
    void_tags = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
    
    for match in tag_regex.finditer(html):
        is_closing = match.group(1) == '/'
        tag_name = match.group(2).lower()
        is_self_closing = match.group(3) == '/'
        
        if tag_name in void_tags or is_self_closing:
            continue
        if is_closing:
            if stack and stack[-1] == tag_name:
                stack.pop()
        else:
            stack.append(tag_name)
            
    # סגירת כל התגיות שנותרו פתוחות בסדר הפוך
    while stack:
        html += f'</{stack.pop()}>'
    return html

def highlight_text(text, query):
    text = close_html_tags(text)
    if not query: return text
    q_words = [hebrew_stem(w) for w in clean_text(query).split() if len(w) > 1]
    if not q_words: return text
    patterns = [r'(?:^|[\s\"\'\-])([ו|מ|ש|ה|ל|ב|כ]?' + re.escape(w) + r')(?=[\s\"\'\.\,\-]|$)' for w in q_words]
    combined_pattern = "|".join(patterns)
    def replacer(match):
        full_match = match.group(0)
        word_match = re.search(r'[א-ת]+', full_match)
        if word_match: return full_match.replace(word_match.group(0), f'<mark>{word_match.group(0)}</mark>')
        return full_match
    try: return re.sub(combined_pattern, replacer, text)
    except: return text

def highlight_text(text, query):
    text = close_html_tags(text)
    if not query: return text
    q_words = [hebrew_stem(w) for w in clean_text(query).split() if len(w) > 1]
    if not q_words: return text

    prefixes = "ובשהלמכ"
    token_pattern = "|".join(re.escape(w) for w in sorted(set(q_words), key=len, reverse=True))
    combined_pattern = re.compile(
        rf"(^|[\s\"'\-])([{prefixes}]?(?:{token_pattern}))(?=[\s\"'\.\,\-]|$)"
    )

    def highlight_plain_segment(segment: str) -> str:
        normalized_chars = []
        index_map = []
        for idx, ch in enumerate(segment):
            normalized = strip_niqqud(ch)
            if not normalized:
                continue
            normalized_chars.append(normalized)
            index_map.extend([idx] * len(normalized))

        normalized_segment = "".join(normalized_chars)
        if not normalized_segment:
            return segment

        ranges = []
        for match in combined_pattern.finditer(normalized_segment):
            start_norm, end_norm = match.span(2)
            if start_norm >= len(index_map) or end_norm <= 0:
                continue
            start_idx = index_map[start_norm]
            end_idx = index_map[end_norm - 1] + 1
            ranges.append((start_idx, end_idx))

        if not ranges:
            return segment

        merged = []
        for start_idx, end_idx in ranges:
            if merged and start_idx <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_idx))
            else:
                merged.append((start_idx, end_idx))

        parts = []
        last_idx = 0
        for start_idx, end_idx in merged:
            parts.append(segment[last_idx:start_idx])
            parts.append(f"<mark>{segment[start_idx:end_idx]}</mark>")
            last_idx = end_idx
        parts.append(segment[last_idx:])
        return "".join(parts)

    try:
        parts = re.split(r"(<[^>]+>)", text)
        for i, part in enumerate(parts):
            if part and not part.startswith("<"):
                parts[i] = highlight_plain_segment(part)
        return "".join(parts)
    except:
        return text

def _result_feat(result: Dict, key: str, default: float = 0.0) -> float:
    return float(result.get("features", {}).get(key, default) or default)

def _passes_relevance_gate(result: Dict, token_count: int, relaxed: bool = False) -> bool:
    vec = _result_feat(result, "vec")
    bm = _result_feat(result, "bm")
    overlap = _result_feat(result, "overlap")
    phrase = _result_feat(result, "phrase")
    prox = _result_feat(result, "prox")

    if phrase >= 1.0:
        return True
    if overlap >= 0.72:
        return True
    if overlap >= 0.5 and (bm >= 0.08 or vec >= 0.22 or prox >= 0.08):
        return True
    if bm >= 0.22 and overlap >= 0.2:
        return True
    if vec >= 0.72 and overlap >= 0.18:
        return True

    if relaxed:
        short_query = token_count <= 2
        if overlap >= (0.2 if short_query else 0.26) and (bm >= 0.05 or vec >= 0.18 or prox >= 0.05):
            return True
        if bm >= 0.16 and (overlap >= 0.12 or vec >= 0.28):
            return True
        if vec >= (0.60 if short_query else 0.66) and overlap >= 0.1:
            return True

    return False

def _adaptive_relevance_filter(raw_results: List[Dict], min_score: float, query: str) -> List[Dict]:
    if not raw_results:
        return []

    token_count = max(1, len(get_tokens(clean_text(query))))
    score_floor = max(0.0, float(min_score or 0.0))
    prefiltered = [r for r in raw_results if float(r.get("score", 0.0)) >= score_floor]
    if not prefiltered:
        return []

    strict_pool = [r for r in prefiltered if _passes_relevance_gate(r, token_count, relaxed=False)]
    relaxed_pool = [r for r in prefiltered if _passes_relevance_gate(r, token_count, relaxed=True)]

    pool = strict_pool
    if len(pool) < SEARCH_MIN_RESULTS:
        pool = relaxed_pool if relaxed_pool else strict_pool
    if not pool:
        return prefiltered[:SEARCH_MIN_RESULTS]

    top_score = float(pool[0]["score"])
    dynamic_floor = max(score_floor, min(0.18, top_score * 0.45))

    if len(pool) > SEARCH_MAX_RESULTS:
        dynamic_floor = max(dynamic_floor, float(pool[SEARCH_MAX_RESULTS - 1]["score"]))
    elif len(pool) < SEARCH_MIN_RESULTS:
        dynamic_floor = max(score_floor, min(0.08, top_score * 0.22))
    elif len(pool) < SEARCH_TARGET_RESULTS:
        dynamic_floor = max(score_floor, min(0.12, top_score * 0.32))

    filtered = [r for r in pool if float(r.get("score", 0.0)) >= dynamic_floor]

    if len(filtered) < SEARCH_MIN_RESULTS and pool is strict_pool and relaxed_pool:
        fallback_floor = max(score_floor, min(0.08, float(relaxed_pool[0]["score"]) * 0.22))
        filtered = [r for r in relaxed_pool if float(r.get("score", 0.0)) >= fallback_floor]

    return filtered if filtered else pool[:SEARCH_MIN_RESULTS]

def _search_cache_key(query: str, selected_books: List[int], min_score: float, top_k: int) -> str:
    cfg = ENGINE.last_cfg or {}
    built = ENGINE.built
    built_sig = (
        getattr(built, "meta_db_path", ""),
        int(getattr(built, "count", 0) or 0),
    )
    scoring_sig = tuple(
        round(float(cfg.get(k, d)), 4)
        for k, d in [
            ("w_vec", 0.35),
            ("w_bm", 0.25),
            ("w_overlap", 0.25),
            ("w_phrase", 0.10),
            ("w_proximity", 0.05),
            ("decay_2", 0.75),
            ("decay_3", 0.35),
            ("decay_4", 0.05),
        ]
    )
    return json.dumps({
        "q": clean_text(query),
        "books": list(selected_books or []),
        "min_score": round(float(min_score or 0.0), 4),
        "top_k": int(top_k),
        "built": built_sig,
        "scoring": scoring_sig,
    }, ensure_ascii=True, sort_keys=True)

def _cache_search_results(key: str, results: List[Dict]):
    cache = ENGINE.search_cache
    cache[key] = results
    while len(cache) > SEARCH_CACHE_SIZE:
        cache.pop(next(iter(cache)))

def _collect_filtered_results(query: str, selected_books: List[int], min_score: float, top_k: int, page: int) -> List[Dict]:
    cache_key = _search_cache_key(query, selected_books, min_score, top_k)
    cached = ENGINE.search_cache.get(cache_key)
    target_results = min(SEARCH_MAX_RESULTS, max(page * top_k, SEARCH_TARGET_RESULTS))
    if cached is not None and len(cached) >= min(target_results, SEARCH_MAX_RESULTS):
        return cached

    candidate_budget = min(SEARCH_MAX_CANDIDATES, max(SEARCH_INITIAL_CANDIDATES, target_results * 3))
    last_filtered: List[Dict] = []
    last_count = -1

    while True:
        raw = ENGINE.search(query, book_filter=selected_books if selected_books else None, top_k=candidate_budget)
        filtered = _adaptive_relevance_filter(raw, min_score, query)
        last_filtered = filtered

        enough_results = len(filtered) >= target_results
        saturated = len(filtered) >= SEARCH_MAX_RESULTS
        exhausted = len(raw) < candidate_budget or candidate_budget >= SEARCH_MAX_CANDIDATES
        stalled = len(filtered) == last_count and enough_results
        if exhausted or saturated or stalled:
            break

        if enough_results and candidate_budget >= min(SEARCH_MAX_CANDIDATES, target_results * 6):
            break

        last_count = len(filtered)
        next_budget = min(SEARCH_MAX_CANDIDATES, max(candidate_budget * 2, candidate_budget + SEARCH_INITIAL_CANDIDATES))
        if next_budget == candidate_budget:
            break
        candidate_budget = next_budget

    _cache_search_results(cache_key, last_filtered)
    return last_filtered

def ensure_offline_assets():
    """Downloads static assets (CSS, JS, Fonts) for offline use."""
    # print("Checking offline assets...") # שקט יותר בהפעלה
    static_dir = STATIC_DIR
    os.makedirs(static_dir, exist_ok=True)
    
    assets = [
        ("bootstrap.min.css", "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"),
        ("bootstrap.bundle.min.js", "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"),
        ("bootstrap-icons.css", "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"),
        ("my_icon.png", "https://raw.githubusercontent.com/Otzaria/otzaria-library/master/resources/images/logo.png") # דוגמה לאייקון
    ]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    for filename, url in assets:
        path = os.path.join(static_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                
                content = r.content
                # Special handling for bootstrap-icons to download fonts
                if filename == "bootstrap-icons.css":
                    css_text = r.text
                    # Find font URLs (usually relative ./fonts/...)
                    font_urls = re.findall(r'url\("?(.+?)(?:\?|#|")\)', css_text)
                    fonts_dir = os.path.join(static_dir, "fonts")
                    os.makedirs(fonts_dir, exist_ok=True)
                    
                    for f_url in set(font_urls):
                        # Clean URL and construct full download URL
                        clean_f_url = f_url.strip("'\"")
                        if not clean_f_url.startswith("http") and not clean_f_url.startswith("data:"):
                            # Assuming standard structure relative to css
                            base_url = url.rsplit('/', 1)[0] + "/"
                            download_url = base_url + clean_f_url
                            f_name = os.path.basename(clean_f_url)
                            
                            f_path = os.path.join(fonts_dir, f_name)
                            if not os.path.exists(f_path):
                                print(f"Downloading font: {f_name}")
                                fr = requests.get(download_url, headers=headers)
                                with open(f_path, "wb") as f: f.write(fr.content)
                            
                            # Update CSS to point to local fonts folder
                            content = content.replace(f_url.encode(), f"fonts/{f_name}".encode())

                with open(path, "wb") as f:
                    f.write(content)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

    # Google Fonts Handling
    fonts_css_path = os.path.join(static_dir, "fonts.css")
    if not os.path.exists(fonts_css_path):
        print("Downloading Google Fonts...")
        try:
            gf_url = "https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=Frank+Ruhl+Libre:wght@400;700&family=Tinos:wght@400;700&family=Cardo:wght@400;700&display=swap"
            r = requests.get(gf_url, headers=headers)
            css = r.text
            
            webfonts_dir = os.path.join(static_dir, "webfonts")
            os.makedirs(webfonts_dir, exist_ok=True)
            
            urls = re.findall(r'url\((https://[^)]+)\)', css)
            for i, url in enumerate(urls):
                ext = url.split('.')[-1]
                if len(ext) > 5: ext = "woff2"
                fname = f"font_{i}.{ext}"
                fpath = os.path.join(webfonts_dir, fname)
                if not os.path.exists(fpath):
                    with open(fpath, "wb") as f: f.write(requests.get(url, headers=headers).content)
                css = css.replace(url, f"/static/webfonts/{fname}")
            
            with open(fonts_css_path, "w", encoding="utf-8") as f: f.write(css)
        except Exception as e: print(f"Fonts error: {e}")

def get_library_tree(db_path: str, indexed_ids: Optional[Set[int]] = None):
    """Builds a hierarchical tree of categories and books."""
    if not os.path.exists(db_path): return []
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        
        # 1. Categories
        categories = {}
        cursor = con.execute("PRAGMA table_info(category)")
        cat_cols = [c['name'] for c in cursor.fetchall()]
        name_col = next((c for c in ["name", "title"] if c in cat_cols), "id")
        
        for row in con.execute(f"SELECT id, {name_col} FROM category"):
            categories[row['id']] = {'id': row['id'], 'name': row[name_col], 'children': [], 'books': []}
        
        # 2. Books
        book_table = "book"
        try:
            con.execute("SELECT 1 FROM book LIMIT 1")
        except sqlite3.OperationalError:
            book_table = "books"
            
        cursor = con.execute(f"PRAGMA table_info({book_table})")
        book_cols = [c['name'] for c in cursor.fetchall()]
        cat_id_col = next((c for c in ["categoryId", "category_id"] if c in book_cols), "categoryId")

        for row in con.execute(f"SELECT id, title, {cat_id_col} FROM {book_table}"):
            bid, title, cid = row['id'], row['title'], row[cat_id_col]
            if indexed_ids is not None and bid not in indexed_ids:
                continue
            if cid in categories:
                categories[cid]['books'].append({'id': bid, 'title': title})
        
        # 3. Hierarchy (using category_closure depth=1 for direct children)
        child_ids = set()
        try:
            for row in con.execute("SELECT ancestor, descendant FROM category_closure WHERE depth = 1"):
                anc, des = row['ancestor'], row['descendant']
                if anc in categories and des in categories and anc != des:
                    categories[anc]['children'].append(categories[des])
                    child_ids.add(des)
        except sqlite3.OperationalError:
            # Fallback if category_closure is missing - use parentId if exists
            cursor = con.execute("PRAGMA table_info(category)")
            cols = [c['name'] for c in cursor.fetchall()]
            parent_col = next((c for c in ["parentId", "parent_id"] if c in cols), None)
            if parent_col:
                for row in con.execute(f"SELECT id, {parent_col} FROM category WHERE {parent_col} IS NOT NULL"):
                    curr_id, p_id = row['id'], row[parent_col]
                    if p_id in categories and curr_id in categories and p_id != curr_id:
                        categories[p_id]['children'].append(categories[curr_id])
                        child_ids.add(curr_id)
        
        con.close()
        
        def prune_empty_categories(cat):
            """Recursively removes categories that have no books and no sub-categories with books."""
            cat['children'] = [child for child in cat['children'] if prune_empty_categories(child)]
            return len(cat['books']) > 0 or len(cat['children']) > 0

        # Roots are categories that are not children of anyone
        roots = [c for cid, c in categories.items() if cid not in child_ids]
        if indexed_ids is not None:
            roots = [c for c in roots if prune_empty_categories(c)]
            
        # Sort alphabetically
        roots.sort(key=lambda x: str(x['name'] or ""))
        return roots
    except Exception as e:
        print(f"Error building tree: {e}")
        return []
@app.template_filter('highlight')
def highlight_filter(text, query): return highlight_text(text, query)

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    book_ids_str = request.args.getlist("book_id")
    selected_books = [int(b) for b in book_ids_str if b.isdigit()]
    page = max(1, request.args.get("page", default=1, type=int) or 1)

    cfg = ENGINE.last_cfg or {}
    # Initialize status variables at the very top to avoid UnboundLocalError
    current_db = cfg.get("db_path", DEFAULT_DB_PATH)
    db_exists = os.path.exists(current_db)
    model_loaded = (ENGINE.model is not None)
    model_expected = model_loaded or has_model_source_available(cfg)
    show_boot_loading = db_exists and model_expected and ENGINE.status["state"] != "error"

    top_k = int(cfg.get("top_k", DEFAULT_TOP_K))
    min_score = float(cfg.get("min_score", DEFAULT_MIN_SCORE)) / 100.0

    results = []
    total_results = 0
    total_pages = 0
    did_you_mean = None
    pagination_pages: List[int] = []
    expanded_query = q  # הוספנו את האתחול כאן כדי למנוע את השגיאה

    # סינון רשימת הספרים: רק מה שקיים באינדקס
    indexed_ids = ENGINE.get_indexed_book_ids()
    all_books = {bid: title for bid, title in ENGINE.book_map.items() if not indexed_ids or bid in indexed_ids}
    
    sorted_books = dict(sorted(all_books.items(), key=lambda item: item[1])[:800])
    index_book_ids = [int(b) for b in cfg.get("index_book_ids", []) if str(b).isdigit()]

    if q:
        if not ENGINE.built and ENGINE.status["state"] not in ("indexing", "downloading"):
            def task():
                try:
                    ENGINE.load_resources(current_db, cfg.get("edition", "v3"), model_source=cfg.get("model_source", "zip"), zip_path=cfg.get("zip_path", ""))
                    ENGINE.build_index(
                        current_db, 
                        int(cfg.get("max_chunks", 100000)),
                        int(cfg.get("ideal_chunk_words", IDEAL_CHUNK_WORDS)),
                        int(cfg.get("max_chunk_words", MAX_CHUNK_WORDS)),
                        int(cfg.get("overlap_words", DEFAULT_OVERLAP_WORDS)),
                        index_book_ids,
                    )
                except Exception as e: ENGINE._update("error", f"שגיאה: {e}", 0)
            threading.Thread(target=task, daemon=True).start()
        else:
            # בדיקת איות (Spell Check)
            correction = ENGINE.check_spelling(q)
            if correction and correction != clean_text(q):
                did_you_mean = correction
            # חילוץ מילים להדגשה (כולל מילים נרדפות מההרחבה)
            expanded_query = q
            if ENGINE.model:
                expanded_terms = ENGINE.get_expanded_terms(clean_text(q))
                expanded_query = " ".join(expanded_terms)

            filtered = _collect_filtered_results(q, selected_books, min_score, top_k, page)
            total_results = len(filtered)
            total_pages = max(1, math.ceil(total_results / top_k)) if total_results else 0
            page = min(page, total_pages) if total_pages else 1
            start_idx = (page - 1) * top_k
            end_idx = start_idx + top_k
            results = filtered[start_idx:end_idx]

            if total_pages <= 7:
                pagination_pages = list(range(1, total_pages + 1))
            elif total_pages:
                window_start = max(1, page - 2)
                window_end = min(total_pages, page + 2)
                pagination_pages = sorted({1, 2, total_pages - 1, total_pages, *range(window_start, window_end + 1)})

    idx_c = ENGINE.built.count if ENGINE.built else 0

    return render_template_string(
        HTML_TEMPLATE, model_loaded=model_loaded, db_exists=db_exists, query=q, did_you_mean=did_you_mean, 
        results=results, db_path=current_db, edition=cfg.get("edition", "v3"), max_chunks=int(cfg.get("max_chunks", 100000)), 
        model_source=cfg.get("model_source", "zip"), zip_path=cfg.get("zip_path", ""), idx_count=idx_c, books=sorted_books, 
        selected_books=selected_books, top_k=top_k, min_score=int(cfg.get("min_score", 0)),
        page=page, total_results=total_results, total_pages=total_pages, pagination_pages=pagination_pages,
        model_expected=model_expected, show_boot_loading=show_boot_loading,
        ideal_chunk_words=int(cfg.get("ideal_chunk_words", IDEAL_CHUNK_WORDS)),
        max_chunk_words=int(cfg.get("max_chunk_words", MAX_CHUNK_WORDS)),
        overlap_words=int(cfg.get("overlap_words", DEFAULT_OVERLAP_WORDS)),
        w_vec=cfg.get("w_vec", 0.35), w_bm=cfg.get("w_bm", 0.25), w_overlap=cfg.get("w_overlap", 0.25), 
        w_phrase=cfg.get("w_phrase", 0.10), w_proximity=cfg.get("w_proximity", 0.05),
        decay_2=cfg.get("decay_2", 0.75), decay_3=cfg.get("decay_3", 0.35), decay_4=cfg.get("decay_4", 0.05),
        expanded_query=expanded_query, index_book_ids=index_book_ids
    )
@app.route("/api/autocomplete")
def autocomplete():
    """השלמה אוטומטית פשוטה: השלמת מילים בודדות מתוך המילון"""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2: return jsonify([])

    clean_q = clean_text(q)
    if not clean_q: return jsonify([])

    words = clean_q.split()
    last_word = words[-1]
    prefix = " ".join(words[:-1])
    
    suggestions = []
    if ENGINE.model:
        # שימוש ב-bisect לחיפוש מהיר ברשימה ממוינת
        start_idx = bisect.bisect_left(ENGINE.model.sorted_vocab, last_word)
        candidates = []
        # איסוף עד 50 מועמדים שמתחילים בקידומת
        for i in range(start_idx, min(start_idx + 50, len(ENGINE.model.sorted_vocab))):
            w = ENGINE.model.sorted_vocab[i]
            if not w.startswith(last_word):
                break
            candidates.append(w)
        
        # מיון לפי שכיחות (הנפוץ ביותר קודם)
        candidates.sort(key=lambda w: -ENGINE.model.word_freqs.get(w, 0))
        
        # החזרת 10 התוצאות הטובות ביותר
        for w in candidates[:10]:
            if prefix:
                suggestions.append(f"{prefix} {w}")
            else:
                suggestions.append(w)

    return jsonify(suggestions)

@app.route("/api/get_tree")
def get_tree_api():
    """API endpoint to fetch the library tree asynchronously."""
    cfg = ENGINE.last_cfg or {}
    db_path = cfg.get("db_path", DEFAULT_DB_PATH)
    show_all = request.args.get("all", "").lower() in ("1", "true", "yes")
    indexed_ids = None if show_all else ENGINE.get_indexed_book_ids()
    tree = get_library_tree(db_path, indexed_ids if indexed_ids else None)
    return jsonify(tree)

@app.route("/api/get_context")
def get_context():
    """מחזיר שורות מסביב לשורה נבחרת כדי להציג הקשר רחב יותר."""
    book_id = request.args.get("book_id", type=int)
    start_line = request.args.get("line_idx", type=int)
    end_line = request.args.get("end_line", type=int)
    query = request.args.get("q", "")
    
    if book_id is None or start_line is None:
        return jsonify({"error": "Missing parameters"}), 400

    cfg = ENGINE.last_cfg or {}
    db_path = cfg.get("db_path", DEFAULT_DB_PATH)
    
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        table_name = "line"
        try:
            con.execute("SELECT 1 FROM lines LIMIT 1")
            table_name = "lines"
        except: pass

        offset = 20  # מספר השורות להצגה לפני ואחרי
        rows = con.execute(
            f"SELECT lineIndex, content FROM {table_name} WHERE bookId = ? AND lineIndex BETWEEN ? AND ? ORDER BY lineIndex",
            (book_id, start_line - offset, (end_line or start_line) + offset)
        ).fetchall()
        con.close()

        html_lines = []
        for r in rows:
            txt = highlight_text(r["content"], query)
            is_target = start_line <= r["lineIndex"] <= (end_line or start_line)
            cls = "highlight-target" if is_target else ""
            html_lines.append(f'<div class="mb-2 serif-text fs-5 {cls}">{txt}</div>')
        
        return jsonify({
            "html": "".join(html_lines),
            "title": ENGINE.book_map.get(book_id, "הקשר")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/feedback", methods=["POST"])
def feedback():
    """למידת מכונה (ML) שמכוונת את המערכת לפי פידבק בזמן אמת"""
    data = request.json
    rating = float(data.get("rating", 0))
    feats = data.get("features", {})
    if rating == 0 or not feats: return jsonify({"status": "error"})
        
    cfg = load_settings()
    weights = {k: float(cfg.get(k, v)) for k, v in [("w_vec", 0.35), ("w_bm", 0.25), ("w_overlap", 0.25), ("w_phrase", 0.10), ("w_proximity", 0.05)]}
    
    lr = 0.05 
    weights["w_vec"] = max(0.01, weights["w_vec"] + lr * rating * feats.get("vec", 0))
    weights["w_bm"] = max(0.01, weights["w_bm"] + lr * rating * feats.get("bm", 0))
    weights["w_overlap"] = max(0.01, weights["w_overlap"] + lr * rating * feats.get("overlap", 0))
    weights["w_phrase"] = max(0.01, weights["w_phrase"] + lr * rating * feats.get("phrase", 0))
    weights["w_proximity"] = max(0.01, weights["w_proximity"] + lr * rating * feats.get("prox", 0))
    
    # Update decay factors if proximity was relevant
    prox_val = feats.get("prox", 0)
    if prox_val > 0.1:
        change = lr * rating * prox_val * 0.2
        cfg["decay_2"] = min(1.0, max(0.0, float(cfg.get("decay_2", 0.75)) + change))
        cfg["decay_3"] = min(1.0, max(0.0, float(cfg.get("decay_3", 0.35)) + change))
        cfg["decay_4"] = min(1.0, max(0.0, float(cfg.get("decay_4", 0.05)) + change))

    total = sum(weights.values())
    for k in weights: cfg[k] = round(weights[k] / total, 3)
        
    save_settings(cfg)
    ENGINE.last_cfg = cfg
    return jsonify({"status": "ok", "new_weights": cfg})

@app.route("/setup", methods=["POST"])
def setup():
    cfg = load_settings()
    
    # שמירת הערכים הישנים להשוואה
    old_db = cfg.get("db_path", DEFAULT_DB_PATH)
    old_edition = cfg.get("edition", "v3")
    old_max_chunks = int(cfg.get("max_chunks", 100000))
    old_source = cfg.get("model_source", "zip")
    old_zip = cfg.get("zip_path", "")
    old_ideal = int(cfg.get("ideal_chunk_words", IDEAL_CHUNK_WORDS))
    old_max_w = int(cfg.get("max_chunk_words", MAX_CHUNK_WORDS))
    old_overlap = int(cfg.get("overlap_words", DEFAULT_OVERLAP_WORDS))
    old_index_book_ids = sorted({int(b) for b in cfg.get("index_book_ids", []) if str(b).isdigit()})

    # קריאת הערכים החדשים
    new_db = request.form.get("db_path", DEFAULT_DB_PATH).strip()
    new_edition = request.form.get("edition", "v3").strip()
    new_max_chunks = int(request.form.get("max_chunks", 100000))
    new_source = request.form.get("model_source", "zip").strip()
    new_zip = request.form.get("zip_path", "").strip()
    new_ideal = int(request.form.get("ideal_chunk_words", IDEAL_CHUNK_WORDS))
    new_max_w = int(request.form.get("max_chunk_words", MAX_CHUNK_WORDS))
    new_overlap = int(request.form.get("overlap_words", DEFAULT_OVERLAP_WORDS))
    selection_present = request.form.get("index_selection_present", "0") == "1"
    total_index_books = request.form.get("index_total_books", type=int) or 0
    if selection_present:
        submitted_index_ids = sorted({int(b) for b in request.form.getlist("index_book_id") if str(b).isdigit()})
        if total_index_books and (len(submitted_index_ids) == 0 or len(submitted_index_ids) == total_index_books):
            new_index_book_ids = []
        else:
            new_index_book_ids = submitted_index_ids
    else:
        new_index_book_ids = old_index_book_ids

    cfg.update({
        "db_path": new_db,
        "edition": new_edition,
        "max_chunks": new_max_chunks,
        "model_source": new_source,
        "zip_path": new_zip,
        "index_book_ids": new_index_book_ids,
        "ideal_chunk_words": new_ideal,
        "max_chunk_words": new_max_w,
        "overlap_words": new_overlap,
        "top_k": int(request.form.get("top_k", 20)),
        "min_score": float(request.form.get("min_score", 0)),
        "w_vec": float(request.form.get("w_vec", 0.35)),
        "w_bm": float(request.form.get("w_bm", 0.25)),
        "w_overlap": float(request.form.get("w_overlap", 0.25)),
        "w_phrase": float(request.form.get("w_phrase", 0.10)),
        "w_proximity": float(request.form.get("w_proximity", 0.05)),
        "decay_2": float(request.form.get("decay_2", 0.75)),
        "decay_3": float(request.form.get("decay_3", 0.35)),
        "decay_4": float(request.form.get("decay_4", 0.05)),
    })
    save_settings(cfg)
    ENGINE.last_cfg = cfg
    ENGINE.search_cache.clear()
    if new_index_book_ids:
        ENGINE.log(f"נשמרו הגדרות. אינדוקס יוגבל ל-{len(new_index_book_ids):,} ספרים.", "info")
    else:
        ENGINE.log("נשמרו הגדרות. אינדוקס יכלול את כל הספרים.", "info")

    # בדיקה אם נדרש טעינה מחדש (אם השתנו פרמטרים מבניים)
    if (new_db != old_db or new_edition != old_edition or 
        new_max_chunks != old_max_chunks or new_source != old_source or new_zip != old_zip or
        new_ideal != old_ideal or new_max_w != old_max_w or new_overlap != old_overlap or
        new_index_book_ids != old_index_book_ids):
        
        def reload_task():
            try:
                # אם השתנה המודל או ה-DB (או שטרם נטען מודל), נטען משאבים מחדש
                if (not ENGINE.model or 
                    new_edition != old_edition or 
                    new_source != old_source or 
                    new_zip != old_zip or 
                    new_db != old_db):
                    ENGINE.load_resources(new_db, new_edition, new_source, new_zip)
                    ENGINE.library_tree = get_library_tree(new_db)
                
                # בניית אינדקס (או טעינה אם קיים)
                ENGINE.build_index(new_db, new_max_chunks, new_ideal, new_max_w, new_overlap, new_index_book_ids)
            except Exception as e:
                ENGINE._update("error", f"שגיאה בטעינה מחדש: {e}", 0)
        
        threading.Thread(target=reload_task, daemon=True).start()

    return redirect("/")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/status")
def status_api():
    s = ENGINE.status.copy()
    if ENGINE.built: s["count"] = ENGINE.built.count
    return jsonify(s)

@app.route("/api/logs")
def logs_api():
    limit = request.args.get("limit", default=200, type=int) or 200
    return jsonify({
        "entries": ENGINE.get_logs(limit=min(limit, 500)),
        "status": ENGINE.status.copy(),
        "count": ENGINE.built.count if ENGINE.built else 0,
    })

@app.route("/api/reset_index", methods=["POST"])
def reset_index():
    try:
        for fname in os.listdir(RUNTIME_DIR):
            if fname.endswith(".index") or fname.endswith(".sqlite") or fname.endswith(".tmp"):
                try:
                    os.remove(os.path.join(RUNTIME_DIR, fname))
                except OSError:
                    pass
        ENGINE.built = None
        ENGINE.log("האינדקס נמחק ואופס בהצלחה על ידי המשתמש.", "info")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

# Helper UI Routes
@app.route("/upload_model", methods=["POST"])
def upload_model():
    file = request.files.get('file')
    if file and file.filename:
        target = os.path.join(MODELS_ZIPS_DIR, secure_filename(file.filename))
        file.save(target)
        cfg = load_settings()
        cfg.update({"model_source": "zip", "zip_path": target})
        save_settings(cfg)
        ENGINE.last_cfg = cfg
        ENGINE.log(f"מודל ZIP חדש הועלה: {os.path.basename(target)}", "info")
        threading.Thread(target=lambda: (
            ENGINE.load_resources(cfg.get("db_path", DEFAULT_DB_PATH), cfg.get("edition", "v3"), "zip", target), 
            ENGINE.build_index(cfg.get("db_path", DEFAULT_DB_PATH), int(cfg.get("max_chunks", 100000)),
                               int(cfg.get("ideal_chunk_words", IDEAL_CHUNK_WORDS)),
                               int(cfg.get("max_chunk_words", MAX_CHUNK_WORDS)),
                               int(cfg.get("overlap_words", DEFAULT_OVERLAP_WORDS)),
                               [int(b) for b in cfg.get("index_book_ids", []) if str(b).isdigit()])
        )).start()
    return redirect("/")

@app.route("/download_db", methods=["POST"])
def download_db():
    def task():
        try:
            ENGINE._update("downloading", "מוריד מסד נתונים...", 0)
            zip_path = os.path.join(DB_DIR, "seforim.zip")
            with requests.get(DB_DOWNLOAD_URL, stream=True) as r:
                total_len = int(r.headers.get('content-length', 0))
                dl = 0
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            dl += len(chunk); f.write(chunk)
                            if total_len: ENGINE._update("downloading", "מוריד מסד נתונים...", int((dl/total_len)*100))
            ENGINE._update("indexing", "מחלץ...", 0)
            with zipfile.ZipFile(zip_path, "r") as z: z.extractall(DB_DIR)
            db_file = next((os.path.join(r, f) for r, d, files in os.walk(DB_DIR) for f in files if f.endswith((".db", ".sqlite"))), None)
            if db_file:
                cfg = load_settings()
                cfg["db_path"] = db_file
                save_settings(cfg); ENGINE.last_cfg = cfg
                ENGINE.log(f"מסד הנתונים עודכן ל-{db_file}", "info")
                ENGINE._update("ready", "הסתיים בהצלחה", 100)
        except Exception as e: ENGINE._update("error", str(e), 0)
    threading.Thread(target=task, daemon=True).start()
    return redirect("/")

@app.route("/upload_db", methods=["POST"])
def upload_db():
    file = request.files.get('file')
    if file and file.filename:
        target = os.path.join(DB_DIR, secure_filename(file.filename))
        file.save(target)
        def task():
            try:
                db_file = target
                if target.lower().endswith(".zip"):
                    ENGINE._update("indexing", "מחלץ...", 0)
                    with zipfile.ZipFile(target, 'r') as z: z.extractall(DB_DIR)
                    db_file = next((os.path.join(r, f) for r, d, files in os.walk(DB_DIR) for f in files if f.endswith((".db", ".sqlite"))), None)
                if db_file:
                    cfg = load_settings(); cfg["db_path"] = db_file; save_settings(cfg); ENGINE.last_cfg = cfg
                    ENGINE.update_book_map(get_book_titles(db_file))
                    if cfg.get("model_source"):
                        ENGINE.load_resources(db_file, cfg.get("edition", "v3"), cfg["model_source"], cfg.get("zip_path", ""))
                        ENGINE.build_index(db_file, int(cfg.get("max_chunks", 100000)),
                                           int(cfg.get("ideal_chunk_words", IDEAL_CHUNK_WORDS)),
                                           int(cfg.get("max_chunk_words", MAX_CHUNK_WORDS)),
                                           int(cfg.get("overlap_words", DEFAULT_OVERLAP_WORDS)),
                                           [int(b) for b in cfg.get("index_book_ids", []) if str(b).isdigit()])
            except Exception as e: ENGINE._update("error", str(e), 0)
        threading.Thread(target=task).start()
    return redirect("/")

@app.route("/api/browse_db")
def browse_db_api():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="בחר מסד נתונים", filetypes=[("DB", "*.db *.sqlite"), ("ZIP", "*.zip"), ("All", "*.*")])
    root.destroy()
    return jsonify({"path": path})

@app.route("/api/browse_zip")
def browse_zip_api():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="בחר מודל", filetypes=[("ZIP", "*.zip"), ("All", "*.*")])
    root.destroy()
    return jsonify({"path": path})

@app.route("/select_local_db")
def select_local_db():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="בחר מסד נתונים", filetypes=[("DB", "*.db *.sqlite"), ("ZIP", "*.zip")])
    root.destroy()
    if path:
        cfg = load_settings(); cfg["db_path"] = path; save_settings(cfg); ENGINE.last_cfg = cfg
    return redirect("/")

@app.route("/select_local_zip")
def select_local_zip():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="בחר מודל", filetypes=[("ZIP", "*.zip")])
    root.destroy()
    if path:
        cfg = load_settings(); cfg.update({"model_source": "zip", "zip_path": path}); save_settings(cfg); ENGINE.last_cfg = cfg
    return redirect("/")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    cfg = load_settings()
    ENGINE.last_cfg = cfg
    initial_db_path = cfg.get("db_path", DEFAULT_DB_PATH)
    if os.path.exists(initial_db_path) and has_model_source_available(cfg):
        ENGINE._update("loading", "מכין את מנוע החיפוש...", 1)

    def boot():
        try:
            ENGINE.load_resources(initial_db_path, cfg.get("edition", "v3"), cfg.get("model_source", "zip"), cfg.get("zip_path", ""))
            ENGINE.build_index(initial_db_path, int(cfg.get("max_chunks", 100000)),
                               int(cfg.get("ideal_chunk_words", IDEAL_CHUNK_WORDS)),
                               int(cfg.get("max_chunk_words", MAX_CHUNK_WORDS)),
                               int(cfg.get("overlap_words", DEFAULT_OVERLAP_WORDS)),
                               [int(b) for b in cfg.get("index_book_ids", []) if str(b).isdigit()])
        except Exception as e: ENGINE._update("error", f"שגיאה בהפעלה: {e}", 0)

    is_frozen = getattr(sys, 'frozen', False)
    
    # Only start the heavy boot process if we are not in the reloader's main process (to avoid double loading)
    if is_frozen or os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        ensure_offline_assets()
        threading.Thread(target=boot, daemon=True).start()

    print("Starting DeepSearch Google-like AI at http://127.0.0.1:8000")
    # use_reloader=False is another option, but checking WERKZEUG_RUN_MAIN is safer if you want debug features
    if is_frozen:
        app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
    else:
        app.run(host="127.0.0.1", port=8000, debug=True)
