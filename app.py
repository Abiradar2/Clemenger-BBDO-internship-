# app.py
# Full Streamlit app with:
# - DB/FAISS bootstrap (via your init module)
# - DB status panel
# - Browse/Search (safe; one video thumbnail per video; skips missing files)
# - Upload page that captures a folder path for NEW files (relative or absolute)
#   and stores it to a variable + text file for later use.
#   IMPORTANT: It WILL ERROR if the path does not already exist (no auto-create).

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import gc
import os

import streamlit as st

# ---- Optional Upload helper (don't break app if missing) ----
try:
    from Upload import list_files_once
except Exception:
    def list_files_once(p: str):
        root = Path(p)
        if root.exists() and root.is_dir():
            for sub in sorted(root.rglob("*")):
                yield str(sub)
        else:
            yield f"[warn] path not found: {p}"

# Tame threaded libs for Streamlit servers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# =======================================
# IMPORT YOUR INIT (db + faiss handles)
# =======================================
from init_logs_bds import get_handles  # or: from init_logs_dbs import get_handles

# =======================================
# PATH CONSTANTS & HELPERS
# =======================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data"
CONFIG_DIR   = DATA_ROOT / "config"
CREATIVE_SRC_JSON = CONFIG_DIR / "creative_source.json"

# Expose workspace defaults for future scripts
INPUT_DIR    = DATA_ROOT / "input"
WS_DIR       = DATA_ROOT / "workspace"
IMAGES_DIR   = WS_DIR / "images"
VIDEOS_DIR   = WS_DIR / "videos"

# --- capture the front-end path for later use ---
NEW_MEDIA_SOURCE_PATH: Optional[Path] = None
NEW_MEDIA_PATH_TXT = CONFIG_DIR / "new_media_source_path.txt"

def _read_creative_source() -> Optional[Dict]:
    try:
        if CREATIVE_SRC_JSON.exists():
            return json.loads(CREATIVE_SRC_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _write_creative_source(payload: Dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CREATIVE_SRC_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _resolve_source_path(mode: str, folder_input: str) -> Tuple[Optional[Path], Optional[str]]:
    folder_input = folder_input.strip()
    if not folder_input:
        return None, "Please enter a folder name/path."
    if mode == "Relative to app.py":
        return (PROJECT_ROOT / folder_input).resolve(), None
    # Absolute path mode
    p = Path(folder_input).expanduser()
    if not p.is_absolute():
        return None, "Absolute mode selected, but the path is not absolute."
    return p.resolve(), None

def _set_new_media_path(p: Path):
    p = p.resolve()
    globals()["NEW_MEDIA_SOURCE_PATH"] = p
    st.session_state["new_media_source_path"] = str(p)
    try:
        NEW_MEDIA_PATH_TXT.parent.mkdir(parents=True, exist_ok=True)
        NEW_MEDIA_PATH_TXT.write_text(str(p), encoding="utf-8")
    except Exception:
        pass

# =======================================
# BOOTSTRAP DBs & INDICES (cached)
# =======================================
@st.cache_resource(show_spinner=False)
def _init_handles() -> Tuple:
    """
    Returns:
      con: sqlite3.Connection
      indices: dict(level -> faiss index)
      snapshot: dict with paths and totals (from init module)
    """
    con, indices, snapshot = get_handles()
    return con, indices, snapshot

con, indices, snapshot = _init_handles()

# =======================================
# DB INTROSPECTION HELPERS
# =======================================
EXPECTED_TABLES = [
    "assets",
    "frames",
    "segments",
    "metadata",
    "special_tags",
    "tags",
    "tag_links",
    "ingestion_log",
]

def list_tables() -> List[str]:
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []

def table_count(name: str) -> int:
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
    except Exception:
        return -1  # -1 means table missing or error

def get_tables_report() -> List[Dict]:
    existing = set(list_tables())
    report = []
    for t in EXPECTED_TABLES:
        cnt = table_count(t) if t in existing else -1
        report.append({
            "table": t,
            "exists": t in existing,
            "rows": cnt if cnt >= 0 else 0
        })
    for t in sorted(existing - set(EXPECTED_TABLES)):
        cnt = table_count(t)
        report.append({
            "table": t,
            "exists": True,
            "rows": max(cnt, 0)
        })
    return report

# =======================================
# MEDIA HELPERS (SAFE)
# =======================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def _iter_image_files_under(root: Path, max_per_creative: int = 24) -> List[Path]:
    files: List[Path] = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
                if len(files) >= max_per_creative:
                    break
    return files

def _safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

# =======================================
# FETCH FUNCTIONS (WORKING)
# =======================================
def fetch_creatives() -> List[str]:
    """All creative folder names from assets.path, sorted."""
    try:
        rows = con.execute("SELECT path FROM assets ORDER BY path").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []

def fetch_media(creative: Optional[str] = None, media_type: str = "Both", limit: int = 500) -> List[Dict]:
    """
    Build a mixed list: images (actual files) and ONE video cover (earliest frame).
    Skips any missing files to avoid Streamlit MediaFileStorageError.
    Searches both workspace (data/workspace/...) and input (data/input/...) trees.
    """
    mm = {"Both": ("image","video"), "Images": ("image",), "Videos": ("video",)}
    want = mm.get(media_type, ("image","video"))

    if creative is None:
        rows = con.execute(
            "SELECT id, path, media_type FROM assets WHERE media_type IN (?,?) ORDER BY path",
            want if len(want) == 2 else (want[0], want[0])
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT id, path, media_type FROM assets WHERE path=? AND media_type IN (?,?)",
            (creative,) + (want if len(want) == 2 else (want[0], want[0]))
        ).fetchall()

    out: List[Dict] = []

    for asset_id, path_str, mtype in rows:
        if mtype == "image":
            ws_dir = (IMAGES_DIR / path_str).resolve()
            in_dir = (INPUT_DIR / "images" / path_str).resolve()
            files = _iter_image_files_under(ws_dir, 24) or _iter_image_files_under(in_dir, 24)
            if not files:
                continue

            industry = product = description = ""
            try:
                row = con.execute(
                    "SELECT COALESCE(industry,''), COALESCE(product,''), COALESCE(description,'') "
                    "FROM images_desc WHERE asset_id=?",
                    (asset_id,)
                ).fetchone()
                if row: industry, product, description = row
            except Exception:
                pass

            tags: List[str] = []
            try:
                tags = [t for (t,) in con.execute(
                    "SELECT tag FROM images_tags WHERE asset_id=? ORDER BY tag", (asset_id,)
                ).fetchall()]
            except Exception:
                pass

            for p in files:
                if not _safe_exists(p):
                    continue
                out.append({
                    "kind": "image",
                    "asset_id": asset_id,
                    "creative": path_str,
                    "file": str(p),
                    "caption": " · ".join([x for x in [path_str, industry or None, product or None] if x]),
                    "tags": tags,
                    "description": description,
                    "labels": {"industry": industry, "product": product},
                })

        elif mtype == "video":
            # earliest frame = cover
            try:
                row = con.execute(
                    "SELECT frame_no, path FROM frames WHERE asset_id=? ORDER BY frame_no ASC LIMIT 1",
                    (asset_id,)
                ).fetchone()
            except Exception:
                row = None

            if not row:
                continue

            frame_no, rel_path = row
            # Try several candidates: workspace primary/alt, then input primary/alt
            ws_primary = (VIDEOS_DIR / rel_path).resolve()
            ws_alt     = (VIDEOS_DIR / path_str / Path(rel_path).name).resolve()
            in_primary = (INPUT_DIR / "videos" / rel_path).resolve()
            in_alt     = (INPUT_DIR / "videos" / path_str / Path(rel_path).name).resolve()
            thumb = next((p for p in [ws_primary, ws_alt, in_primary, in_alt] if _safe_exists(p)), None)
            if not thumb:
                continue  # skip missing

            # video labels
            industry = product = ""
            try:
                vdesc = con.execute(
                    "SELECT COALESCE(industry,''), COALESCE(product,'') FROM videos_desc WHERE asset_id=?",
                    (asset_id,)
                ).fetchone()
                if vdesc: industry, product = vdesc
            except Exception:
                pass

            # stats (best-effort)
            try:
                fc = con.execute("SELECT COUNT(*) FROM frames WHERE asset_id=?", (asset_id,)).fetchone()[0]
            except Exception:
                fc = 0
            try:
                f_emb = con.execute(
                    "SELECT COALESCE(SUM(has_embedding),0) FROM frames WHERE asset_id=?",
                    (asset_id,)
                ).fetchone()[0]
            except Exception:
                f_emb = 0

            caption_bits = [path_str, f"f{frame_no}"]
            if industry: caption_bits.append(industry)
            if product:  caption_bits.append(product)

            out.append({
                "kind": "video",
                "asset_id": asset_id,
                "creative": path_str,
                "file": str(thumb),
                "caption": " · ".join(caption_bits),
                "tags": [],
                "description": "",
                "video_stats": {
                    "frames_total": int(fc),
                    "frames_with_embeddings": int(f_emb),
                },
                "labels": {"industry": industry, "product": product},
            })

        if len(out) >= limit:
            break

    return out[:limit]

# =======================================
# RENDERER (SAFE)
# =======================================
def grid_gallery(items: List[Dict], cols: int = 5):
    if not items:
        st.info("No media found for the current selection.")
        return

    cols_list = st.columns(cols)
    for i, it in enumerate(items):
        c = cols_list[i % cols]
        p = Path(it["file"])
        with c:
            if _safe_exists(p):
                st.image(str(p), use_container_width=True, caption=it.get("caption", ""))
            else:
                st.caption(it.get("caption", ""))
                st.warning(f"File not found: {p.name}")

            with st.expander("Details", expanded=False):
                st.text(f"Creative: {it.get('creative','')}")
                if it.get("kind") == "video":
                    vs = it.get("video_stats", {})
                    st.text(f"Frames: {vs.get('frames_total', 0)} (emb: {vs.get('frames_with_embeddings', 0)})")
                lbl = it.get("labels", {})
                if lbl.get("industry") or lbl.get("product"):
                    st.text("Labels: " + " · ".join([x for x in [lbl.get('industry'), lbl.get('product')] if x]))
                if it.get("tags"):
                    st.text("Tags: " + ", ".join(it["tags"]))
                if it.get("description"):
                    st.text("Desc: " + it["description"])

# =======================================
# SIMPLE FILTERS (Search)
# =======================================
def _contains(hay: str, needle: str) -> bool:
    return (needle or "").lower() in (hay or "").lower()

def filename_matches(q: str, m: Dict) -> bool:
    return (not q) or _contains(Path(m.get("file","")).name, q)

def description_matches(q: str, m: Dict) -> bool:
    return (not q) or _contains(m.get("description",""), q)

def tags_match(tags_q: str, m: Dict, mode: str = "any") -> bool:
    if not tags_q:
        return True
    want = [t.strip().lower() for t in tags_q.split(",") if t.strip()]
    have = [t.lower() for t in m.get("tags", [])]
    if not want:
        return True
    if mode == "all":
        return all(t in have for t in want)
    return any(t in have for t in want)

def filter_media(items: List[Dict], filename_q: str, desc_q: str, tags_q: str, tag_mode: str) -> List[Dict]:
    out = []
    for m in items:
        if not filename_matches(filename_q, m):    continue
        if not description_matches(desc_q, m):     continue
        if not tags_match(tags_q, m, tag_mode):    continue
        out.append(m)
    return out

def prompt_search_stub(q: str, pool: List[Dict]) -> List[Dict]:
    """Very simple placeholder: match any word in caption/desc."""
    if not q:
        return pool
    toks = [t for t in q.lower().split() if t]
    def _ok(m: Dict) -> bool:
        hay = (m.get("caption","") + " " + m.get("description","")).lower()
        return any(t in hay for t in toks)
    return [m for m in pool if _ok(m)]

# =======================================
# SEARCH LIFECYCLE HELPERS
# =======================================
SEARCH_PREFIX = "search_"

def _enter_search_mode():
    st.session_state[SEARCH_PREFIX + "active"] = True
    st.session_state.setdefault(SEARCH_PREFIX + "prompt_q", "")
    st.session_state.setdefault(SEARCH_PREFIX + "tags_q", "")
    st.session_state.setdefault(SEARCH_PREFIX + "desc_q", "")
    st.session_state.setdefault(SEARCH_PREFIX + "filename_q", "")
    st.session_state.setdefault(SEARCH_PREFIX + "tag_mode", "any")
    st.session_state.setdefault(SEARCH_PREFIX + "media_type", "Both")

def _exit_search_mode():
    # Drop all search_* keys so they don't linger
    for k in list(st.session_state.keys()):
        if str(k).startswith(SEARCH_PREFIX):
            try:
                st.session_state.pop(k, None)
            except Exception:
                pass
    try:
        gc.collect()
    except Exception:
        pass

# =======================================
# STREAMLIT APP (UI)
# =======================================
st.set_page_config(page_title="Creative Browser", layout="wide")
st.title("Creative Browser & Search")

# Track page switches to run enter/exit hooks
with st.sidebar:
    st.header("Navigate")
    current_page = st.radio("Pages", ["Browse", "Search", "Upload", "App usage"], index=0, key="nav_page")

# Run Search lifecycle hooks based on page transitions
prev_page = st.session_state.get("prev_page")
if prev_page != current_page:
    if prev_page == "Search":
        _exit_search_mode()
    if current_page == "Search":
        _enter_search_mode()
    st.session_state["prev_page"] = current_page

# Sidebar extras (status panels)
with st.sidebar:
    st.divider()
    st.header("DB status")
    st.caption("Initialized via init_logs_bds.get_handles()")

    snap_sqlite = snapshot.get("paths", {}).get("sqlite", "<unknown>")
    snap_faiss  = snapshot.get("paths", {}).get("faiss", {})

    st.code(
        "SQLite: " + str(snap_sqlite) + "\n"
        + "\n".join([f"FAISS[{lvl}]: {p}" for lvl, p in (snap_faiss or {}).items()]),
        language="bash",
    )

    totals = snapshot.get("totals", {}) or {}
    st.metric("Assets", totals.get("assets", 0))
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Images", totals.get("images", 0))
    with c2: st.metric("Videos", totals.get("videos", 0))
    with c3: st.metric("Creatives", totals.get("creatives", 0))

    st.divider()
    st.header("DB tables & counts")
    report = get_tables_report()
    for row in report:
        label = f"{row['table']}"
        value = row["rows"] if row["exists"] else "missing"
        st.text(f"{label:>14}: {value}")

page = current_page

# -------------------------
# BROWSE
# -------------------------
if page == "Browse":
    st.subheader("Browse creatives")
    creatives = fetch_creatives()
    selected = st.selectbox("Creative", ["<All creatives>"] + creatives, index=0)
    media_type = st.selectbox("Media type", ["Both", "Images", "Videos"], index=0)

    media = fetch_media(None if selected == "<All creatives>" else selected, media_type)
    st.write(f"Showing {len(media)} items")
    grid_gallery(media, cols=5)

# -------------------------
# SEARCH
# -------------------------
elif page == "Search":
    st.subheader("Search creatives")

    with st.form("search_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt_q = st.text_input(
                "Prompt (placeholder, not CLIP yet)",
                value=st.session_state.get(SEARCH_PREFIX + "prompt_q", ""),
                placeholder="e.g., 'phone on a desk'",
                key=SEARCH_PREFIX + "prompt_q_input",
            )
            tags_q   = st.text_input(
                "Tags (comma separated)",
                value=st.session_state.get(SEARCH_PREFIX + "tags_q", ""),
                placeholder="e.g., 'logo, car, outdoor'",
                key=SEARCH_PREFIX + "tags_q_input",
            )
            desc_q   = st.text_input(
                "Description contains",
                value=st.session_state.get(SEARCH_PREFIX + "desc_q", ""),
                placeholder="e.g., 'father and child'",
                key=SEARCH_PREFIX + "desc_q_input",
            )
        with col2:
            filename_q = st.text_input(
                "Filename contains",
                value=st.session_state.get(SEARCH_PREFIX + "filename_q", ""),
                placeholder="partial filename…",
                key=SEARCH_PREFIX + "filename_q_input",
            )
            tag_mode   = st.selectbox(
                "Tag match mode",
                ["any", "all"],
                index=0 if st.session_state.get(SEARCH_PREFIX + "tag_mode", "any") == "any" else 1,
                key=SEARCH_PREFIX + "tag_mode_select",
            )
            media_type = st.selectbox(
                "Media type",
                ["Both", "Images", "Videos"],
                index=["Both","Images","Videos"].index(st.session_state.get(SEARCH_PREFIX + "media_type", "Both")),
                key=SEARCH_PREFIX + "media_type_select",
            )

        submitted = st.form_submit_button("Search")

    if submitted:
        # pool media, apply simple filters and prompt stub
        base_pool = fetch_media(None, media_type, limit=2000)
        pool = filter_media(base_pool, filename_q, desc_q, tags_q, tag_mode)
        results = prompt_search_stub(prompt_q, pool)
        st.success(f"Found {len(results)} results")
        grid_gallery(results, cols=5)

# -------------------------
# UPLOAD (front-end picks the new files path; ERROR if path doesn't exist)
# -------------------------
elif page == "Upload":
    st.subheader("Creatives source folder (for later ingest)")

    if "creative_source" not in st.session_state:
        st.session_state.creative_source = _read_creative_source()

    st.markdown("#### Set the folder where **new files** are located")
    mode = st.radio(
        "How do you want to specify it?",
        ["Relative to app.py", "Absolute path"],
        index=0
    )

    if mode == "Relative to app.py":
        rel_hint = "e.g., 'creatives/new_batch', 'data/input/campaign_oct'"
        new_folder_input = st.text_input("Relative folder (from app.py)", value="creatives", help=rel_hint)
    else:
        abs_hint = "e.g., '/Users/you/Projects/myrepo/creatives/new_batch'"
        new_folder_input = st.text_input("Absolute folder path", value="", placeholder=abs_hint)

    # We DO NOT create missing paths. We error if it doesn't exist.
    if st.button("Use this folder for NEW files"):
        resolved, err = _resolve_source_path(mode, new_folder_input)
        if err:
            st.error(err)
        else:
            if not resolved.exists():
                st.error(f"Path does not exist: {resolved}")
                st.stop()
            if not resolved.is_dir():
                st.error(f"Path exists but is not a directory: {resolved}")
                st.stop()

            try:
                payload = {"input_mode": mode, "input_value": new_folder_input, "resolved_path": str(resolved)}
                _write_creative_source(payload)
                _set_new_media_path(resolved)

                # Preview: stream yielded names
                output_area = st.empty()
                for msg in list_files_once(str(resolved)):
                    output_area.code(str(msg), language="bash")

                st.success(f"NEW_MEDIA_SOURCE_PATH set → {resolved}")
            except Exception as e:
                st.error(f"Failed to use folder: {e}")

# -------------------------
# APP USAGE
# -------------------------
else:
    st.subheader("App usage")
    st.markdown(
        """
### What this provides
- **Browse**: pulls creatives from `assets`, shows image files and **one** thumbnail per video with video-level stats.
- **Search**: same logic with simple filters. Search state is **cleared** when you leave the tab.
- **Upload**: pick a filesystem folder for new files (no ingest here).
        """
    )
    report = get_tables_report()
    st.table(report)

    st.markdown(
        f"""
### Where the chosen path is saved
- In-memory variable: `NEW_MEDIA_SOURCE_PATH`
- Session key: `st.session_state['new_media_source_path']`
- On-disk text file: `data/config/new_media_source_path.txt`

### Next steps
- Your ingest script can read the path from either:
  - In the same Streamlit process: `from app import NEW_MEDIA_SOURCE_PATH`
  - Separate script: `Path('data/config/new_media_source_path.txt').read_text().strip()`
- Move/copy from that source into:
  - `{IMAGES_DIR}` (images)
  - `{VIDEOS_DIR}` (videos)
- Then index into SQLite/FAISS using your pipeline.
        """
    )
