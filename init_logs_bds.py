# init_simple.py
from __future__ import annotations
from pathlib import Path
import sqlite3, json, datetime, os
from typing import Dict, Tuple
import faiss

# ------------------
# PATHS / CONFIG
# ------------------
DATA_ROOT   = Path("data")
DB_DIR      = DATA_ROOT / "index"
LOG_DIR     = DATA_ROOT / "logs"
SQLITE_DB   = DB_DIR / "app.db"
INPUT_DIR   = DATA_ROOT / "input"
IMAGES_DIR  = INPUT_DIR / "images"
VIDEOS_DIR  = INPUT_DIR / "videos"

FAISS_FILES = {
    "frames":   DB_DIR / "openclip.frames.faiss",   # video frames
    "images":   DB_DIR / "openclip.images.faiss",   # image creatives (keyed by asset_id)
    "segments": DB_DIR / "openclip.segments.faiss", # reserved (not used below)
}
FAISS_DIM    = 512       # OpenCLIP ViT-B/32 default
FAISS_METRIC = "cosine"  # cosine -> IndexFlatIP + L2-normalize at add/search

STATUS_JSON    = LOG_DIR / "app_status.json"
HISTORY_NDJSON = LOG_DIR / "history.ndjson"

# ------------------
# SCHEMA
# ------------------
# Frames table holds *video frames only*.
# Segments always link to asset_id and may link to a frame_id (NULL for image segments).
SQL_SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS assets (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,   -- creative folder name (e.g., "ac6698638526")
  media_type TEXT CHECK(media_type IN ('image','video')) NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Only *video* frames here
CREATE TABLE IF NOT EXISTS frames (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL,
  frame_no INTEGER NOT NULL,
  path TEXT UNIQUE NOT NULL,   -- "<creative>/frames_medoid/rep_000123.jpg" etc.
  has_embedding INT DEFAULT 0,
  description TEXT DEFAULT '', -- free-text description for the frame
  emotion TEXT DEFAULT NULL,   -- optional emotion label (e.g., 'Joyful')
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

-- Segments belong to an asset (always) and may belong to a frame (video) or be NULL (image)
CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL,
  frame_id INTEGER NULL,       -- NULL for image segments; set for video segments
  path TEXT UNIQUE NOT NULL,   -- stored as "<creative>/<subpath>"
  has_embedding INT DEFAULT 0,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
  FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE SET NULL
);

-- One descriptive row per *image* asset
CREATE TABLE IF NOT EXISTS images_desc (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL UNIQUE,
  industry TEXT,
  product TEXT,
  description TEXT,
  emotion TEXT,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

-- Many tags per *image* asset; 1-word tags; de-duped
CREATE TABLE IF NOT EXISTS images_tags (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL,
  tag TEXT NOT NULL,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
  UNIQUE(asset_id, tag)
);

-- NEW: Many tags per *video frame*; 1-word per row; de-duped per frame
CREATE TABLE IF NOT EXISTS frames_tags (
  id INTEGER PRIMARY KEY,                  -- 1,2,3,4,... (rowid)
  frame_id INTEGER NOT NULL,               -- references frames.id
  tag TEXT NOT NULL,                       -- one word per row
  FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE,
  UNIQUE(frame_id, tag),                   -- de-dup per frame
  CHECK(tag NOT LIKE '% %')                -- enforce single word
);

-- NEW: One descriptive row per *video* asset
CREATE TABLE IF NOT EXISTS videos_desc (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL UNIQUE,        -- must reference a VIDEO asset
  industry TEXT,
  product TEXT,
  description TEXT, -- breif
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_frames_asset_id    ON frames(asset_id);
CREATE INDEX IF NOT EXISTS idx_segments_asset_id  ON segments(asset_id);
CREATE INDEX IF NOT EXISTS idx_segments_frame_id  ON segments(frame_id);
CREATE INDEX IF NOT EXISTS idx_frames_tags_frame  ON frames_tags(frame_id);

-- Enforce videos_desc.asset_id refers to a VIDEO asset
CREATE TRIGGER IF NOT EXISTS trg_videos_desc_asset_is_video_ins
BEFORE INSERT ON videos_desc
FOR EACH ROW
BEGIN
  SELECT CASE
    WHEN (SELECT media_type FROM assets WHERE id = NEW.asset_id) <> 'video'
    THEN RAISE(ABORT, 'videos_desc.asset_id must refer to a video asset')
  END;
END;

CREATE TRIGGER IF NOT EXISTS trg_videos_desc_asset_is_video_upd
BEFORE UPDATE OF asset_id ON videos_desc
FOR EACH ROW
BEGIN
  SELECT CASE
    WHEN (SELECT media_type FROM assets WHERE id = NEW.asset_id) <> 'video'
    THEN RAISE(ABORT, 'videos_desc.asset_id must refer to a video asset')
  END;
END;

-- Invariant: if frame_id is set, its asset must match segments.asset_id
CREATE TRIGGER IF NOT EXISTS trg_segments_frame_asset_guard
BEFORE INSERT ON segments
FOR EACH ROW
BEGIN
  SELECT CASE
    WHEN NEW.frame_id IS NULL THEN NULL
    WHEN (SELECT asset_id FROM frames WHERE id = NEW.frame_id) <> NEW.asset_id
    THEN RAISE(ABORT, 'segments.asset_id must match frames.asset_id when frame_id is set')
  END;
END;

CREATE TRIGGER IF NOT EXISTS trg_segments_frame_asset_guard_upd
BEFORE UPDATE OF frame_id, asset_id ON segments
FOR EACH ROW
BEGIN
  SELECT CASE
    WHEN NEW.frame_id IS NULL THEN NULL
    WHEN (SELECT asset_id FROM frames WHERE id = NEW.frame_id) <> NEW.asset_id
    THEN RAISE(ABORT, 'segments.asset_id must match frames.asset_id when frame_id is set')
  END;
END;
"""

# ------------------
# DIRS / SQLITE / FAISS
# ------------------
def _ensure_dirs():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

def init_sqlite() -> sqlite3.Connection:
    con = sqlite3.connect(str(SQLITE_DB), isolation_level=None, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    con.executescript(SQL_SCHEMA)
    _migrate_frames_add_desc_emotion(con)   # ensure new columns on existing DBs
    _migrate_segments_secondary_key(con)    # existing migration (idempotent)
    return con

def _new_faiss(dim: int, metric: str) -> faiss.IndexIDMap2:
    base = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
    return faiss.IndexIDMap2(base)

def init_faiss_file(path: Path, dim: int, metric: str) -> faiss.IndexIDMap2:
    if path.exists():
        try:
            return faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
        except Exception:
            return faiss.read_index(str(path))
    idx = _new_faiss(dim, metric)
    faiss.write_index(idx, str(path))
    return idx

# ------------------
# PREPOPULATE ASSETS (exactly one row per creative folder)
# ------------------
def _upsert_asset(con: sqlite3.Connection, creative: str, media_type: str) -> int:
    row = con.execute("SELECT id FROM assets WHERE path=?", (creative,)).fetchone()
    if row:
        return row[0]
    con.execute("INSERT INTO assets (path, media_type) VALUES (?, ?)", (creative, media_type))
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]

def prepopulate_assets_from_input(con: sqlite3.Connection):
    # images/*
    if IMAGES_DIR.exists():
        for p in sorted(x for x in IMAGES_DIR.iterdir() if x.is_dir()):
            _upsert_asset(con, p.name, "image")
    # videos/*
    if VIDEOS_DIR.exists():
        for p in sorted(x for x in VIDEOS_DIR.iterdir() if x.is_dir()):
            _upsert_asset(con, p.name, "video")

# ------------------
# SNAPSHOT / LOGS
# ------------------
def snapshot_status(con: sqlite3.Connection) -> Dict:
    def _count(q: str) -> int:
        try:
            return int(con.execute(q).fetchone()[0])
        except Exception:
            return 0
    snap = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "paths": {
            "sqlite": str(SQLITE_DB.resolve()),
            "faiss": {k: str(v.resolve()) for k, v in FAISS_FILES.items()},
        },
        "totals": {
            "assets":        _count("SELECT COUNT(*) FROM assets"),
            "images":        _count("SELECT COUNT(*) FROM assets WHERE media_type='image'"),
            "videos":        _count("SELECT COUNT(*) FROM assets WHERE media_type='video'"),
            "frames":        _count("SELECT COUNT(*) FROM frames"),
            "segments":      _count("SELECT COUNT(*) FROM segments"),
            "images_desc":   _count("SELECT COUNT(*) FROM images_desc"),
            "images_tags":   _count("SELECT COUNT(*) FROM images_tags"),
            "frames_tags":   _count("SELECT COUNT(*) FROM frames_tags"),
            "videos_desc":   _count("SELECT COUNT(*) FROM videos_desc")
        }
    }
    return snap

def write_logs(snapshot: Dict):
    STATUS_JSON.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    with open(HISTORY_NDJSON, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": snapshot["timestamp"], "event": "init_or_open", "totals": snapshot["totals"]}) + "\n")

# ------------------
# MIGRATIONS (idempotent)
# ------------------
def _migrate_frames_add_desc_emotion(con: sqlite3.Connection):
    """
    Ensure frames.description (TEXT DEFAULT '') and frames.emotion (TEXT NULL) exist.
    Safe to run repeatedly.
    """
    cols = {name for (_cid, name, *_rest) in con.execute("PRAGMA table_info(frames)").fetchall()}
    to_add = []
    if "description" not in cols:
        to_add.append(("description", "TEXT DEFAULT ''"))
    if "emotion" not in cols:
        to_add.append(("emotion", "TEXT DEFAULT NULL"))
    if to_add:
        con.execute("BEGIN")
        try:
            for col, decl in to_add:
                con.execute(f"ALTER TABLE frames ADD COLUMN {col} {decl}")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

def _migrate_segments_secondary_key(con: sqlite3.Connection):
    """
    If an older 'segments' table lacked asset_id or had non-nullable frame_id,
    bring it to the new shape:
      - segments.asset_id NOT NULL
      - segments.frame_id NULL
      - triggers + indexes present
    """
    # ensure indexes + triggers exist (idempotent)
    con.executescript("""
    CREATE INDEX IF NOT EXISTS idx_segments_asset_id ON segments(asset_id);
    CREATE INDEX IF NOT EXISTS idx_segments_frame_id ON segments(frame_id);

    CREATE TRIGGER IF NOT EXISTS trg_segments_frame_asset_guard
    BEFORE INSERT ON segments
    FOR EACH ROW
    BEGIN
      SELECT CASE
        WHEN NEW.frame_id IS NULL THEN NULL
        WHEN (SELECT asset_id FROM frames WHERE id = NEW.frame_id) <> NEW.asset_id
        THEN RAISE(ABORT, 'segments.asset_id must match frames.asset_id when frame_id is set')
      END;
    END;

    CREATE TRIGGER IF NOT EXISTS trg_segments_frame_asset_guard_upd
    BEFORE UPDATE OF frame_id, asset_id ON segments
    FOR EACH ROW
    BEGIN
      SELECT CASE
        WHEN NEW.frame_id IS NULL THEN NULL
        WHEN (SELECT asset_id FROM frames WHERE id = NEW.frame_id) <> NEW.asset_id
        THEN RAISE(ABORT, 'segments.asset_id must match frames.asset_id when frame_id is set')
      END;
    END;
    """)

    cols = con.execute("PRAGMA table_info(segments)").fetchall()
    colnames = {c[1]: c for c in cols}

    # add asset_id if missing and backfill
    if "asset_id" not in colnames:
        con.execute("BEGIN")
        try:
            con.execute("ALTER TABLE segments ADD COLUMN asset_id INTEGER")
            # backfill from frames when possible
            con.execute("""
            UPDATE segments
            SET asset_id = (SELECT asset_id FROM frames f WHERE f.id = segments.frame_id)
            WHERE frame_id IS NOT NULL AND asset_id IS NULL;
            """)
            # infer from creative prefix if still NULL (path like "<creative>/...")
            con.execute("""
            WITH c AS (
              SELECT id, substr(path, 1, instr(path, '/')-1) AS creative
              FROM segments WHERE asset_id IS NULL
            )
            UPDATE segments
            SET asset_id = (SELECT a.id FROM assets a, c WHERE a.path = c.creative AND c.id = segments.id)
            WHERE asset_id IS NULL;
            """)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

# ------------------
# PUBLIC ENTRYPOINT
# ------------------
def get_handles(
    sqlite_path: Path = SQLITE_DB,
    faiss_dim: int = FAISS_DIM,
    metric: str = FAISS_METRIC,
) -> Tuple[sqlite3.Connection, Dict[str, faiss.IndexIDMap2], Dict]:
    _ensure_dirs()
    global SQLITE_DB
    SQLITE_DB = Path(sqlite_path)
    con = init_sqlite()

    # IMPORTANT: prepopulate assets strictly from folder names
    prepopulate_assets_from_input(con)

    indices = {k: init_faiss_file(v, faiss_dim, metric) for k, v in FAISS_FILES.items()}
    snap = snapshot_status(con)
    write_logs(snap)
    return con, indices, snap

if __name__ == "__main__":
    con, idxs, snap = get_handles()
    print("SQLite:", snap["paths"]["sqlite"])
    print("Totals:", snap["totals"])
    print("\n[assets]")
    for r in con.execute("SELECT id, path, media_type, created_at FROM assets ORDER BY media_type, path").fetchall():
        print(" ", r)
