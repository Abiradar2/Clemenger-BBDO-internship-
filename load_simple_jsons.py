# load_simple_jsons.py
from __future__ import annotations
import os, json, sqlite3, re
from pathlib import Path
from typing import Optional, Tuple, List, Iterable, Dict, Any
import numpy as np
import faiss

# Import from your init file
from init_logs_bds import (
    get_handles,
    INPUT_DIR,
    FAISS_FILES,
    FAISS_METRIC,
)

# ---------- helpers ----------
DIGITS = re.compile(r"(\d+)")
REP_RE  = re.compile(r"(rep_\d+)")
TAG_SPLIT_RE = re.compile(r"[;,]")

def normsep(p: str) -> str:
    return p.replace("\\", "/")

def infer_creative_from_json(json_path: Path) -> Tuple[str, str]:
    """
    Returns (creative, media_type) based on:
      data/input/images/<creative>/* -> ("<creative>", "image")
      data/input/videos/<creative>/* -> ("<creative>", "video")
    """
    creative = json_path.parent.name
    up = json_path.parent.parent.name.lower()
    media = "image" if up == "images" else "video"
    return creative, media

def frame_no_from_name(name: str) -> int:
    m = DIGITS.search(name)
    return int(m.group(1)) if m else 0

def is_ndjson(path: Path) -> bool:
    s = path.suffix.lower()
    return s in {".jsons", ".ndjson"}

def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield dict records from either:
      - JSON array file (*.json), or
      - NDJSON file (*.jsons / *.ndjson), one JSON object per line.
    Lines starting with '#' or blank lines are ignored in NDJSON.
    """
    if is_ndjson(path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                except json.JSONDecodeError:
                    continue
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        yield rec
        except json.JSONDecodeError:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(rec, dict):
                    yield rec
            except Exception:
                pass

# ----- schema guard for new columns -----
def ensure_frames_meta_columns(con: sqlite3.Connection) -> None:
    """
    Make sure frames has columns: description TEXT, emotion TEXT.
    Idempotent: only ALTER TABLE if missing.
    """
    cols = {row[1] for row in con.execute("PRAGMA table_info(frames)").fetchall()}
    if "description" not in cols:
        con.execute("ALTER TABLE frames ADD COLUMN description TEXT")
    if "emotion" not in cols:
        con.execute("ALTER TABLE frames ADD COLUMN emotion TEXT")

# ----- assets -----
def ensure_asset_row(con: sqlite3.Connection, creative: str, media: str) -> int:
    row = con.execute("SELECT id FROM assets WHERE path=?", (creative,)).fetchone()
    if row:
        return row[0]
    con.execute("INSERT INTO assets (path, media_type) VALUES (?,?)", (creative, media))
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]

def ensure_assets_from_input(con: sqlite3.Connection) -> None:
    for root, media in [(INPUT_DIR / "images", "image"), (INPUT_DIR / "videos", "video")]:
        if root.exists():
            for child in sorted(p for p in root.iterdir() if p.is_dir()):
                ensure_asset_row(con, child.name, media)

def get_asset_id(con: sqlite3.Connection, creative: str) -> int:
    row = con.execute("SELECT id FROM assets WHERE path=?", (creative,)).fetchone()
    if not row:
        raise RuntimeError(f"Asset for creative '{creative}' not found. Run asset init first.")
    return row[0]

# ----- frames & segments -----
def ensure_frame(con: sqlite3.Connection, asset_id: int, rel_path: str, frame_no: int) -> int:
    row = con.execute("SELECT id FROM frames WHERE path=?", (rel_path,)).fetchone()
    if row:
        return row[0]
    con.execute(
        "INSERT INTO frames (asset_id, frame_no, path, has_embedding) VALUES (?,?,?,0)",
        (asset_id, frame_no, rel_path)
    )
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]

def update_frame_meta(con: sqlite3.Connection, frame_id: int, desc: Optional[str], emotion: Optional[str]) -> None:
    fields = []
    params = []
    if desc:
        fields.append("description=?")
        params.append(desc)
    if emotion:
        fields.append("emotion=?")
        params.append(emotion)
    if not fields:
        return
    params.append(frame_id)
    con.execute(f"UPDATE frames SET {', '.join(fields)} WHERE id=?", params)

def ensure_segment(con: sqlite3.Connection, asset_id: int, rel_path: str, frame_id: Optional[int] = None) -> int:
    row = con.execute("SELECT id FROM segments WHERE path=?", (rel_path,)).fetchone()
    if row:
        con.execute("UPDATE segments SET asset_id=?, frame_id=? WHERE id=?", (asset_id, frame_id, row[0]))
        return row[0]
    con.execute(
        "INSERT INTO segments (asset_id, frame_id, path, has_embedding) VALUES (?,?,?,0)",
        (asset_id, frame_id, rel_path)
    )
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]

def normalize_if_ip(X: np.ndarray, idx: faiss.Index) -> np.ndarray:
    base = getattr(idx, "index", idx)
    if isinstance(base, faiss.IndexFlatIP) and FAISS_METRIC == "cosine":
        faiss.normalize_L2(X)
    return X

def resolve_parent_frame_for_segment(
    con: sqlite3.Connection,
    asset_id: int,
    creative: str,
    media: str,
    segment_file_sub: str
) -> Optional[int]:
    if media == "image":
        return None

    seg_name = Path(segment_file_sub).name
    m = REP_RE.search(seg_name)
    if m:
        rep = m.group(1)
        found = con.execute(
            "SELECT id FROM frames WHERE asset_id=? AND path LIKE ? LIMIT 1",
            (asset_id, f"{creative}/frames_medoid/{rep}.%")
        ).fetchone()
        if found:
            return found[0]
        parent_rel = f"{creative}/frames_medoid/{rep}.jpg"
        fno = frame_no_from_name(rep)
        return ensure_frame(con, asset_id, parent_rel, fno)

    row = con.execute(
        "SELECT id FROM frames WHERE asset_id=? AND path LIKE ? LIMIT 1",
        (asset_id, f"{creative}/frames_medoid/%"),
    ).fetchone()
    if row:
        return row[0]

    parent_rel = f"{creative}/frames_medoid/rep_000000.jpg"
    return ensure_frame(con, asset_id, parent_rel, 0)

# ----- images_desc & images_tags -----
def upsert_image_desc(
    con: sqlite3.Connection,
    asset_id: int,
    industry: str,
    product: str,
    description: str,
    emotion: str,
) -> None:
    """
    One row per *image* asset in images_desc, including emotion.
    Only non-empty incoming fields overwrite existing ones.
    """
    con.execute(
        """
        INSERT INTO images_desc (asset_id, industry, product, description, emotion)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
          industry    = CASE
                           WHEN excluded.industry    <> '' THEN excluded.industry
                           ELSE industry
                        END,
          product     = CASE
                           WHEN excluded.product     <> '' THEN excluded.product
                           ELSE product
                        END,
          description = CASE
                           WHEN excluded.description <> '' THEN excluded.description
                           ELSE description
                        END,
          emotion     = CASE
                           WHEN excluded.emotion     <> '' THEN excluded.emotion
                           ELSE emotion
                        END
        """,
        (
            asset_id,
            (industry or "").strip(),
            (product or "").strip(),
            (description or "").strip(),
            (emotion or "").strip(),
        ),
    )

def add_image_tags(con: sqlite3.Connection, asset_id: int, tags_field: str) -> int:
    if not tags_field:
        return 0
    raw = [t.strip() for t in re.split(TAG_SPLIT_RE, tags_field) if t.strip()]
    inserted = 0
    for tag in raw:
        try:
            con.execute("INSERT OR IGNORE INTO images_tags (asset_id, tag) VALUES (?,?)", (asset_id, tag))
            if con.execute("SELECT changes()").fetchone()[0] == 1:
                inserted += 1
        except Exception:
            pass
    return inserted

# ----- videos_desc helpers -----
def upsert_video_desc(
    con: sqlite3.Connection,
    asset_id: int,
    industry: str,
    product: str,
    description: str,
) -> None:
    """
    One row per VIDEO asset in videos_desc.
    Only non-empty incoming fields overwrite existing ones.
    """
    con.execute(
        """
        INSERT INTO videos_desc (asset_id, industry, product, description)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
          industry = CASE
                       WHEN excluded.industry    <> '' THEN excluded.industry
                       ELSE industry
                     END,
          product  = CASE
                       WHEN excluded.product     <> '' THEN excluded.product
                       ELSE product
                     END,
          description = CASE
                          WHEN excluded.description <> '' THEN excluded.description
                          ELSE description
                        END
        """,
        (
            asset_id,
            (industry or "").strip(),
            (product or "").strip(),
            (description or "").strip(),
        ),
    )

# ----- frames_tags helpers -----
def _clean_one_word(tag: str) -> Optional[str]:
    """
    Enforce single-word tags to satisfy DB CHECK(tag NOT LIKE '% %').
    - Strip punctuation at ends
    - Lowercase
    - If whitespace remains, take the first token
    - Drop empty results
    """
    if not tag:
        return None
    t = tag.strip().strip(",.;:!?'\"").lower()
    if not t:
        return None
    # keep only the first word to avoid CHECK violation
    t = t.split()[0]
    if not t or " " in t:
        return None
    return t

def add_frame_tags(con: sqlite3.Connection, frame_id: int, tags_field: str) -> int:
    """
    Insert 1-word tags for a given frame into frames_tags.
    Returns number of newly inserted tags.
    """
    if not tags_field:
        return 0
    raw = [t for t in re.split(TAG_SPLIT_RE, tags_field) if t and t.strip()]
    inserted = 0
    for tag in raw:
        t = _clean_one_word(tag)
        if not t:
            continue
        try:
            con.execute("INSERT OR IGNORE INTO frames_tags (frame_id, tag) VALUES (?,?)", (frame_id, t))
            if con.execute("SELECT changes()").fetchone()[0] == 1:
                inserted += 1
        except sqlite3.Error:
            # If any unexpected constraint error (shouldn't happen with OR IGNORE), skip tag
            continue
    return inserted

# ---------- loaders ----------
def load_clip_embeddings(con: sqlite3.Connection, idx_frames: faiss.Index, idx_images: faiss.Index, json_path: Path) -> None:
    creative, media = infer_creative_from_json(json_path)
    asset_id = get_asset_id(con, creative)
    records = list(iter_records(json_path))

    if media == "image":
        root_items = [it for it in records if it.get("file") and "/" not in normsep(it["file"]) and "embedding" in it]
        if not root_items:
            print(f"⏭️  image has no root file embedding in {json_path} — skipping IMAGES FAISS add")
            return
        emb = np.asarray(root_items[0]["embedding"], dtype="float32")[None, :]
        X = normalize_if_ip(emb.astype("float32"), idx_images)
        try:
            idx_images.remove_ids(np.asarray([asset_id], dtype=np.int64))
        except Exception:
            pass
        idx_images.add_with_ids(X, np.asarray([asset_id], dtype=np.int64))
        faiss.write_index(idx_images, str(FAISS_FILES["images"]))
        print(f"✅ image embedding → {json_path}  (added 1 vector to IMAGES FAISS with id={asset_id})")
        return

    ids, vecs = [], []
    for item in records:
        if "embedding" not in item:
            continue
        file_sub = normsep(item.get("file", ""))
        if not file_sub or not file_sub.startswith("frames_medoid/"):
            continue
        rel_path = f"{creative}/{file_sub}"
        fno = frame_no_from_name(Path(file_sub).name)
        frame_id = ensure_frame(con, asset_id, rel_path, fno)

        row = con.execute("SELECT has_embedding FROM frames WHERE id=?", (frame_id,)).fetchone()
        already = bool(row and row[0] == 1)

        emb = np.asarray(item["embedding"], dtype="float32")[None, :]
        if not already:
            vecs.append(emb)
            ids.append(frame_id)
            con.execute("UPDATE frames SET has_embedding=1 WHERE id=?", (frame_id,))

    if vecs:
        X = np.vstack(vecs).astype("float32")
        X = normalize_if_ip(X, idx_frames)
        idx_frames.add_with_ids(X, np.asarray(ids, dtype=np.int64))
        faiss.write_index(idx_frames, str(FAISS_FILES["frames"]))
    print(f"✅ video clip embeddings → {json_path}  (added {len(ids)} frames to FRAMES FAISS)")

def load_qwen_multi_prompts(con: sqlite3.Connection, json_path: Path) -> None:
    """
    For image creatives:
      - Upserts images_desc from "__ALL_IMAGES__"/IMG-ALL and IMG-ORIG (incl. emotion).
      - Adds images_tags from IMG-ORIG items.

    For video creatives:
      - Upserts videos_desc from "__ALL_MEDOIDS__" (source_set 'VID-FRAMES'),
        including the brief video-level description ('desc').
      - Ensures frames for 'frames_medoid/*' items, updates per-frame description/emotion,
        and inserts per-frame tags into frames_tags.
      - Ensures segments for '__obj_*' items with resolved parent frame.
    """
    creative, media = infer_creative_from_json(json_path)
    asset_id = get_asset_id(con, creative)

    made_frames = made_segments = 0
    desc_written = False
    tags_written_images = 0
    tags_written_frames = 0
    video_meta_upserts = 0

    for item in iter_records(json_path):
        file_sub = normsep(item.get("file", "") or "")
        source_set = (item.get("source_set") or "").upper()

        # ---------- IMAGES ----------
        if media == "image":
            # Use both IMG-ALL / __ALL_IMAGES__ AND IMG-ORIG to populate images_desc
            if (
                file_sub == "__ALL_IMAGES__"
                or source_set == "IMG-ALL"
                or source_set == "IMG-ORIG"
            ):
                upsert_image_desc(
                    con,
                    asset_id,
                    industry=item.get("industry", "") or "",
                    product=item.get("product", "") or "",
                    description=item.get("desc", "") or "",
                    emotion=item.get("emotion", "") or "",
                )
                desc_written = True

            if source_set == "IMG-ORIG" and item.get("tags"):
                tags_written_images += add_image_tags(con, asset_id, item["tags"])

            # image segments (object crops) if present
            if "__obj_" in file_sub and file_sub:
                rel_path = f"{creative}/{file_sub}"
                ensure_segment(con, asset_id=asset_id, rel_path=rel_path, frame_id=None)
                made_segments += 1
            continue

        # ---------- VIDEOS ----------
        # Whole-video metadata (industry/product/description) — ONLY from __ALL_MEDOIDS__
        if media == "video" and file_sub == "__ALL_MEDOIDS__":
            industry    = (item.get("industry") or "").strip()
            product     = (item.get("product")  or "").strip()
            description = (item.get("desc")     or "").strip()
            if industry or product or description:
                upsert_video_desc(con, asset_id, industry, product, description)
                video_meta_upserts += 1

        if not file_sub:
            continue

        rel_path = f"{creative}/{file_sub}"
        is_segment = "__obj_" in file_sub

        if is_segment:
            frame_id = resolve_parent_frame_for_segment(con, asset_id, creative, media, file_sub)
            ensure_segment(con, asset_id=asset_id, rel_path=rel_path, frame_id=frame_id)
            made_segments += 1
            continue

        # Frame-level items (per-frame desc/emotion/tags)
        if file_sub.startswith("frames_medoid/"):
            fno = frame_no_from_name(Path(file_sub).name)
            frame_id = ensure_frame(con, asset_id, rel_path, fno)

            # Update per-frame metadata
            update_frame_meta(
                con,
                frame_id=frame_id,
                desc=item.get("desc"),
                emotion=item.get("emotion")
            )

            # Add per-frame tags → frames_tags
            if item.get("tags"):
                tags_written_frames += add_frame_tags(con, frame_id, item["tags"])

            made_frames += 1

    msg_extra = []
    if media == "image":
        if desc_written:
            msg_extra.append("images_desc ✔")
        if tags_written_images:
            msg_extra.append(f"images_tags +{tags_written_images}")
    else:
        if video_meta_upserts:
            msg_extra.append(f"videos_desc upserts={video_meta_upserts}")
        if tags_written_frames:
            msg_extra.append(f"frames_tags +{tags_written_frames}")
    extra = f" [{' ,'.join(msg_extra)}]" if msg_extra else ""
    print(f"✅ qwen prompts → {json_path}  (frames+segments ensured: {made_frames}+{made_segments}){extra}")

# ---------- diagnostics ----------
def print_frames_schema_and_samples(con: sqlite3.Connection, limit: int = 10) -> None:
    print("\n== frames schema ==")
    cols = con.execute("PRAGMA table_info(frames)").fetchall()
    for cid, name, ctype, notnull, dflt, pk in cols:
        print(f"  {name:<14} {ctype:<10} notnull={notnull} default={dflt} pk={pk}")

    print("\n== sample frames (including description/emotion) ==")
    try:
        rows = con.execute(
            "SELECT id, asset_id, frame_no, path, has_embedding, description, emotion "
            "FROM frames ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            print("  <no frames>")
        else:
            for r in rows:
                print(" ", r)
    except sqlite3.Error as e:
        print(f"  (error selecting frames with new columns): {e}")

def print_frames_tags_samples(con: sqlite3.Connection, limit: int = 15) -> None:
    print("\n== frames_tags (sample) ==")
    try:
        rows = con.execute(
            "SELECT ft.id, ft.frame_id, ft.tag, f.path "
            "FROM frames_tags ft "
            "JOIN frames f ON f.id = ft.frame_id "
            "ORDER BY ft.id LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            print("  <no frame tags>")
        else:
            for r in rows:
                print(" ", r)
    except sqlite3.Error as e:
        print(f"  (error selecting frames_tags): {e}")

def print_videos_desc_samples(con: sqlite3.Connection, limit: int = 10) -> None:
    print("\n== videos_desc (sample) ==")
    try:
        rows = con.execute(
            "SELECT vd.id, vd.asset_id, a.path, vd.industry, vd.product, vd.description "
            "FROM videos_desc vd "
            "JOIN assets a ON a.id = vd.asset_id "
            "ORDER BY vd.id LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            print("  <no video rows>")
        else:
            for r in rows:
                print(" ", r)
    except sqlite3.Error as e:
        print(f"  (error selecting videos_desc): {e}")

# ---------- driver ----------
def load_all():
    con, indices, _ = get_handles()
    # make sure frames has the new columns, even if init ran before you added them
    ensure_frames_meta_columns(con)

    idx_frames = indices["frames"]
    idx_images = indices["images"]

    # 1) Ensure assets exist (works for any number of image/video creatives)
    ensure_assets_from_input(con)

    # 2) Process all creatives
    for dirpath, _, files in os.walk(INPUT_DIR):
        files_set = set(files)
        root = Path(dirpath)

        clip_json = None
        if "clip_embeddings.json" in files_set:
            clip_json = root / "clip_embeddings.json"
        elif "clip_embeddings.jsons" in files_set or "clip_embeddings.ndjson" in files_set:
            clip_json = root / ("clip_embeddings.jsons" if "clip_embeddings.jsons" in files_set else "clip_embeddings.ndjson")

        if clip_json is not None:
            load_clip_embeddings(con, idx_frames, idx_images, clip_json)

        qwen_json = None
        if "qwen_multi_prompts.jsons" in files_set:
            qwen_json = root / "qwen_multi_prompts.jsons"
        elif "qwen_multi_prompts.ndjson" in files_set:
            qwen_json = root / "qwen_multi_prompts.ndjson"
        elif "qwen_multi_prompts.json" in files_set:
            qwen_json = root / "qwen_multi_prompts.json"

        if qwen_json is not None:
            load_qwen_multi_prompts(con, qwen_json)

    # quick totals
    def _count(q):
        try:
            return int(con.execute(q).fetchone()[0])
        except Exception:
            return 0

    print("\nTotals:")
    print("  assets        :", _count("SELECT COUNT(*) FROM assets"))
    print("  frames        :", _count("SELECT COUNT(*) FROM frames"))
    print("  segments      :", _count("SELECT COUNT(*) FROM segments"))
    print("  emb(fr)       :", _count("SELECT COUNT(*) FROM frames WHERE has_embedding=1"))
    print("  images_desc   :", _count("SELECT COUNT(*) FROM images_desc"))
    print("  images_tags   :", _count("SELECT COUNT(*) FROM images_tags"))
    print("  frames_tags   :", _count("SELECT COUNT(*) FROM frames_tags"))
    print("  videos_desc   :", _count("SELECT COUNT(*) FROM videos_desc"))

    # show samples
    print_frames_schema_and_samples(con, limit=10)
    print_frames_tags_samples(con, limit=15)
    print_videos_desc_samples(con, limit=10)

if __name__ == "__main__":
    load_all()
