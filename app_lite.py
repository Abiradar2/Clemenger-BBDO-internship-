# app.py
# Streamlit app with:
# - DB/FAISS bootstrap (via init_logs_bds.get_handles)
# - Semantic CLIP search for images + videos (Search page)
# - Two-page layout with sidebar: Home / Search

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import os
from base64 import b64encode

import numpy as np
import streamlit as st

# ---------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------
st.set_page_config(page_title="Creative Search", layout="wide")

# ---------------------------------------
# GLOBAL STYLE + OUTFIT FONT
# ---------------------------------------
def _load_outfit_font_css() -> str:
    """
    Embed local Outfit font as a base64 data URL so the browser can use it.
    Looks for: Outfit/Outfit-VariableFont_wght.ttf next to app.py
    """
    try:
        font_path = Path(__file__).resolve().parent / "Outfit" / "Outfit-VariableFont_wght.ttf"
        with open(font_path, "rb") as f:
            font_data = f.read()
        font_b64 = b64encode(font_data).decode("utf-8")

        return f"""
        <style>
        @font-face {{
            font-family: 'Outfit';
            src: url(data:font/ttf;base64,{font_b64}) format('truetype');
            font-weight: 100 900;
            font-style: normal;
        }}

        html, body, [class*="css"]  {{
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 16px;
            background-color: #ffffff;
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            background-color: #ffffff;
        }}
        </style>
        """
    except Exception:
        # Fallback: no custom font, still white
        return """
        <style>
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 16px;
            background-color: #ffffff;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            background-color: #ffffff;
        }
        </style>
        """

# Apply global CSS (font + background)
st.markdown(_load_outfit_font_css(), unsafe_allow_html=True)

# Tame threaded libs for Streamlit servers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# =======================================
# IMPORT YOUR INIT (db + faiss handles)
# =======================================
from init_logs_bds import get_handles  # or: from init_simple import get_handles

# =======================================
# PATH CONSTANTS
# =======================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data"
INPUT_DIR    = DATA_ROOT / "input"
WS_DIR       = DATA_ROOT / "workspace"
IMAGES_DIR   = WS_DIR / "images"
VIDEOS_DIR   = WS_DIR / "videos"

# =======================================
# BOOTSTRAP DBs & INDICES (cached)
# =======================================
@st.cache_resource(show_spinner=False)
def _init_handles() -> Tuple:
    con, indices, snapshot = get_handles()
    return con, indices, snapshot

con, indices, snapshot = _init_handles()

# =======================================
# OPENCLIP MODEL (cached)
# =======================================
@st.cache_resource(show_spinner=True)
def _load_openclip():
    """
    Load OpenCLIP ViT-B/32 once and cache it.
    Uses CPU unless CUDA is available.
    """
    import torch
    import open_clip

    model_name = "ViT-B-32"
    pretrained = "openai"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, device

def text_to_clip_embedding(text: str) -> np.ndarray:
    """
    Convert a text prompt into a normalized 1x512 embedding for cosine FAISS search.
    """
    import torch

    model, tokenizer, device = _load_openclip()
    if not text.strip():
        raise ValueError("Empty text for CLIP embedding")

    with torch.no_grad():
        tokens = tokenizer([text]).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    vec = text_features.cpu().numpy().astype("float32")  # shape (1,512)
    return vec

# =======================================
# MEDIA HELPERS
# =======================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def _safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _iter_image_files_under(root: Path, max_per_creative: int = 24) -> List[Path]:
    """
    Return only top-level image files in the creative folder.
    """
    files: List[Path] = []
    if root.exists() and root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
                if len(files) >= max_per_creative:
                    break
    return files

def _find_video_file_for_creative(creative: str) -> Optional[Path]:
    """
    Best-effort: given a creative path (e.g. 'ACME/TVC_15s'),
    look for a real video file under INPUT_DIR/videos/creative
    or VIDEOS_DIR/creative.
    """
    if not creative:
        return None

    candidates: List[Path] = []
    roots = [
        (INPUT_DIR / "videos" / creative),
        (VIDEOS_DIR / creative),
    ]
    for root in roots:
        if root.exists() and root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: len(str(x)))
    return candidates[0]

def fetch_media(
    creative: Optional[str] = None,
    media_type: str = "Both",
    limit: int = 500,
) -> List[Dict]:
    """
    Build a mixed list: images (actual files) and ONE video cover (earliest frame).
    """
    mm = {"Both": ("image", "video"), "Images": ("image",), "Videos": ("video",)}
    want = mm.get(media_type, ("image", "video"))

    if creative is None:
        rows = con.execute(
            "SELECT id, path, media_type FROM assets "
            "WHERE media_type IN (?,?) ORDER BY path",
            want if len(want) == 2 else (want[0], want[0]),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT id, path, media_type FROM assets "
            "WHERE path=? AND media_type IN (?,?)",
            (creative,) + (want if len(want) == 2 else (want[0], want[0])),
        ).fetchall()

    out: List[Dict] = []

    for asset_id, path_str, mtype in rows:
        if mtype == "image":
            ws_dir = (IMAGES_DIR / path_str).resolve()
            in_dir = (INPUT_DIR / "images" / path_str).resolve()
            files = _iter_image_files_under(ws_dir, 24) or _iter_image_files_under(
                in_dir, 24
            )
            if not files:
                continue

            industry = product = description = emotion = ""
            try:
                row = con.execute(
                    "SELECT "
                    "COALESCE(industry,''), "
                    "COALESCE(product,''), "
                    "COALESCE(description,''), "
                    "COALESCE(emotion,'') "
                    "FROM images_desc WHERE asset_id=?",
                    (asset_id,),
                ).fetchone()
                if row:
                    industry, product, description, emotion = row
            except Exception:
                pass

            tags: List[str] = []
            try:
                tags = [
                    t
                    for (t,) in con.execute(
                        "SELECT tag FROM images_tags "
                        "WHERE asset_id=? ORDER BY tag",
                        (asset_id,),
                    ).fetchall()
                ]
            except Exception:
                pass

            for p in files:
                if not _safe_exists(p):
                    continue
                out.append(
                    {
                        "kind": "image",
                        "asset_id": asset_id,
                        "creative": path_str,
                        "file": str(p),
                        "caption": " · ".join(
                            [x for x in [path_str, industry or None, product or None] if x]
                        ),
                        "tags": tags,
                        "description": description,
                        "emotion": emotion,
                        "labels": {
                            "industry": industry,
                            "product": product,
                            "emotion": emotion,
                        },
                    }
                )

        elif mtype == "video":
            try:
                row = con.execute(
                    "SELECT frame_no, path FROM frames "
                    "WHERE asset_id=? ORDER BY frame_no ASC LIMIT 1",
                    (asset_id,),
                ).fetchone()
            except Exception:
                row = None

            if not row:
                continue

            frame_no, rel_path = row
            ws_primary = (VIDEOS_DIR / rel_path).resolve()
            ws_alt     = (VIDEOS_DIR / path_str / Path(rel_path).name).resolve()
            in_primary = (INPUT_DIR / "videos" / rel_path).resolve()
            in_alt     = (INPUT_DIR / "videos" / path_str / Path(rel_path).name).resolve()
            thumb = next(
                (p for p in [ws_primary, ws_alt, in_primary, in_alt] if _safe_exists(p)),
                None,
            )
            if not thumb:
                continue

            industry = product = v_description = ""
            try:
                vdesc = con.execute(
                    "SELECT "
                    "COALESCE(industry,''), "
                    "COALESCE(product,''), "
                    "COALESCE(description,'') "
                    "FROM videos_desc WHERE asset_id=?",
                    (asset_id,),
                ).fetchone()
            except Exception:
                vdesc = None

            if vdesc:
                industry, product, v_description = vdesc

            video_tags: List[str] = []
            try:
                video_tags = [
                    t
                    for (t,) in con.execute(
                        """
                        SELECT DISTINCT ft.tag
                        FROM frames_tags ft
                        JOIN frames f
                          ON f.id = ft.frame_id
                        WHERE f.asset_id=?
                        ORDER BY ft.tag
                        """,
                        (asset_id,),
                    ).fetchall()
                ]
            except Exception:
                video_tags = []

            try:
                fc = con.execute(
                    "SELECT COUNT(*) FROM frames WHERE asset_id=?",
                    (asset_id,),
                ).fetchone()[0]
            except Exception:
                fc = 0
            try:
                f_emb = con.execute(
                    "SELECT COALESCE(SUM(has_embedding),0) "
                    "FROM frames WHERE asset_id=?",
                    (asset_id,),
                ).fetchone()[0]
            except Exception:
                f_emb = 0

            emotion_chain: List[Dict[str, Any]] = []
            try:
                rows_em = con.execute(
                    """
                    SELECT frame_no, COALESCE(emotion,'')
                    FROM frames
                    WHERE asset_id=?
                    ORDER BY frame_no
                    """,
                    (asset_id,),
                ).fetchall()
                for fn, em in rows_em:
                    em = (em or "").strip()
                    if not em:
                        continue
                    emotion_chain.append(
                        {"frame_no": int(fn), "emotion": em}
                    )
            except Exception:
                emotion_chain = []

            caption_bits = [path_str, f"f{frame_no}"]
            if industry:
                caption_bits.append(industry)
            if product:
                caption_bits.append(product)

            out.append(
                {
                    "kind": "video",
                    "asset_id": asset_id,
                    "creative": path_str,
                    "file": str(thumb),
                    "caption": " · ".join(caption_bits),
                    "tags": video_tags,
                    "description": v_description,
                    "emotion_chain": emotion_chain,
                    "video_stats": {
                        "frames_total": int(fc),
                        "frames_with_embeddings": int(f_emb),
                    },
                    "labels": {
                        "industry": industry,
                        "product": product,
                    },
                }
            )

        if len(out) >= limit:
            break

    return out[:limit]

# =======================================
# FIXED LAYOUT SETTINGS
# =======================================
DETAIL_MAX_VH = 45
GRID_COLS     = 3

# =======================================
# RENDERERS
# =======================================
def grid_gallery(items: List[Dict], cols: Optional[int] = None):
    if not items:
        st.info("No media found for the current selection.")
        return

    if cols is None:
        cols = GRID_COLS

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

            # Similarity score on the search page
            sim = it.get("score", None)
            if sim is None and "clip_score" in it:
                sim = it.get("clip_score")

            if sim is not None:
                # Show 0–1 as 2-decimal similarity; change to percentage if you prefer
                st.caption(f"Similarity: {sim:.2f}")

            if st.button("View details", key=f"view_{i}"):
                st.session_state["detail_item"] = it
                st.rerun()

def render_detail(item: Dict):
    from pathlib import Path

    st.markdown(
        f"""
        <style>
        .detail-title {{
            font-size: 1.9rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
            word-break: break-word;
            text-align: center;
        }}
        .detail-filename {{
            font-size: 0.95rem;
            color: #666;
            text-align: center;
            margin-bottom: 0.75rem;
        }}
        .detail-media-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            max-height: {DETAIL_MAX_VH}vh;
            margin-bottom: 1.5rem;
        }}
        .detail-media-wrapper img,
        .detail-media-wrapper video {{
            max-height: {DETAIL_MAX_VH}vh;
            width: 100%;
            object-fit: contain;
        }}
        .meta-section {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .meta-header {{
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }}
        .meta-key-primary {{
            font-size: 1.1rem;
            font-weight: 800;
            text-transform: uppercase;
        }}
        .meta-key-colon {{
            font-size: 1.0rem;
            font-weight: 600;
            padding: 0 0.15rem;
        }}
        .meta-value-primary {{
            font-size: 1.05rem;
            font-weight: 400;
        }}
        .meta-key {{
            font-size: 1.02rem;
            font-weight: 650;
            margin-top: 0.6rem;
            margin-bottom: 0.2rem;
        }}
        .meta-value {{
            font-size: 1.0rem;
            font-weight: 400;
            margin-left: 0.15rem;
            line-height: 1.4;
        }}
        .meta-desc {{
            font-size: 1.03rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Creative details")

    if st.button("← Back to results", key="back_to_results"):
        st.session_state["detail_item"] = None
        st.rerun()

    kind      = item.get("kind", "image")
    file_path = item.get("file", "")
    creative  = item.get("creative", "")
    labels    = item.get("labels") or {}
    tags      = item.get("tags") or []
    desc      = item.get("description", "") or ""

    image_emotion = item.get("emotion", "") or labels.get("emotion", "") or ""
    emotion_chain = item.get("emotion_chain") or []

    filename = Path(file_path).name if file_path else "—"

    st.markdown("---")

    center_left, center_main, center_right = st.columns([1, 2, 1])
    with center_main:
        st.markdown(
            f"<div class='detail-title'>{creative}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='detail-filename'>{filename}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='detail-media-wrapper'>", unsafe_allow_html=True)

        if kind == "image":
            if _safe_exists(Path(file_path)):
                st.image(file_path, use_container_width=True)
            else:
                st.error(f"Image file not found: {file_path}")
        else:
            video_path = _find_video_file_for_creative(creative)
            if video_path and _safe_exists(video_path):
                st.video(str(video_path))
            else:
                if _safe_exists(Path(file_path)):
                    st.image(file_path, use_container_width=True)
                    st.warning(
                        "No video file found for this creative; showing cover frame instead."
                    )
                else:
                    st.error("No playable video or thumbnail found for this creative.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    industry = labels.get("industry", "") or "—"
    product  = labels.get("product", "") or "—"
    vs       = item.get("video_stats", {}) or {}

    st.markdown("#### Metadata")

    if kind == "image":
        meta_html = f"""
        <div class="meta-section">
          <div class="meta-header">Core labels</div>
          <div>
            <span class="meta-key-primary">INDUSTRY</span>
            <span class="meta-key-colon">:</span>
            <span class="meta-value-primary">{industry}</span>
          </div>
          <div style="margin-top: 0.3rem;">
            <span class="meta-key-primary">PRODUCT</span>
            <span class="meta-key-colon">:</span>
            <span class="meta-value-primary">{product}</span>
          </div>
          <div style="margin-top: 0.3rem;">
            <span class="meta-key-primary">EMOTION</span>
            <span class="meta-key-colon">:</span>
            <span class="meta-value-primary">{image_emotion or "—"}</span>
          </div>

          <div class="meta-key">Keywords</div>
          <div class="meta-value">{", ".join(tags) if tags else "—"}</div>

          <div class="meta-key">Description</div>
          <div class="meta-value meta-desc">{desc if desc else "—"}</div>
        </div>
        """
    else:
        keywords = ", ".join(tags) if tags else "—"
        vdesc    = desc if desc else "—"

        if emotion_chain:
            ec_lines = []
            for e in emotion_chain:
                fn = e.get("frame_no", "—")
                em = e.get("emotion", "")
                if not em:
                    continue
                ec_lines.append(f"Frame {fn}: {em}")
            emotion_chain_html = "<br>".join(ec_lines) if ec_lines else "—"
        else:
            emotion_chain_html = "—"

        meta_html = f"""
        <div class="meta-section">
          <div class="meta-header">Core labels</div>
          <div>
            <span class="meta-key-primary">INDUSTRY</span>
            <span class="meta-key-colon">:</span>
            <span class="meta-value-primary">{industry}</span>
          </div>
          <div style="margin-top: 0.3rem;">
            <span class="meta-key-primary">PRODUCT</span>
            <span class="meta-key-colon">:</span>
            <span class="meta-value-primary">{product}</span>
          </div>

          <div class="meta-key">Keywords from frames</div>
          <div class="meta-value">{keywords}</div>

          <div class="meta-key">Video description</div>
          <div class="meta-value meta-desc">{vdesc}</div>

          <div class="meta-key">Emotion chain (frame-by-frame)</div>
          <div class="meta-value">{emotion_chain_html}</div>

          <div class="meta-key">Video stats</div>
          <div class="meta-value">
            Frames total: {vs.get('frames_total', 0)}<br>
            Frames with embeddings: {vs.get('frames_with_embeddings', 0)}
          </div>
        </div>
        """

    st.markdown(meta_html, unsafe_allow_html=True)

    st.markdown("---")  

    with st.expander("Technical details"):
        st.write(f"**Asset ID:** {item.get('asset_id', '—')}")
        st.write(f"**Caption:** {item.get('caption', '')}")
        st.write(f"**File path (thumb):** `{file_path}`")
        st.write(f"**Creative path key:** `{creative}`")
        if "frame_no" in item:
            st.write(f"**Frame no:** {item['frame_no']}")
        if "clip_score" in item:
            st.write(f"**CLIP score:** {item['clip_score']:.4f}")
        if "score" in item:
            st.write(f"**Combined rank score:** {item['score']:.4f}")

# =======================================
# MATCH / RANK HELPERS
# =======================================
def _token_overlap_score(q: str, hay: str) -> float:
    if not q:
        return 0.0
    tokens = [t.strip().lower() for t in q.split() if t.strip()]
    if not tokens:
        return 0.0
    hay_l = (hay or "").lower()
    hits = sum(1 for t in tokens if t in hay_l)
    return hits / len(tokens)

def _tag_match_score(tags_q: str, item_tags: List[str]) -> float:
    if not tags_q:
        return 0.0
    want = [t.strip().lower() for t in tags_q.split(",") if t.strip()]
    if not want:
        return 0.0
    have = [t.lower() for t in (item_tags or [])]
    if not have:
        return 0.0
    inter = len(set(want) & set(have))
    return inter / len(want)

def rank_by_clip_and_metadata(
    items: List[Dict],
    tags_q: str,
    desc_q: str,
    filename_q: str,
) -> List[Dict]:
    if not items:
        return []

    raw_scores = [float(it.get("clip_score", 0.0)) for it in items]
    s_min, s_max = min(raw_scores), max(raw_scores)

    if s_max > s_min:
        clip_norm = [(s - s_min) / (s_max - s_min) for s in raw_scores]
    else:
        clip_norm = [0.5 for _ in raw_scores]

    W_CLIP     = 0.6
    W_TAGS     = 0.25
    W_DESC     = 0.10
    W_FILENAME = 0.05

    ranked: List[Dict] = []
    for it, cn in zip(items, clip_norm):
        ts = _tag_match_score(tags_q, it.get("tags", []))
        ds = _token_overlap_score(desc_q, it.get("description", ""))
        fname = Path(it.get("file", "")).name
        fs = _token_overlap_score(filename_q, fname)

        combined = W_CLIP * cn + W_TAGS * ts + W_DESC * ds + W_FILENAME * fs
        j = dict(it)
        j["score"] = float(combined)
        ranked.append(j)

    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked

def rank_by_metadata_only(
    items: List[Dict],
    filename_q: str,
    desc_q: str,
    tags_q: str,
    industry_q: str,
    product_q: str,
    prompt_q: str,
) -> List[Dict]:
    if not items:
        return []

    W_TAGS     = 0.35
    W_DESC     = 0.25
    W_FILENAME = 0.20
    W_INDUSTRY = 0.10
    W_PRODUCT  = 0.05
    W_PROMPT   = 0.05

    ranked: List[Dict] = []
    for it in items:
        desc   = it.get("description", "") or ""
        fname  = Path(it.get("file", "")).name
        labels = it.get("labels") or {}
        industry = labels.get("industry", "") or ""
        product  = labels.get("product", "") or ""

        ts = _tag_match_score(tags_q, it.get("tags", []))
        ds = _token_overlap_score(desc_q, desc)
        fs = _token_overlap_score(filename_q, fname)
        iscore = _token_overlap_score(industry_q, industry)
        pscore = _token_overlap_score(product_q, product)
        ptext  = _token_overlap_score(prompt_q, (it.get("caption", "") + " " + desc))

        combined = (
            W_TAGS * ts
            + W_DESC * ds
            + W_FILENAME * fs
            + W_INDUSTRY * iscore
            + W_PRODUCT * pscore
            + W_PROMPT * ptext
        )
        j = dict(it)
        j["score"] = float(combined)
        ranked.append(j)

    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked

# =======================================
# CLIP SEARCH (images + videos)
# =======================================
def clip_semantic_search_images(
    prompt: str,
    top_k: int = 50,
) -> List[Dict]:
    prompt = prompt.strip()
    if not prompt:
        return []

    idx = indices.get("images")
    if idx is None or idx.ntotal == 0:
        return []

    q_vec = text_to_clip_embedding(prompt)
    k = min(top_k, idx.ntotal)
    D, I = idx.search(q_vec, k)

    asset_ids = [int(i) for i in I[0] if i != -1]
    if not asset_ids:
        return []

    asset_scores: Dict[int, float] = {}
    for score, idx_id in zip(D[0], I[0]):
        if idx_id == -1:
            continue
        aid = int(idx_id)
        if aid not in asset_scores:
            asset_scores[aid] = float(score)
        else:
            asset_scores[aid] = max(asset_scores[aid], float(score))

    placeholders = ",".join("?" for _ in asset_ids)
    rows = con.execute(
        f"SELECT id, path FROM assets WHERE id IN ({placeholders})",
        asset_ids,
    ).fetchall()
    id_to_path = {int(r[0]): r[1] for r in rows}

    items: List[Dict] = []
    for aid in asset_ids:
        path_str = id_to_path.get(aid)
        if not path_str:
            continue
        media_items = fetch_media(path_str, "Images", limit=24)
        clip_score = asset_scores.get(aid, 0.0)
        for m in media_items:
            j = dict(m)
            j["clip_score"] = float(clip_score)
            items.append(j)

    return items

def clip_semantic_search_videos(
    prompt: str,
    top_k: int = 50,
) -> List[Dict]:
    prompt = prompt.strip()
    if not prompt:
        return []

    idx = indices.get("frames")
    if idx is None or idx.ntotal == 0:
        return []

    q_vec = text_to_clip_embedding(prompt)
    k = min(top_k * 3, idx.ntotal)
    D, I = idx.search(q_vec, k)

    frame_ids = [int(i) for i in I[0] if i != -1]
    if not frame_ids:
        return []

    frame_scores: Dict[int, float] = {}
    for score, fid in zip(D[0], I[0]):
        if fid == -1:
            continue
        frame_scores[int(fid)] = float(score)

    placeholders = ",".join("?" for _ in frame_ids)
    rows = con.execute(
        f"""
        SELECT
            f.id,
            f.asset_id,
            f.frame_no,
            f.path,
            a.path AS creative_path
        FROM frames f
        JOIN assets a
          ON a.id = f.asset_id
        WHERE f.id IN ({placeholders})
        """,
        frame_ids,
    ).fetchall()

    by_frame_id = {int(r[0]): r for r in rows}

    seen_assets = set()
    ranked_assets: List[Tuple[int, str]] = []
    asset_best_score: Dict[int, float] = {}

    for fid in frame_ids:
        row = by_frame_id.get(fid)
        if not row:
            continue
        _fid, asset_id, frame_no, rel_path, creative_path = row
        s = frame_scores.get(fid, 0.0)
        if asset_id not in asset_best_score:
            asset_best_score[asset_id] = s
        else:
            asset_best_score[asset_id] = max(asset_best_score[asset_id], s)

        if asset_id in seen_assets:
            continue
        seen_assets.add(asset_id)
        ranked_assets.append((asset_id, creative_path))
        if len(ranked_assets) >= top_k:
            break

    items: List[Dict] = []
    for asset_id, creative_path in ranked_assets:
        vids = fetch_media(creative_path, "Videos", limit=1)
        clip_score = asset_best_score.get(asset_id, 0.0)
        for v in vids:
            j = dict(v)
            j["clip_score"] = float(clip_score)
            items.append(j)

    return items

# =======================================
# SESSION STATE INIT
# =======================================
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
if "results_header" not in st.session_state:
    st.session_state["results_header"] = ""
if "detail_item" not in st.session_state:
    st.session_state["detail_item"] = None

# =======================================
# SEARCH PAGE RENDERER
# =======================================
def render_search_page():
    # Header
    st.markdown(
        """
        <div style="text-align: center; padding-top: 0.5rem; padding-bottom: 1rem;">
          <h1 style="
                margin-bottom: 0.2rem;
                font-size: 2.8rem;
                font-weight: 750;
                letter-spacing: 0.03em;
                font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          ">
            Creative Search
          </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    outer_left, center_col, outer_right = st.columns([1, 2.5, 1])

    with center_col:
        search_mode = st.radio(
            "Search mode",
            ["Simple", "Advanced"],
            horizontal=True,
            key="search_mode_radio",
        )

        st.markdown(
            """
            <div style="
                border-radius: 0.75rem;
                padding: 1.25rem 1.5rem;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
                background-color: #ffffff;
                border: 1px solid #dddddd;
            ">
            """,
            unsafe_allow_html=True,
        )

        with st.form("search_form", clear_on_submit=False):
            if search_mode == "Simple":
                search_query = st.text_input(
                    "Search",
                    placeholder="Filename, caption, etc.",
                    key="search_query_simple",
                )

                col_s1, col_s2 = st.columns([3, 1])
                with col_s1:
                    media_type = st.selectbox(
                        "Media type",
                        ["Both", "Images", "Videos"],
                        key="search_media_type_simple",
                    )
                with col_s2:
                    max_results = st.slider(
                        "Max results",
                        min_value=5,
                        max_value=100,
                        value=25,
                        step=5,
                        key="max_results_simple",
                    )

                prompt_q   = search_query
                filename_q = search_query
                tags_q     = ""
                desc_q     = ""
                industry_q = ""
                product_q  = ""
                use_clip   = False

            else:
                st.markdown(
                    "<p style='font-size:0.95rem; color:#666; margin-bottom:0.3rem;'>"
                    "Describe the creative and refine with metadata signals.</p>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div style='font-weight:600; font-size:0.95rem; margin-top:0.2rem;'>Semantic search</div>",
                    unsafe_allow_html=True,
                )
                prompt_q = st.text_input(
                    "",
                    placeholder="e.g., 'family eating dinner at home, warm lighting'",
                    key="search_prompt_q_adv",
                    label_visibility="collapsed",
                )
                use_clip = st.checkbox(
                    "Use semantic search (images + video frames)",
                    value=True,
                    key="use_clip_checkbox_adv",
                )

                st.markdown("<hr style='margin:0.7rem 0 0.4rem 0;'>", unsafe_allow_html=True)

                st.markdown(
                    "<div style='font-weight:600; font-size:0.95rem; margin-bottom:0.1rem;'>Metadata</div>",
                    unsafe_allow_html=True,
                )
                meta_col1, meta_col2 = st.columns(2)
                with meta_col1:
                    industry_q = st.text_input(
                        "Industry",
                        placeholder="e.g., insurance, retail, telco",
                        key="search_industry_q_adv",
                    )
                    product_q = st.text_input(
                        "Product",
                        placeholder="e.g., home insurance, credit card",
                        key="search_product_q_adv",
                    )
                with meta_col2:
                    tags_q = st.text_input(
                        "Tags (comma separated)",
                        placeholder="e.g., logo, outdoor, education",
                        key="search_tags_q_adv",
                    )
                    desc_q = st.text_input(
                        "Description contains",
                        placeholder="Words in internal description…",
                        key="search_desc_q_adv",
                    )

                st.markdown("<hr style='margin:0.7rem 0 0.4rem 0;'>", unsafe_allow_html=True)

                tech_col1, tech_col2, tech_col3 = st.columns([2, 1, 1])
                with tech_col1:
                    filename_q = st.text_input(
                        "Filename contains",
                        placeholder="Partial filename or ID…",
                        key="search_filename_q_adv",
                    )
                with tech_col2:
                    media_type = st.selectbox(
                        "Media type",
                        ["Both", "Images", "Videos"],
                        key="search_media_type_adv",
                    )
                with tech_col3:
                    max_results = st.slider(
                        "Max results",
                        min_value=5,
                        max_value=100,
                        value=25,
                        step=5,
                        key="max_results_adv",
                    )

            btn_col_left, btn_col_center, btn_col_right = st.columns([1, 2, 1])
            with btn_col_center:
                submitted = st.form_submit_button("Search")

        st.markdown("</div>", unsafe_allow_html=True)

    # =======================================
    # HANDLE SEARCH
    # =======================================
    if 'submitted' in locals() and submitted:
        if 'use_clip' in locals() and use_clip and prompt_q.strip():
            clip_items: List[Dict] = []

            if media_type in ("Videos", "Both"):
                clip_items.extend(clip_semantic_search_videos(prompt_q, top_k=max_results))

            if media_type in ("Images", "Both"):
                clip_items.extend(clip_semantic_search_images(prompt_q, top_k=max_results))

            results = rank_by_clip_and_metadata(
                clip_items,
                tags_q=tags_q,
                desc_q=desc_q,
                filename_q=filename_q,
            )

            # Respect user-selected max results
            results = results[:max_results]

            header = (
                f"### Results\n"
                f"Found **{len(results)}** items for "
                f"`{prompt_q.strip()}` (semantic + metadata)."
            )

        else:
            base_pool = fetch_media(None, media_type, limit=2000)

            results = rank_by_metadata_only(
                base_pool,
                filename_q=filename_q,
                desc_q=desc_q,
                tags_q=tags_q,
                industry_q=industry_q if 'industry_q' in locals() else "",
                product_q=product_q if 'product_q' in locals() else "",
                prompt_q=prompt_q or "",
            )

            # Respect user-selected max results
            results = results[:max_results]

            if search_mode == "Simple":
                header = f"### Results\nFound **{len(results)}** items (metadata only)."
            else:
                header = f"### Results\nFound **{len(results)}** items (metadata-ranked)."

        st.session_state["last_results"]   = results
        st.session_state["results_header"] = header
        st.session_state["detail_item"]    = None

    # =======================================
    # RENDER RESULTS OR DETAIL VIEW
    # =======================================
    results      = st.session_state.get("last_results", [])
    detail_item  = st.session_state.get("detail_item")
    results_hdr  = st.session_state.get("results_header", "")

    st.markdown("---")

    if detail_item:
        render_detail(detail_item)
    elif results:
        if results_hdr:
            st.markdown(results_hdr)
        grid_gallery(results)
    else:
        st.info("Run a search to see creatives.")

# =======================================
# HOME PAGE RENDERER (placeholder)
# =======================================
def render_home_page():
    # Hero section
    st.markdown(
        """
        <div style="text-align: center; padding-top: 1.0rem; padding-bottom: 1.0rem;">
          <h1 style="
                margin-bottom: 0.3rem;
                font-size: 2.6rem;
                font-weight: 750;
                letter-spacing: 0.03em;
                font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          ">
            Creative Intelligence Hub
          </h1>
          <p style="font-size:1.0rem; color:#555; max-width: 680px; margin: 0 auto;">
            Explore and search ad creatives using vision–language AI. 
            Qwen describes each asset, detects emotions, and generates rich tags so you can 
            analyse campaigns at scale.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Flow chart + short explanation
    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown(
            """
            <h3 style="margin-top:0;">How the pipeline works</h3>
            <p style="font-size:0.98rem; color:#444;">
              Every creative (image or video) is processed through a single AI pipeline:
            </p>
            <ul style="font-size:0.95rem; color:#444; line-height:1.4;">
              <li><b>Qwen</b> looks at the image or video frames and explains what is happening in natural language.</li>
              <li>From these descriptions we extract <b>tags</b> – industry, mood, products, objects, actions and scenes.</li>
              <li>For videos, we pick only the most important frames and build a <b>video-level story</b> and emotion flow.</li>
              <li>All of this is stored in a database and indexed with CLIP so you can <b>search creatives by text</b>.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        flow_path = PROJECT_ROOT / "Flow_chart.png"
        if _safe_exists(flow_path):
            st.image(str(flow_path), use_container_width=True, caption="End-to-end creative analysis flow")
        else:
            st.info("Flow_chart.png not found next to app.py – add it to show the pipeline diagram here.")

    st.markdown("---")

    # Cards for Qwen, Emotions, Tags, Video descriptions
    st.markdown(
        """
        <style>
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            grid-gap: 1.0rem;
            margin-top: 0.5rem;
        }
        .info-card {
            border-radius: 0.9rem;
            border: 1px solid #e2e2e2;
            padding: 0.9rem 1.0rem;
            background: #fafafa;
        }
        .info-card h4 {
            margin-top: 0;
            margin-bottom: 0.35rem;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .info-card p {
            margin: 0;
            font-size: 0.93rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What the system captures")

    st.markdown(
        """
        <div class="card-grid">
          <div class="info-card">
            <h4>Qwen (vision–language AI)</h4>
            <p>
              Qwen is a modern AI model that understands both images and text. 
              It looks at images or video frames, recognises what’s happening, and explains it in natural language. 
              We use Qwen to automatically describe creatives and generate useful tags 
              <b>(industry, mood, products, descriptions)</b> so we can search and analyse ads at scale.
            </p>
          </div>

          <div class="info-card">
            <h4>Emotion detection</h4>
            <p>
              Emotion detection identifies the mood an ad is trying to create — joy, excitement, calmness, nostalgia, urgency, and more. 
              Ads can express a wide range of feelings, from positive (happy, warm, comforting) to intense (dramatic, tense, sad). 
              In videos, multiple emotions appear in sequence, shifting from one mood to another as the story develops.
            </p>
          </div>

          <div class="info-card">
            <h4>Video descriptions</h4>
            <p>
              Video descriptions summarise what’s happening across a clip — the actions, setting, people, products, and overall story. 
              They give a high-level narrative, capturing how the message unfolds over time. 
              We generate them using Qwen on only the important frames, so you see the key story beats without noise.
            </p>
          </div>

          <div class="info-card">
            <h4>Tags & searchable metadata</h4>
            <p>
              Tags capture key elements in an ad — objects, actions, settings, products, colours, themes, and brand cues. 
              They provide a quick, searchable summary of what appears in the creative. 
              In videos, tags can change scene-by-scene, reflecting shifts in objects, locations, or activities.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Final “how to use this app” strip
    st.markdown(
        """
        <div style="
            max-width: 900px;
            margin: 0.5rem auto 0 auto;
            padding: 0.9rem 1.1rem;
            border-radius: 0.8rem;
            border: 1px solid #e2e2e2;
            background: #fdfdfd;
            font-size: 0.95rem;
        ">
          <b>How to use this hub</b>
          <ul style="margin-top: 0.4rem; padding-left: 1.1rem;">
            <li>Go to the <b>Search</b> page to find creatives by description, tags, industry or product.</li>
            <li>Click <b>View details</b> on any result to see full metadata, tags, descriptions and (for videos) the emotion chain.</li>
            <li>Use this to explore campaigns, compare ads across industries, or quickly surface examples for creative reviews.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =======================================
# SIDEBAR NAVIGATION (Home / Search)
# =======================================
with st.sidebar:
    # Clemenger BBDO logo at top of nav
    logo_path = PROJECT_ROOT / "ClemengerBBDO.webp"
    if _safe_exists(logo_path):
        st.image(str(logo_path), use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    st.header("Navigate")
    page = st.radio(
        "Pages",
        ["Home", "Search"],
        index=1,   # default to Search
        key="nav_page",
    )

# =======================================
# PAGE ROUTING
# =======================================
if page == "Home":
    render_home_page()
else:  # "Search"
    render_search_page()
