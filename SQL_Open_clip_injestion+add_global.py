# QL_Open_clip_ingestion_add_global.py
# Drop-in OpenCLIP ingestion for your pipeline.
# - Per-creative:  <creative>/clip_embeddings.json
# - Global index:  data/index/clip_index.{csv,parquet}
# - Cached text feats: data/index/clip_text_features.pt

from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

import torch
from PIL import Image, UnidentifiedImageError

# Optional dependency hint (kept lightweight on import)
try:
    import open_clip
except Exception as _e:
    open_clip = None

# -----------------
# CONFIG DEFAULTS
# -----------------
DATA_INPUT_ROOT = Path.cwd() / "data" / "input"
INDEX_DIR       = Path.cwd() / "data" / "index"
EXTS            = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

DEFAULT_PROMPTS = [
    # Objects / electronics
    "men's smartwatch product photo",
    "women's smartwatch fashion ad",
    "Android smartphone close-up",
    "Samsung Galaxy smartphone",
    "tablet computer on table",
    "wireless headphones product shot",
    "laptop open on desk",
    "smartwatch fitness features",
    "charging cable and accessories",
    "electronics retail display",
    # Occasions / themes
    "Father's Day sale banner",
    "thank you dad greeting card",
    "gift for dad",
    "Mother's Day gift idea",
    "Valentine's Day couple advertisement",
    "back-to-school electronics sale",
    # Retail / offer language
    "limited time offer sticker",
    "big discount badge",
    "percentage off label",
    "clearance sale banner",
    "special offer tag",
    "shop now call-to-action",
    "new arrivals banner",
    "bundle deal promotion",
    "thank you message card",
]

# ---- env caps helpful for stability ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _emit(msg: str, report):
    if report: report(msg)
    else: print(msg)

def _list_files(dir_path: Path) -> list[str]:
    if not dir_path.is_dir(): return []
    out = []
    for e in EXTS:
        out.extend(str(p) for p in dir_path.glob(f"*{e}"))
    return sorted(out)

def _collect_creatives(base: Path) -> dict[Path, Dict[str, list[str]]]:
    """
    Returns: { creative_path: {"IMG-ORIG":[...], "VID-FRAMES":[...]} }
    """
    grouped: dict[Path, Dict[str, list[str]]] = {}

    images_root = base / "images"
    if images_root.is_dir():
        for creative in sorted(p for p in images_root.iterdir() if p.is_dir()):
            origs = _list_files(creative)
            if origs:
                grouped.setdefault(creative, {"IMG-ORIG": [], "VID-FRAMES": []})
                grouped[creative]["IMG-ORIG"].extend(origs)

    videos_root = base / "videos"
    if videos_root.is_dir():
        for creative in sorted(p for p in videos_root.iterdir() if p.is_dir()):
            fm = creative / "frames_medoids"
            if not fm.is_dir():
                fm = creative / "frames_medoid"
            frames = _list_files(fm)
            if frames:
                grouped.setdefault(creative, {"IMG-ORIG": [], "VID-FRAMES": []})
                grouped[creative]["VID-FRAMES"].extend(frames)

    return grouped

def _safe_open_rgb(path: str) -> Image.Image | None:
    try:
        im = Image.open(path).convert("RGB")
        return im
    except (UnidentifiedImageError, OSError):
        return None

@torch.no_grad()
def process_all_input(
    report: Optional[callable] = None,
    *,
    device: str = "cpu",
    batch_size: int = 16,
    top_k: int = 5,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    prompts: Optional[List[str]] = None,
    print_vectors: bool = False,
) -> Iterable[str]:
    """
    Generator that:
      1) Loads/caches text embeddings
      2) Walks data/input creatives
      3) Writes per-creative clip_embeddings.json
      4) Appends to global index at data/index/clip_index.{csv,parquet}
      5) Saves text cache under data/index/clip_text_features.pt
    """
    if open_clip is None:
        msg = "OpenCLIP not installed; pip install open-clip-torch"
        _emit(msg, report); yield msg
        return

    if not DATA_INPUT_ROOT.is_dir():
        msg = f"Expected folder not found: {DATA_INPUT_ROOT}"
        _emit(msg, report); yield msg
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --------- Model & preprocess ----------
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
    except TypeError:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # --------- Prompts & text feats cache ----------
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()

    # Optional debug
    if print_vectors:
        _emit("# === Prompt token IDs ===", report); yield "# prompt token ids below"
        for p, ids in zip(prompts, text_tokens.detach().cpu().tolist()):
            _emit(f"- {p}\n  token_ids = {ids}", report); yield "."

        _emit("# === Prompt embeddings (normalized) ===", report); yield "# prompt embeddings below"
        for p, vec in zip(prompts, text_feats.detach().cpu().tolist()):
            _emit(f"- {p}\n  vec[len={len(vec)}] (hidden)", report); yield "."

    # Save/cache text features for other components to reuse
    cache_pt = INDEX_DIR / "clip_text_features.pt"
    torch.save(
        {
            "prompts": prompts,
            "text_tokens": text_tokens.detach().cpu(),
            "text_features": text_feats.detach().cpu(),
            "model_name": model_name,
            "pretrained": pretrained,
            "logit_scale": float(logit_scale.item()),
        },
        cache_pt,
    )
    msg_cache = f"[cache] Saved text features → {cache_pt}"
    _emit(msg_cache, report); yield msg_cache

    # --------- Discover creatives ----------
    grouped = _collect_creatives(DATA_INPUT_ROOT)
    if not grouped:
        msg = f"No creatives found under {DATA_INPUT_ROOT}"
        _emit(msg, report); yield msg
        return

    _emit("Discovered creatives:", report); yield "Discovered creatives:"
    for c, sets in grouped.items():
        _emit(f"  {c}\n    IMG-ORIG:   {len(sets['IMG-ORIG'])}\n    VID-FRAMES: {len(sets['VID-FRAMES'])}", report)
        yield "."

    # Global aggregation (kept in memory, then flushed once)
    global_rows: list[Dict[str, Any]] = []

    # --------- Per-creative pass ----------
    for creative, sets in grouped.items():
        records: list[Dict[str, Any]] = []

        def embed_paths(paths: list[str], source_key: str):
            if not paths: return
            # micro-batching
            for i in range(0, len(paths), batch_size):
                chunk = paths[i:i+batch_size]

                imgs = []
                valid_paths = []
                for p in chunk:
                    im = _safe_open_rgb(p)
                    if im is None:
                        _emit(f"[warn] Skipping unreadable image: {p}", report); continue
                    imgs.append(preprocess(im))
                    valid_paths.append(p)

                if not imgs:
                    continue

                images = torch.stack(imgs, 0).to(device)
                img_feats = model.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                sims  = img_feats @ text_feats.T
                probs = torch.softmax(logit_scale * sims, dim=1)

                for b, path in enumerate(valid_paths):
                    sim_b  = sims[b].detach().cpu()
                    prob_b = probs[b].detach().cpu()
                    order  = torch.argsort(sim_b, descending=True)
                    top    = order[:top_k].tolist()

                    rec = {
                        "file": str(Path(path).relative_to(creative)),
                        "source_set": source_key,
                        "embedding": img_feats[b].detach().cpu().tolist(),
                        "best_prompt": prompts[top[0]],
                        "best_sim": float(sim_b[top[0]]),
                        "best_prob": float(prob_b[top[0]]),
                        "topk": [
                            {
                                "prompt": prompts[i_top],
                                "sim": float(sim_b[i_top]),
                                "prob": float(prob_b[i_top]),
                            } for i_top in top
                        ],
                        "all_scores": {
                            prompts[i]: {"sim": float(sim_b[i]), "prob": float(prob_b[i])}
                            for i in range(len(prompts))
                        },
                    }
                    records.append(rec)

                    # add to global summary
                    global_rows.append({
                        "creative": str(creative),
                        "relative_file": str(Path(path).relative_to(creative)),
                        "source_set": source_key,
                        "best_prompt": rec["best_prompt"],
                        "best_sim": rec["best_sim"],
                        "best_prob": rec["best_prob"],
                        "model_name": model_name,
                        "pretrained": pretrained,
                    })

        # do both sources
        embed_paths(sets.get("IMG-ORIG", []), "IMG-ORIG")
        embed_paths(sets.get("VID-FRAMES", []), "VID-FRAMES")

        # Save creative JSON
        out_json = creative / "clip_embeddings.json"
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            _emit(f"[save] {len(records)} embedding record(s) → {out_json}", report)
            yield "."
        except Exception as e:
            _emit(f"[save] Failed to write JSON for {creative}: {e}", report)
            yield "."

    # --------- Flush global index ----------
    # Avoid importing pandas unless we actually have rows to write.
    if global_rows:
        try:
            import pandas as pd
            df = pd.DataFrame(global_rows)
            csv_p = INDEX_DIR / "clip_index.csv"
            pq_p  = INDEX_DIR / "clip_index.parquet"
            df.to_csv(csv_p, index=False)
            try:
                df.to_parquet(pq_p, index=False)
                _emit(f"[index] Wrote global index → {csv_p} and {pq_p}", report); yield "[index parquet]"
            except Exception as e:
                _emit(f"[index] Wrote CSV (Parquet failed: {e}) → {csv_p}", report); yield "[index csv]"
        except Exception as e:
            _emit(f"[index] Failed to write global index: {e}", report); yield "[index fail]"
    else:
        _emit("[index] No rows to write (no embeddings produced).", report); yield "[index empty]"

    _emit("[done] OpenCLIP ingestion complete.", report); yield "[done]"

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OpenCLIP ingestion over data/input/* creatives.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--print-vectors", action="store_true")
    args = parser.parse_args()

    for _ in process_all_input(
        report=None,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        model_name=args.model,
        pretrained=args.pretrained,
        print_vectors=args.print_vectors,
    ):
        pass
