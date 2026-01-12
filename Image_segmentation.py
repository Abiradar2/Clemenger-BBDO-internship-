# Image_segmentation.py
# Batch Mask2Former over data/input/{images, videos/*/frames_medoid} + single-image demo
from __future__ import annotations

# --- add this at the very top, before importing transformers ---
import os, sys, types
from importlib.machinery import ModuleSpec

# Keep HF from touching torchvision (must be set BEFORE importing transformers)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# If something preloaded torchvision, purge it
for k in list(sys.modules):
    if k == "torchvision" or k.startswith("torchvision."):
        del sys.modules[k]

# Stub torchvision minimally so any accidental imports succeed without pulling onnx/ml_dtypes
tv = types.ModuleType("torchvision"); tv.__spec__ = ModuleSpec("torchvision", None, is_package=True); tv.__path__ = []
tvt = types.ModuleType("torchvision.transforms"); tvt.__spec__ = ModuleSpec("torchvision.transforms", None, is_package=True); tvt.__path__ = []

class InterpolationMode:  # transformers references this enum sometimes
    NEAREST=0; BOX=1; BILINEAR=2; BICUBIC=3; LANCZOS=4; HAMMING=5
tvt.InterpolationMode = InterpolationMode

class Compose:
    def __init__(self, transforms): self.transforms = list(transforms) if transforms else []
    def __call__(self, x):
        for t in self.transforms: x = t(x)
        return x
tvt.Compose = Compose

class _NoOp:
    def __init__(self, *a, **k): pass
    def __call__(self, x): return x
for _n in ["Resize","CenterCrop","Normalize","ToTensor","RandomResizedCrop","ColorJitter","RandomHorizontalFlip"]:
    setattr(tvt, _n, _NoOp)

tvf = types.ModuleType("torchvision.transforms.functional")
tvf.__spec__ = ModuleSpec("torchvision.transforms.functional", None, is_package=False)
def _noop(*a, **k): return a[0] if a else None
for _n in ["resize","center_crop","normalize","pad","to_pil_image","pil_to_tensor","convert_image_dtype"]:
    setattr(tvf, _n, _noop)

sys.modules["torchvision"] = tv
sys.modules["torchvision.transforms"] = tvt
sys.modules["torchvision.transforms.functional"] = tvf
tvt.functional = tvf
# --- end stub ---

import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from typing import Optional, Tuple, Iterable, Callable, List

import numpy as np
from PIL import Image
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# ===================== Project + Local Model =====================
PROJECT_ROOT = Path("/Users/aditya.biradar/Library/CloudStorage/OneDrive-OneWorkplace/Desktop/WORK")
MODEL_DIR    = PROJECT_ROOT / "models" / "m2f-panoptic"   # must contain config.json + weights

# --- Single-image demo defaults (kept for compatibility/testing) ---
IMAGE_PATH   = PROJECT_ROOT / "Samsung_first_frame.png"
OUT_DIR_DEMO = PROJECT_ROOT / "m2f_objects"

# Offline cache (optional but tidy)
HF_CACHE = PROJECT_ROOT / "data" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))

# Force local-only (no Hub calls / no safetensors conversion PRs)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===================== Config =====================
DEVICE     = "cpu"          # "cuda" if you have a GPU
TASK       = "instance"     # "instance" | "panoptic"

USE_ALPHA_MASK = True       # True → transparent background in cutout .png

# Absolute minimum instance area in pixels (tiny specks are dropped)
MIN_PIX_AREA   = 5000

# Optional relative area filter (fraction of full image area).
# If > 0, the effective threshold becomes: max(MIN_PIX_AREA, REL_MIN_AREA * W * H)
REL_MIN_AREA   = 0.02        # e.g., 0.002 means keep masks ≥ 0.2% of the image

# Confidence filter (instance seg has per-instance scores; panoptic may not)
SCORE_THRESHOLD = 0.90

# Cutout buffer around object bbox (affects cutout and optional buffered context)
BUFFER_RATIO   = 0.20       # 20% of image width/height
BUFFER_PX      = 0          # if >0, overrides ratio
MIN_BOX_SIDE   = 0          # if >0, enforce min crop side in px

# -------- Assign every background pixel to its nearest object (centroid Voronoi) --------
DO_ASSIGN_BACKGROUND   = False  # set True to append all unmasked pixels to nearest object
DOWNSCALE_FOR_ASSIGN   = 1      # 1 = full-res; 2 or 4 reduces memory/compute for very large images
TIE_BREAK_BY_SCORE     = True   # if two objects equally near, prefer higher-score one

# Also save a visible "object + surrounding pixels" (no transparency) using the buffered box
DO_BUFFERED_CONTEXT = False
BUFFERED_CONTEXT_FORMAT = "jpg"  # "jpg" or "png"

# Fixed-size context crop (same size for every object, centered on object)
CONTEXT_BOX_W_RATIO = 0.60  # 60% of full image width
CONTEXT_BOX_H_RATIO = 0.60  # 60% of full image height

# File filters (align with Upload.py)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# ===================== Guards / Validation =====================
def _fail(msg: str):
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(2)

def _validate_model():
    model_dir = Path(MODEL_DIR)
    if not model_dir.is_dir():
        _fail(f"MODEL_DIR not found: {model_dir}")

    cfg = model_dir / "config.json"
    has_bin = (model_dir / "pytorch_model.bin").is_file()
    has_st  = (model_dir / "model.safetensors").is_file()
    if not cfg.is_file() or not (has_bin or has_st):
        _fail(
            "MODEL_DIR is missing required files:\n"
            f"  - config.json present? {cfg.is_file()}\n"
            f"  - pytorch_model.bin present? {has_bin}\n"
            f"  - model.safetensors present? {has_st}\n"
            f"MODEL_DIR = {model_dir}"
        )
    return model_dir, has_st

# ===================== Geometry / Save Helpers =====================
def mask_area(m_bool: np.ndarray) -> int:
    return int(m_bool.sum())

def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W, int(x1)); y1 = min(H, int(y1))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

def tight_bbox(mask_bool: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return (x0, y0, x1, y1)

def expand_box_to_min_side(x0, y0, x1, y1, min_side, W, H):
    w = x1 - x0; h = y1 - y0
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    nw = max(w, min_side); nh = max(h, min_side)
    x0n = int(round(cx - nw / 2.0)); x1n = int(round(cx + nw / 2.0))
    y0n = int(round(cy - nh / 2.0)); y1n = int(round(cy + nh / 2.0))
    return clamp_box(x0n, y0n, x1n, y1n, W, H)

def buffered_bbox_from_mask(mask_bool, W, H, buffer_ratio=BUFFER_RATIO, buffer_px=BUFFER_PX, min_side=MIN_BOX_SIDE):
    tb = tight_bbox(mask_bool)
    if tb is None:
        return None
    x0, y0, x1, y1 = tb

    if buffer_px > 0:
        bx = by = buffer_px
    else:
        bx = int(round(buffer_ratio * W))
        by = int(round(buffer_ratio * H))

    x0 -= bx; x1 += bx
    y0 -= by; y1 += by
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

    if min_side and min_side > 0:
        x0, y0, x1, y1 = expand_box_to_min_side(x0, y0, x1, y1, min_side, W, H)
    return (x0, y0, x1, y1)

def fixed_context_box_around_object(tb, W, H, wr, hr):
    """Fixed-size context crop centered on the object's bbox center; clamped to image bounds."""
    x0, y0, x1, y1 = tb
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    cw = max(1, int(round(wr * W)))
    ch = max(1, int(round(hr * H)))
    x0n = int(round(cx - cw / 2.0)); x1n = x0n + cw
    y0n = int(round(cy - ch / 2.0)); y1n = y0n + ch
    return clamp_box(x0n, y0n, x1n, y1n, W, H)

def cutout_from_mask(img_np, mask_bool, bbox, use_alpha=True):
    x0, y0, x1, y1 = bbox
    crop = img_np[y0:y1, x0:x1].copy()
    if use_alpha:
        seg_crop = mask_bool[y0:y1, x0:x1]
        rgba = np.dstack([crop, (seg_crop.astype(np.uint8) * 255)])
        return Image.fromarray(rgba).convert("RGBA")
    else:
        return Image.fromarray(crop).convert("RGB")

# ===================== Model Load & Core Segmentation =====================
def _load_local_model(model_dir: Path, use_safetensors_if_present: bool, device: str):
    processor = Mask2FormerImageProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_safetensors=use_safetensors_if_present,
    ).to(device).eval()
    return processor, model

def _augment_masks_assign_background(
    masks_bool: np.ndarray,           # [K, H, W] booleans
    scores: np.ndarray,               # [K] floats
) -> np.ndarray:
    """Assign ALL background pixels to nearest instance (centroid/Voronoi)."""
    K, H, W = masks_bool.shape
    if K == 0:
        return masks_bool

    owned = masks_bool.any(0)  # [H,W] claimed
    bg = ~owned
    if not bg.any():
        return masks_bool

    centroids = []
    for k in range(K):
        ys, xs = np.where(masks_bool[k])
        if len(ys) == 0:
            centroids.append((1e9, 1e9))
        else:
            centroids.append((ys.mean(), xs.mean()))
    centroids = np.array(centroids, dtype=np.float32)  # [K,2]

    ds = max(1, int(DOWNSCALE_FOR_ASSIGN))
    if ds > 1:
        h, w = (H + ds - 1) // ds, (W + ds - 1) // ds
        yy, xx = np.mgrid[0:h, 0:w]
        yy = (yy * ds).astype(np.float32)
        xx = (xx * ds).astype(np.float32)
        d2 = (yy[None, ...] - centroids[:, 0:1, None])**2 + (xx[None, ...] - centroids[:, 1:2, None])**2
        nearest_small = d2.argmin(axis=0)  # [h,w]
        nearest_full = nearest_small.repeat(ds, 0).repeat(ds, 1)[:H, :W]
    else:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d2 = (yy[None, ...] - centroids[:, 0:1, None])**2 + (xx[None, ...] - centroids[:, 1:2, None])**2
        nearest_full = d2.argmin(axis=0)  # [H,W]

    if TIE_BREAK_BY_SCORE:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ranks = (-scores).argsort().argsort().astype(np.float32)  # 0 is best
        eps = (ranks / max(1, K-1)) * 1e-6
        d2 = (yy[None, ...] - centroids[:, 0:1, None])**2 + (xx[None, ...] - centroids[:, 1:2, None])**2
        d2 = d2 + eps[:, None, None]
        nearest_full = d2.argmin(axis=0)

    augmented = masks_bool.copy()
    for k in range(K):
        take = (nearest_full == k) & bg
        if take.any():
            augmented[k, take] = True
    return augmented

def _effective_min_area(W: int, H: int) -> int:
    if REL_MIN_AREA and REL_MIN_AREA > 0.0:
        return max(MIN_PIX_AREA, int(REL_MIN_AREA * (W * H)))
    return MIN_PIX_AREA

def _segment_one_image(
    img_path: Path,
    out_dir: Path,
    processor: Mask2FormerImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
    task: str = TASK,
    use_alpha: bool = USE_ALPHA_MASK,
    min_pix_area: int = MIN_PIX_AREA,   # kept for API compat; overridden per-image by REL_MIN_AREA if set
) -> int:
    """
    Segment a single image and write cutouts (+ optional contexts) into out_dir.
    If DO_ASSIGN_BACKGROUND is True, background pixels are appended to the nearest object.
    Returns number of kept objects.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[seg] ! failed to open {img_path}: {e}")
        return 0

    W, H = img.size
    np_img = np.array(img)

    # Per-image effective area threshold (supports REL_MIN_AREA)
    eff_min_area = max(min_pix_area, _effective_min_area(W, H))

    inputs = processor(images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [img.size[::-1]]
    if task == "panoptic":
        processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)[0]
        seg = processed["segmentation"].cpu().numpy()
        segments_info = processed["segments_info"]
        KEEP_STUFF = False
        keep_ids = [s["id"] for s in segments_info if (s.get("isthing", True) or KEEP_STUFF)]
    else:
        processed = processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)[0]
        seg = processed["segmentation"].cpu().numpy()
        segments_info = processed["segments_info"]
        keep_ids = [s["id"] for s in segments_info]

    if len(segments_info) == 0:
        return 0

    # Collect instance masks + metadata (filter by score + area)
    masks, scores, meta = [], [], []
    for s in segments_info:
        if s["id"] not in keep_ids:
            continue

        # Score exists for instance; for panoptic it may be missing → default 1.0
        score = float(s.get("score", 1.0))
        if score < SCORE_THRESHOLD:
            continue

        inst_mask = (seg == s["id"])
        if mask_area(inst_mask) < eff_min_area:
            continue

        masks.append(inst_mask)
        scores.append(score)
        meta.append(s)

    if len(masks) == 0:
        return 0

    masks_bool = np.stack(masks, axis=0)  # [K,H,W]
    scores_np  = np.array(scores, dtype=np.float32)

    # Optional: assign background to nearest object
    if DO_ASSIGN_BACKGROUND:
        masks_bool = _augment_masks_assign_background(masks_bool, scores_np)

    id2label = model.config.id2label
    kept = 0
    stem_base = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-object outputs using (possibly augmented) masks
    for k, s in enumerate(meta):
        inst_mask = masks_bool[k]
        if mask_area(inst_mask) < 1:
            continue  # extremely tiny after filtering

        label_id = int(s["label_id"])
        class_name = id2label.get(label_id, f"id{label_id}")
        score = float(scores[k])
        safe = class_name.replace(" ", "_").replace("/", "_")
        stem = f"{stem_base}__obj_{kept:02d}_{safe}_score{score:.2f}"

        # (A) masked cutout with buffer (background transparent if use_alpha=True)
        cut_bbox = buffered_bbox_from_mask(inst_mask, W, H)
        if cut_bbox is None:
            continue

        cut = cutout_from_mask(np_img, inst_mask, cut_bbox, use_alpha=use_alpha)
        if use_alpha:
            cut.save(out_dir / f"{stem}.png")
        else:
            cut.convert("RGB").save(out_dir / f"{stem}.jpg", quality=95)

        # (B) OPTIONAL: visible buffered context (same box as cutout; no transparency)
        if DO_BUFFERED_CONTEXT:
            x0, y0, x1, y1 = cut_bbox
            ctx_img = Image.fromarray(np_img[y0:y1, x0:x1]).convert("RGB")
            if BUFFERED_CONTEXT_FORMAT.lower() == "png":
                ctx_img.save(out_dir / f"{stem}_buf.png")
            else:
                ctx_img.save(out_dir / f"{stem}_buf.jpg", quality=95)

        # (C) fixed-size context crop (same size for all objects)
        tb = tight_bbox(inst_mask)
        if tb is not None:
            ctx_bbox = fixed_context_box_around_object(
                tb, W, H, wr=CONTEXT_BOX_W_RATIO, hr=CONTEXT_BOX_H_RATIO
            )
            x0c, y0c, x1c, y1c = ctx_bbox
            ctx_fixed = Image.fromarray(np_img[y0c:y1c, x0c:x1c], mode="RGB")
            ctx_fixed.save(out_dir / f"{stem}_ctx.jpg", quality=95)

        kept += 1

    return kept

# ===================== Single-image demo (kept) =====================
def _run_single_image(device: str = DEVICE, task: str = TASK) -> str:
    model_dir, has_st = _validate_model()
    if not Path(IMAGE_PATH).is_file():
        _fail(f"IMAGE_PATH not found: {IMAGE_PATH}")

    print("[seg] Single-image mode")
    print(f"[seg] Image: {IMAGE_PATH}")
    print(f"[seg] Model dir: {model_dir}")
    print(f"[seg] Device: {device}  (safetensors={has_st})")

    processor, model = _load_local_model(model_dir, has_st, device)
    kept = _segment_one_image(IMAGE_PATH, OUT_DIR_DEMO, processor, model, task=task)
    msg = f"[seg] Objects saved: {kept} → {OUT_DIR_DEMO}"
    print(msg)
    return msg

# ===================== Batch directory walkers =====================
def _iter_images_under_creatives(images_base: Path) -> List[Path]:
    """
    Expect: images_base/<creative>/* (files)
    """
    if not images_base or not Path(images_base).is_dir():
        return []
    items: List[Path] = []
    for creative_dir in sorted(Path(images_base).iterdir()):
        if not creative_dir.is_dir():
            continue
        for f in sorted(creative_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                items.append(f)
    return items

def _iter_frames_medoid(videos_base: Path, frames_subdir: str = "frames_medoid") -> List[Path]:
    """
    Expect: videos_base/<creative>/frames_medoid/* (files)
    """
    if not videos_base or not Path(videos_base).is_dir():
        return []
    items: List[Path] = []
    for creative_dir in sorted(Path(videos_base).iterdir()):
        if not creative_dir.is_dir():
            continue
        fm = creative_dir / frames_subdir
        if not fm.is_dir():
            continue
        for f in sorted(fm.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                items.append(f)
    return items

def _outdir_for_image_source(img_path: Path) -> Path:
    """
    For images:  data/input/images/<creative>/m2f_objects/
    For frames:  data/input/videos/<creative>/m2f_objects/
    (frames live in .../<creative>/frames_medoid/*, we write results to the creative-level outdir)
    """
    parent = img_path.parent
    # If the image is inside frames_medoid, step up one to the creative folder
    if parent.name == "frames_medoid":
        creative_dir = parent.parent
    else:
        creative_dir = parent
    return creative_dir / "m2f_objects"

# ===================== Pipeline entry: process_all_input =====================
def process_all_input(
    report: Optional[Callable[[str], None]] = None,
    *,
    device: str = DEVICE,
    task: str = TASK,
    images_base=None,
    videos_base=None,
    frames_subdir: str = "frames_medoid",
) -> Iterable[str]:
    """
    Walks:
      - images_base/<creative>/*   (images)
      - videos_base/<creative>/frames_medoid/*  (frames)
    Segments each file, writing outputs into <creative>/m2f_objects/.
    """
    log = report or print

    # Validate & load model once
    model_dir, has_st = _validate_model()
    log(f"[seg] Model dir: {model_dir}")
    log(f"[seg] Device: {device}  (safetensors={has_st})")
    processor, model = _load_local_model(model_dir, has_st, device)

    # Collect sources
    images_base = Path(images_base) if images_base else None
    videos_base = Path(videos_base) if videos_base else None

    image_files  = _iter_images_under_creatives(images_base) if images_base else []
    frame_files  = _iter_frames_medoid(videos_base, frames_subdir=frames_subdir) if videos_base else []

    total_images = len(image_files)
    total_frames = len(frame_files)
    total = total_images + total_frames

    log(f"[seg] Found {total_images} image(s) and {total_frames} frame(s) to segment. Total={total}")
    yield f"[seg] Queue: images={total_images}, frames={total_frames}"

    # Process images first, then frames
    done = 0
    for i, p in enumerate(image_files, start=1):
        out_dir = _outdir_for_image_source(p)
        kept = _segment_one_image(p, out_dir, processor, model, task=task)
        done += 1
        msg = f"[seg] [{done}/{total}] image {p.name} → {kept} obj(s)"
        log(msg); yield msg

    for i, p in enumerate(frame_files, start=1):
        out_dir = _outdir_for_image_source(p)
        kept = _segment_one_image(p, out_dir, processor, model, task=task)
        done += 1
        msg = f"[seg] [{done}/{total}] frame {p.name} → {kept} obj(s)"
        log(msg); yield msg

    summary = f"[seg] Completed. Processed {done}/{total} file(s)."
    log(summary); yield summary

# ===================== CLI (optional test) =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Mask2Former over data/input/{images, videos/*/frames_medoid} or single-image demo.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE)
    parser.add_argument("--task", choices=["instance", "panoptic"], default=TASK)
    parser.add_argument("--images-base", type=str, default=str(PROJECT_ROOT / "data" / "input" / "images"))
    parser.add_argument("--videos-base", type=str, default=str(PROJECT_ROOT / "data" / "input" / "videos"))
    parser.add_argument("--frames-subdir", type=str, default="frames_medoid",
                        help="Subfolder under each video creative containing frames to segment")
    parser.add_argument("--single-image", action="store_true", help="Run the single-image demo instead of batch")

    args = parser.parse_args()

    if args.single_image:
        _run_single_image(device=args.device, task=args.task)
    else:
        for line in process_all_input(
            report=None,
            device=args.device,
            task=args.task,
            images_base=Path(args.images_base),
            videos_base=Path(args.videos_base),
            frames_subdir=args.frames_subdir,
        ):
            pass
