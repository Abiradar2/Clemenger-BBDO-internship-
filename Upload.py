# Upload.py
from __future__ import annotations

# ====================== ENV / PERFORMANCE CAPS (run first) ======================
import os

# Let env override; falls back to ~half your cores (fast on i9 CPU-only)
def _auto_threads(default_min=4):
    n = os.cpu_count() or 8
    return int(os.environ.get("PIPE_THREADS", max(default_min, n // 2)))


CPU_THREADS = _auto_threads()   # e.g. 4..6 on laptop
INTEROP_THREADS = 1

# Hard overrides so this entrypoint wins
os.environ["OMP_NUM_THREADS"]        = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_THREADS)   # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"]    = str(CPU_THREADS)

# Keep MKL/OMP steady and avoid busy-wait between ops (mirrors your fast notebook)
os.environ["MKL_DYNAMIC"]   = "FALSE"
os.environ["OMP_PROC_BIND"] = "TRUE"
os.environ["OMP_PLACES"]    = "cores"
os.environ["KMP_BLOCKTIME"] = "0"     # reduce spin overhead on OpenMP

# Tokenizers / Transformers misc (safe & reduces overhead)
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]    = "1"

# Inform PyTorch too (works even if torch is not installed yet)
try:
    import torch
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
except Exception:
    pass
# ===============================================================================

from pathlib import Path
from typing import Callable, Iterable, Optional, List
import shutil, sys
import importlib.util, subprocess, runpy
import multiprocessing as mp

# -------- multiprocessing safe start method --------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# -------- constants --------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
ALLOWED    = IMAGE_EXTS | VIDEO_EXTS

# -------- helpers --------
def _project_root() -> Path:
    # Directory containing app.py and this Upload.py
    return Path(__file__).resolve().parent

def _input_dirs() -> tuple[Path, Path, Path]:
    """Return (INPUT_DIR, IMAGES_DIR, VIDEOS_DIR) under data/input."""
    root = _project_root()
    input_dir  = root / "data" / "input"
    images_dir = input_dir / "images"
    videos_dir = input_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, images_dir, videos_dir

def _purge_torchvision_stubs():
    for k in list(sys.modules):
        if k == "torchvision" or k.startswith("torchvision."):
            del sys.modules[k]

def _unique_name(parent: Path, base: str) -> Path:
    """
    Ensure a unique subdirectory name under `parent` using `base`.
    If parent/base exists, append _(<n>).
    Returns the path to the unique subdirectory.
    """
    candidate = parent / base
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        alt = parent / f"{base}_({i})"
        if not alt.exists():
            return alt
        i += 1

def _emit(msg: str, report: Optional[Callable[[str], None]]):
    if report is not None:
        report(msg)
    else:
        print(msg)

# -------- DB/FAISS APPEND helper (can be called standalone) --------
def append_to_db_only() -> None:
    """
    Minimal entrypoint: append creatives into SQLite/FAISS using your
    load_simple_jsons.load_all() function. Assumes prior steps have already
    produced the JSON artifacts in data/input/** (OpenCLIP, QWEN, etc.).
    """
    print("Appending creatives to SQLite/FAISS (load_simple_jsons.load_all) ...")
    try:
        # ✅ Correct ingestion module/function
        from load_simple_jsons import load_all as load_sql_faiss
    except Exception as e:
        print(f"DB ingest step skipped (load_simple_jsons import failed): {e}")
        return

    try:
        load_sql_faiss()     # prints its own totals
        print("SQLite/FAISS append complete.")
    except Exception as e:
        print(f"DB ingest failed while running load_simple_jsons.load_all(): {e}")

# -------- main API --------
def list_files_once(
    path_str: str,
    report: Optional[Callable[[str], None]] = None,
    stages: Optional[List[str]] = None,  # kept for compatibility; unused
    recursive: bool = False,
    *,
    cluster_videos: bool = True,
    out_subdir: str = "frames_medoid",
) -> Iterable[str]:
    """
    Copy media from `path_str` into:
      data/input/images/<file_stem>/original_name.ext
      data/input/videos/<file_stem>/original_name.ext
    Then (optionally) run frame clustering over data/input/videos/*/
    Finally, run Mask2Former segmentation over BOTH images and clustered frames.
    After segmentation, run QWEN to tag/summarize creatives.
    Then run OpenCLIP embeddings + global index.
    Finally, append results into SQLite/FAISS (load_simple_jsons.load_all()).
    """
    # show interpreter + thread policy to catch venv/system mixups
    interp = f"[upload] Python: {sys.executable}"
    _emit(interp, report); yield interp
    _emit(f"[upload] Threads={CPU_THREADS} interop={INTEROP_THREADS}", report); yield f"[upload] Threads={CPU_THREADS} interop={INTEROP_THREADS}"

    # -----------------------------
    # COPY MEDIA INTO data/input
    # -----------------------------
    src = Path(path_str).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        msg = f"Not a directory: {src}"
        _emit(msg, report); yield msg
        return

    input_dir, images_dir, videos_dir = _input_dirs()

    iterator = src.rglob("*") if recursive else src.iterdir()
    files = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in ALLOWED]
    total = len(files)

    start_msg = f"Found {total} media file(s) in {src}"
    _emit(start_msg, report); yield start_msg

    if total == 0:
        done_msg = "No eligible files to copy."
        _emit(done_msg, report); yield done_msg
        return

    copied = 0
    for idx, fp in enumerate(files, start=1):
        ext = fp.suffix.lower()
        is_image = ext in IMAGE_EXTS
        parent_dir = images_dir if is_image else videos_dir

        per_file_dir = _unique_name(parent_dir, fp.stem)
        per_file_dir.mkdir(parents=True, exist_ok=True)

        dst = per_file_dir / fp.name

        msg1 = f"[{idx}/{total}] {'image' if is_image else 'video'}: {fp.name}"
        _emit(msg1, report); yield msg1

        try:
            shutil.copy2(fp, dst)
            copied += 1
            rel = dst.relative_to(_project_root())
            msg2 = f"  → copied to {rel}"
            _emit(msg2, report); yield msg2
        except Exception as e:
            msg_err = f"  ! copy failed: {e}"
            _emit(msg_err, report); yield msg_err

    summary = f"Done. Copied {copied}/{total} file(s) into {input_dir}."
    _emit(summary, report); yield summary

    # -----------------------------
    # FRAME CLUSTERING
    # -----------------------------
    if cluster_videos:
        try:
            from Frame_clustering import process_all_video_folders
        except Exception as e:
            err = f"Clustering step skipped (Frame_clustering import failed): {e}"
            _emit(err, report); yield err
            return

        kick = f"Starting frame clustering over {videos_dir} ..."
        _emit(kick, report); yield kick

        for line in process_all_video_folders(
            base_dir=videos_dir,
            out_subdir=out_subdir,
            report=report,
        ):
            yield line

        done = "Clustering pass completed."
        _emit(done, report); yield done

    # -----------------------------
    # SEGMENTATION (Mask2Former)
    # -----------------------------
    try:
        # import here so Upload.py has no hard dependency if you just want copying
        from Image_segmentation import process_all_input as run_segmentation
    except Exception as e:
        err = f"Segmentation step skipped (Image_segmentation import failed): {e}"
        _emit(err, report); yield err
        # keep flow going to QWEN / OpenCLIP even if segmentation missing
    else:
        kick2 = f"Starting segmentation (images_base={images_dir}, videos_base={videos_dir}) ..."
        _emit(kick2, report); yield kick2

        # Your process_all_input should walk folders internally
        for line in run_segmentation(
            report=report,
            device="cpu",            # change to "cuda" if you enable GPU
            task="instance",         # or "panoptic"
            images_base=images_dir,  # provided for future use
            videos_base=videos_dir,  # provided for future use
        ):
            _emit(line, report); yield line

        done2 = "Segmentation pass completed."
        _emit(done2, report); yield done2

    # -----------------------------
    # QWEN — tagging/summary
    # -----------------------------
    try:
        from QWEN import process_all_input as run_qwen
    except Exception as e:
        err = f"QWEN step skipped (QWEN import failed): {e}"
        _emit(err, report); yield err
    else:
        kick = "Starting QWEN tagging/summary ..."
        _emit(kick, report); yield kick
        for line in run_qwen(report=report):  # QWEN uses CWD/data/input by default
            _emit(line, report); yield line

    # -----------------------------
    # OpenCLIP — embed creatives + build global index
    # -----------------------------
    _purge_torchvision_stubs()
    openclip_ok = False
    try:
        # Preferred clean importable name:
        from SQL_Open_clip_ingestion_add_global import process_all_input as run_clip
    except Exception as e:
        # Fallback to a filename with '+' or different spelling via runpy
        try:
            clip_path = _project_root() / "SQL_Open_clip_injestion+add_global.py"
            if clip_path.exists():
                _emit("Starting OpenCLIP (runpy path mode) ...", report); yield "Starting OpenCLIP (runpy path mode) ..."
                runpy.run_path(str(clip_path), run_name="__main__")
                openclip_ok = True
            else:
                raise FileNotFoundError(f"{clip_path} not found")
        except Exception as e2:
            err = f"OpenCLIP step skipped (import/run failed): {e} | {e2}"
            _emit(err, report); yield err
    else:
        kick = "Starting OpenCLIP embedding/indexing ..."
        _emit(kick, report); yield kick
        for line in run_clip(
            report=report,
            device="cpu",      # switch to "cuda" if available
            batch_size=16,
            top_k=5,
            model_name="ViT-B-32",
            pretrained="openai",
            print_vectors=False,
        ):
            _emit(line, report); yield line
        done = "OpenCLIP embedding pass completed."
        _emit(done, report); yield done
        openclip_ok = True

    # -----------------------------
    # SQL/FAISS INGESTION (append to DB after OpenCLIP)
    # -----------------------------
    if openclip_ok:
        try:
            # ✅ Correct module and function to append to SQLite/FAISS
            from load_simple_jsons import load_all as load_sql_faiss
        except Exception as e:
            err = f"DB ingest step skipped (load_simple_jsons import failed): {e}"
            _emit(err, report); yield err
        else:
            kick = "Appending creatives to SQLite/FAISS (load_simple_jsons.load_all) ..."
            _emit(kick, report); yield kick
            # load_all prints its own totals; just call it
            try:
                load_sql_faiss()
                done = "SQLite/FAISS append complete."
                _emit(done, report); yield done
            except Exception as e:
                err = f"DB ingest failed while running load_simple_jsons.load_all(): {e}"
                _emit(err, report); yield err
    else:
        warn = "Skipping DB append because OpenCLIP did not complete."
        _emit(warn, report); yield warn


# -----------------------------
# Optional CLI usage
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Copy media → (optional) cluster frames → segment → QWEN → OpenCLIP → DB append."
    )
    parser.add_argument("path", help="Source directory containing images/videos")
    parser.add_argument("--no-cluster", action="store_true", help="Skip the clustering step")
    parser.add_argument("--recursive", action="store_true", help="Traverse subdirectories when copying")
    parser.add_argument("--out-subdir", default="frames_medoid",
                        help="Medoid output subfolder inside each video folder")

    args = parser.parse_args()

    for _line in list_files_once(
        args.path,
        report=None,      # prints to terminal by default
        recursive=args.recursive,
        cluster_videos=not args.no_cluster,
        out_subdir=args.out_subdir,
    ):
        pass
