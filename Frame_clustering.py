from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# ---------------------------------------------
# Minimal frame clustering that still uses PCA
# - No FPS downsampling
# - No embeddings
# - PCA over grayscale frames (centered by PCA)
# - Simple DBSCAN surface (eps, min_samples only)
# - Yields progress strings (compatible with your pipeline)
# ---------------------------------------------

# Small constants (not exposed as CLI args)
RESIZE_HW: Tuple[int, int] = (160, 160)  # keep lightweight
PCA_COMPONENTS: int = 20                 # clipped to valid range at runtime
DBSCAN_EPS: float = 0.02
DBSCAN_MIN_SAMPLES: int = 5
JPEG_QUALITY: int = 95


def _emit(msg: str, report=None):
    if report:
        report(msg)
    else:
        print(msg)


def _read_all_frames_gray(video_path: str, resize_hw: Tuple[int, int]) -> Tuple[List[int], List[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idxs: List[int] = []
    frames_gray: List[np.ndarray] = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize_hw:
            gray = cv2.resize(gray, resize_hw, interpolation=cv2.INTER_AREA)
        frames_gray.append(gray)
        idxs.append(i)
        i += 1

    cap.release()

    if not frames_gray:
        raise RuntimeError("No frames read from video.")

    return idxs, frames_gray


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def _medoid_indices_cosine(Z: np.ndarray, labels: np.ndarray) -> List[int]:
    """Return medoid indices (0..N-1) per cluster using cosine distance on L2-normalized Z.
    labels = DBSCAN labels (-1 is noise)."""
    medoids: List[int] = []
    cluster_ids = sorted({c for c in labels if c != -1})
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        sub = Z[idx]  # (m, d) L2-normalized
        # cosine distance matrix: D = 1 - sub @ sub.T
        D = 1.0 - (sub @ sub.T)
        s = D.sum(axis=1)
        medoids.append(idx[int(np.argmin(s))])
    return medoids


def _cluster_with_pca(frames_gray: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z, labels):
    Z  : L2-normalized PCA embeddings (N, k)
    labels: DBSCAN labels with -1 as noise
    """
    X = np.stack(frames_gray)  # (N,H,W)
    Xf = X.reshape(len(X), -1).astype(np.float32)  # (N,H*W)

    # PCA: sklearn centers the data; clip components to valid range
    n_samples, n_feats = Xf.shape
    k = max(1, min(PCA_COMPONENTS, n_samples - 1, n_feats))
    pca = PCA(n_components=k, random_state=0)
    Zp = pca.fit_transform(Xf)  # (N,k)

    Z = _l2_normalize_rows(Zp)

    # Simple DBSCAN on cosine distance
    clu = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    labels = clu.fit_predict(Z)
    return Z, labels



def _save_reps(video_path: str, out_dir: Path, rep_global_indices: List[int]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(rep_global_indices)
    saved: List[str] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot re-open video: {video_path}")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            outp = out_dir / f"rep_{i:06d}.jpg"
            cv2.imwrite(str(outp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            saved.append(str(outp))
            if len(saved) == len(want):
                break
        i += 1

    cap.release()
    return saved


def _cluster_single_video(video: Path, out_dir: Path, report=None) -> dict:
    idxs, frames_gray = _read_all_frames_gray(str(video), RESIZE_HW)
    Z, labels = _cluster_with_pca(frames_gray)
    medoids_local = _medoid_indices_cosine(Z, labels)
    rep_global = [idxs[i] for i in medoids_local]
    rep_paths = _save_reps(str(video), out_dir, rep_global)

    n_clusters = len({c for c in labels if c != -1})
    noise = int(np.sum(labels == -1))

    summary = {
        "video": video.name,
        "total_frames": len(idxs),
        "n_clusters": n_clusters,
        "noise_frames": noise,
        "rep_frame_indices": rep_global,
        "rep_paths": rep_paths,
        "pca_components": int(Z.shape[1]),
        "eps": DBSCAN_EPS,
        "min_samples": DBSCAN_MIN_SAMPLES,
    }
    _emit(json.dumps(summary, indent=2), report)
    return summary


def find_video_in_folder(folder: Path) -> Optional[Path]:
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def process_all_video_folders(
    base_dir: Optional[Path] = None,
    out_subdir: str = "frames_min",
    report=None,
) -> Iterable[str]:
    """Compatible with your pipeline import and call site.

    - Yields progress/status lines
    - Uses internal defaults (no hyperparameters forwarded)
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / "data" / "input" / "videos"

    base_dir = Path(base_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        msg = f"Videos folder not found: {base_dir}"
        _emit(msg, report); yield msg
        return

    folders = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
    start = f"Scanning {base_dir} → {len(folders)} folder(s) found"
    _emit(start, report); yield start

    if not folders:
        done = "No video folders present."
        _emit(done, report); yield done
        return

    for i, folder in enumerate(folders, start=1):
        hdr = f"[{i}/{len(folders)}] {folder.name}"
        _emit(hdr, report); yield hdr

        video = find_video_in_folder(folder)
        if not video:
            msg = "  ! no video found in this folder; skipping"
            _emit(msg, report); yield msg
            continue

        out_dir = folder / out_subdir
        info = f"  → clustering frames for: {video.name}  (out: {out_dir.name})"
        _emit(info, report); yield info

        try:
            summary = _cluster_single_video(video, out_dir, report)
            msg_ok = (
                f"  ✓ saved {len(summary['rep_paths'])} representative frame(s)  "
                f"| pca_k={summary['pca_components']} "
                f"| eps={summary['eps']} min_samples={summary['min_samples']}"
            )
            _emit(msg_ok, report); yield msg_ok
        except Exception as e:
            msg_err = f"  ! clustering failed: {e}"
            _emit(msg_err, report); yield msg_err

    final = "All video folders processed."
    _emit(final, report); yield final
