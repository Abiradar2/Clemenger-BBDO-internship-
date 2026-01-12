# qwen_bench_fp32_only_multi_prompts_pipeline_savejson_medoid_desc.py
# Fast CPU-only Qwen2-VL runner:
# - DESC only for medoid frames (VID-FRAMES)
# - Non-medoids: TAGS + EMOTION (desc = "")
# - Creative-level summaries:
#     * MEDOID summary: {"industry": "..."} & {"product": "..."} + brief video-level DESC
#     * IMAGE summary over IMG-ORIG ∪ IMG-M2F: industry/product + overall DESC into "__ALL_IMAGES__"

from __future__ import annotations

# ====================== ENV / PERFORMANCE CAPS (run first) ======================
import os, sys, types, json, time
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Iterable, Callable
import re, random, math

def _auto_threads(default_min=4):
    n = os.cpu_count() or 8
    preset = int(os.environ.get("PIPE_THREADS", max(default_min, n // 2)))
    return max(1, preset)

CPU_THREADS     = _auto_threads()  # e.g., 4..(n//2)
INTEROP_THREADS = 1

# Make sure these land BEFORE torch/np/transformers import
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"

# BLAS/OMP threading caps (hard overrides so this file wins)
os.environ["OMP_NUM_THREADS"]        = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = str(CPU_THREADS)
os.environ["MKL_DYNAMIC"]            = "FALSE"
os.environ["OMP_PROC_BIND"]          = "TRUE"
os.environ["OMP_PLACES"]             = "cores"
os.environ["KMP_BLOCKTIME"]          = "0"     # yield quickly between ops

# ==== FIRST-RUN GUARD: keep Transformers from importing torchvision/onnx ====
for k in list(sys.modules):
    if k == "torchvision" or k.startswith("torchvision."):
        del sys.modules[k]

tv = types.ModuleType("torchvision"); tv.__spec__ = ModuleSpec("torchvision", None, is_package=True); tv.__path__ = []
tvt = types.ModuleType("torchvision.transforms"); tvt.__spec__ = ModuleSpec("torchvision.transforms", None, is_package=True); tvt.__path__ = []
class InterpolationMode: NEAREST=0; BOX=1; BILINEAR=2; BICUBIC=3; LANCZOS=4; HAMMING=5
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
tvf = types.ModuleType("torchvision.transforms.functional"); tvf.__spec__ = ModuleSpec("torchvision.transforms.functional", None, is_package=False)
def _noop(*a, **k): return a[0] if a else None
for _n in ["resize","center_crop","normalize","pad","to_pil_image","pil_to_tensor","convert_image_dtype"]:
    setattr(tvf, _n, _noop)
sys.modules["torchvision"] = tv
sys.modules["torchvision.transforms"] = tvt
sys.modules["torchvision.transforms.functional"] = tvf
tvt.functional = tvf
# ===========================================================================

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig
try:
    from transformers import AutoModelForImageTextToText as ModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as ModelCls

# ===== CONFIG =====
MODEL_ID         = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE           = "cpu"
DTYPE            = torch.float32
TARGET_MAX       = 300
USE_CACHE        = True
LIST_EXT         = (".png", ".jpg", ".jpeg", ".webp")
RANDOM_SEED      = 42

VERBOSE          = True
PRINT_EVERY      = 1
BATCH_SIZE       = 7

# ---- medoid/image summary cap ----
MAX_FRAMES_FOR_MEDOID_SUMMARY = 12

# Torch thread caps
try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
except RuntimeError:
    pass

# ===== PROMPTS =====
PROMPT_TAGS = (
    "Return 5 short, comma-separated tags that describe the image "
    "(objects, people, setting, or emotions). No sentences. No extra words. "
    "Example format: 'dog, park, running, child, happy'."
)
PROMPT_DESC = (
    "Write a concise description of what is happening in the image as a single paragraph. "
    "Be concrete and visual (who/what/where and any notable text or logos). "
    "Target length: 100–200 tokens."
)
PROMPT_EMOTION = (
    "List the dominant emotional tone conveyed by the image in 1–2 words only "
    "(e.g., joyful, trusting, nostalgic, energetic, calm). Output only the words."
)

# ---- medoid/image summary prompts (industry + product), JSON one-liners ----
PROMPT_MEDOID_INDUSTRY = (
    "These images are frames from the SAME advertisement. Using ALL frames together, answer ONLY this for the WHOLE ad:\n"
    "• What is the most likely industry? (e.g., supermarket, pharmacy, electronics retail, restaurant, fashion retail, automotive, banking, telecom, travel, alcohol, FMCG, etc.)\n\n"
    "Return ONLY this JSON object on a single line: {\"industry\":\"...\"}\n"
    "Constraints: value <= 5 words. If uncertain, use \"unknown\"."
)
PROMPT_MEDOID_PRODUCT = (
    "These images are frames from the SAME advertisement. Using ALL frames together, answer ONLY this for the WHOLE ad:\n"
    "• What specific product, brand, or service appears to be advertised? (succinct name)\n\n"
    "Return ONLY this JSON object on a single line: {\"product\":\"...\"}\n"
    "Constraints: value <= 8 words. If uncertain, use \"unknown\"."
)

# ---- overall image-creative description (for __ALL_IMAGES__) ----
PROMPT_IMAGESET_DESC = (
    "You are given multiple still images from the SAME creative. "
    "Write a single concise paragraph (100–200 tokens) that best describes the overall creative: "
    "who/what is featured, setting/context, any visible text/logos/branding, and the core message. "
    "Avoid repetition and avoid guessing beyond what is visible."
)

# ---- NEW: overall video description (for the whole video) ----
PROMPT_VIDEO_DESC = (
    "You are given multiple key frames from the SAME video advertisement. "
    "Write a single brief paragraph (60–150 tokens) that describes the overall video: "
    "who or what appears, where it takes place, the main actions or transitions, "
    "and the core message or call-to-action. "
    "Do not guess beyond what is visible."
)

# Token budgets
TOKENS_TAGS, TOKENS_DESC, TOKENS_EMOTION = 12, 220, 12
TOKENS_MEDOID_INDUSTRY, TOKENS_MEDOID_PRODUCT = 48, 64
TOKENS_IMAGESET_DESC = 220
TOKENS_VIDEO_DESC    = 180  # new: video-level brief description

# Task presets
TASK_TAGS      = {"key": "tags",     "prompt": PROMPT_TAGS,         "max_new_tokens": TOKENS_TAGS,         "last_line_only": True}
TASK_DESC      = {"key": "desc",     "prompt": PROMPT_DESC,         "max_new_tokens": TOKENS_DESC,         "last_line_only": False}
TASK_EMOTION   = {"key": "emotion",  "prompt": PROMPT_EMOTION,      "max_new_tokens": TOKENS_EMOTION,      "last_line_only": True}

# Apply DESC only for medoids (VID-FRAMES)
TASKS_FOR_FRAMES   = [TASK_TAGS, TASK_DESC, TASK_EMOTION]
TASKS_FOR_NONFRAME = [TASK_TAGS,            TASK_EMOTION]

# medoid/image-only task configs
TASK_MEDOID_INDUSTRY = {"key": "industry", "prompt": PROMPT_MEDOID_INDUSTRY, "max_new_tokens": TOKENS_MEDOID_INDUSTRY, "last_line_only": True}
TASK_MEDOID_PRODUCT  = {"key": "product",  "prompt": PROMPT_MEDOID_PRODUCT,  "max_new_tokens": TOKENS_MEDOID_PRODUCT,  "last_line_only": True}
TASK_IMAGESET_DESC   = {"key": "desc",     "prompt": PROMPT_IMAGESET_DESC,   "max_new_tokens": TOKENS_IMAGESET_DESC,   "last_line_only": False}
TASK_VIDEO_DESC      = {"key": "desc",     "prompt": PROMPT_VIDEO_DESC,      "max_new_tokens": TOKENS_VIDEO_DESC,      "last_line_only": False}

# ===== ENV / RNG =====
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
random.seed(RANDOM_SEED)

# ===== PROCESSOR (load once) =====
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ===== UTILS =====
def load_and_downscale(path, target_max=TARGET_MAX):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > target_max:
        s = target_max / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
    return img

def build_inputs_batch(proc, imgs, prompt: str):
    messages_batch = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        for _ in imgs
    ]
    text_inputs = proc.apply_chat_template(messages_batch, add_generation_prompt=True, tokenize=False)
    enc = proc(text=text_inputs, images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(DEVICE) for k, v in enc.items()}, messages_batch

def build_inputs_multi(proc, imgs, prompt: str):
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text_inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = proc(text=text_inputs, images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(DEVICE) for k, v in enc.items()}, messages

def sanitize_generation_config(model, tokenizer=None):
    gc = GenerationConfig.from_model_config(model.config)
    gc.do_sample = False
    for attr in ("temperature","top_p","top_k","typical_p","penalty_alpha"):
        if hasattr(gc, attr): setattr(gc, attr, None)
    if tokenizer is not None:
        if getattr(gc, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
            gc.eos_token_id = tokenizer.eos_token_id
        if getattr(gc, "pad_token_id", None) is None:
            gc.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.generation_config = gc

def _last_nonempty_line(s: str) -> str:
    if not s: return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _decode_new_tokens(tokenizer, sequences, prompt_lens, last_line_only: bool, prompt_text: str):
    outs = []
    for row_ids, pl in zip(sequences, prompt_lens):
        new_ids = row_ids[pl:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip() if tokenizer else ""
        if prompt_text and prompt_text in text:
            text = text.replace(prompt_text, "").strip()
        outs.append(_last_nonempty_line(text) if last_line_only else text)
    return outs

def run_task_batch(model, proc, imgs, task) -> Tuple[float, List[str]]:
    inputs, _ = build_inputs_batch(proc, imgs, task["prompt"])
    with torch.inference_mode():
        if "attention_mask" in inputs:
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        else:
            pad_id = getattr(proc.tokenizer, "pad_token_id", 0)
            prompt_lens = inputs["input_ids"].ne(pad_id).sum(dim=1).tolist()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=task["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=USE_CACHE,
            return_dict_in_generate=True,
        )
        dt = time.perf_counter() - t0
    decoded = _decode_new_tokens(
        getattr(proc, "tokenizer", None),
        out.sequences,
        prompt_lens,
        task["last_line_only"],
        task["prompt"],
    )
    return dt, decoded

def run_task_multi(model, proc, imgs, task) -> Tuple[float, str]:
    inputs, _ = build_inputs_multi(proc, imgs, task["prompt"])
    with torch.inference_mode():
        if "attention_mask" in inputs:
            prompt_len = inputs["attention_mask"].sum(dim=1).tolist()[0]
        else:
            pad_id = getattr(proc.tokenizer, "pad_token_id", 0)
            prompt_len = inputs["input_ids"].ne(pad_id).sum(dim=1).tolist()[0]
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=task["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=USE_CACHE,
            return_dict_in_generate=True,
        )
        dt = time.perf_counter() - t0
    text = _decode_new_tokens(
        getattr(proc, "tokenizer", None),
        out.sequences,
        [prompt_len],
        task["last_line_only"],
        task["prompt"],
    )[0]
    return dt, text

def _subsample_uniform(paths: List[str], k: int) -> List[str]:
    if len(paths) <= k: return paths
    step = math.ceil(len(paths) / k)
    return paths[::step][:k]

def _parse_json_value(s: str, key: str) -> str:
    try:
        obj = json.loads(s.strip())
        val = str(obj.get(key, "")).strip()
        if val:
            return val
    except Exception:
        pass
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', s)
    return m.group(1).strip() if m else s.strip()

def list_image_files(dir_path: Path) -> List[str]:
    if not dir_path.is_dir(): return []
    out = []
    for ext in LIST_EXT:
        out.extend(str(p) for p in dir_path.glob(f"*{ext}"))
    return sorted(out)

def collect_grouped_by_creative(base: Path) -> Dict[Path, Dict[str, List[str]]]:
    grouped: Dict[Path, Dict[str, List[str]]] = {}
    def _add(creative: Path, key: str, files: List[str]):
        if not files: return
        bucket = grouped.setdefault(
            creative,
            {"IMG-ORIG": [], "IMG-M2F": [], "VID-M2F": [], "VID-FRAMES": []}
        )
        bucket[key].extend(files)

    images_root = base / "images"
    videos_root = base / "videos"

    if images_root.is_dir():
        for creative in sorted(p for p in images_root.iterdir() if p.is_dir()):
            _add(creative, "IMG-ORIG", list_image_files(creative))
            _add(creative, "IMG-M2F", list_image_files(creative / "m2f_objects"))

    if videos_root.is_dir():
        for creative in sorted(p for p in videos_root.iterdir() if p.is_dir()):
            _add(creative, "VID-M2F", list_image_files(creative / "m2f_objects"))
            fm = creative / "frames_medoids"
            if not fm.is_dir(): fm = creative / "frames_medoid"
            _add(creative, "VID-FRAMES", list_image_files(fm))

    return grouped

def _relative_to_creative(creative: Path, abs_path_str: str) -> str:
    try:
        return str(Path(abs_path_str).relative_to(creative))
    except Exception:
        return Path(abs_path_str).name

# ===== MODEL BUILD =====
def _load_base_model():
    model = ModelCls.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(DEVICE).eval()
    sanitize_generation_config(model, getattr(processor, "tokenizer", None))
    return model

def build_fp32():
    t0 = time.perf_counter()
    model = _load_base_model()
    build = time.perf_counter() - t0
    return model, processor, build

# ===== PROCESS & SAVE PER CREATIVE =====
def process_and_save_for_creative(creative: Path, paths_by_set: Dict[str, List[str]], model, proc):
    results: List[Dict[str, str]] = []

    # Per-source images
    for source_key in ("IMG-ORIG", "IMG-M2F", "VID-M2F", "VID-FRAMES"):
        files = paths_by_set.get(source_key, [])
        if not files:
            continue

        tasks = TASKS_FOR_FRAMES if source_key == "VID-FRAMES" else TASKS_FOR_NONFRAME

        print(f"\n--- {creative.name} :: {source_key}: {len(files)} image(s) ---")
        n = len(files)
        for i0 in range(0, n, BATCH_SIZE):
            chunk_paths = files[i0:i0 + BATCH_SIZE]
            imgs = [load_and_downscale(p, TARGET_MAX) for p in chunk_paths]
            names_rel = [_relative_to_creative(creative, p) for p in chunk_paths]

            try:
                task_outputs: Dict[str, List[str]] = {}
                for task in tasks:
                    _, decoded = run_task_batch(model, proc, imgs, task)
                    task_outputs[task["key"]] = decoded

                if "desc" not in task_outputs:
                    task_outputs["desc"] = [""] * len(names_rel)

                for idx, name_rel in enumerate(names_rel):
                    results.append({
                        "file": name_rel,
                        "source_set": source_key,
                        "tags": task_outputs["tags"][idx],
                        "desc": task_outputs["desc"][idx],
                        "emotion": task_outputs["emotion"][idx],
                        "industry": "",
                        "product": "",
                    })

                if VERBOSE:
                    for idx, name_rel in enumerate(names_rel):
                        print(f"[{creative.name}/{source_key}] {name_rel}")
                        print(f"  TAGS:      {task_outputs['tags'][idx]}")
                        if source_key == "VID-FRAMES":
                            print(f"  DESC:      {task_outputs['desc'][idx]}")
                        print(f"  EMOTION:   {task_outputs['emotion'][idx]}")
                else:
                    done = min(i0 + len(imgs), n)
                    print(f"[{creative.name}/{source_key}] {done:4d}/{n} processed")

            except Exception as e:
                for name_rel in names_rel:
                    print(f"[{creative.name}/{source_key}] {name_rel} ERROR: {e}")

    # MEDOID summary (industry + product + brief video-level DESC)
    medoid_paths = paths_by_set.get("VID-FRAMES", [])
    if medoid_paths:
        print(f"\n--- {creative.name} :: MEDOID-SUMMARY over {len(medoid_paths)} frame(s) ---")
        medoid_eff = _subsample_uniform(medoid_paths, MAX_FRAMES_FOR_MEDOID_SUMMARY)
        imgs_all = [load_and_downscale(p, TARGET_MAX) for p in medoid_eff]

        industry_txt = "unknown"
        product_txt  = "unknown"
        video_desc   = ""

        try:
            _, out_ind = run_task_multi(model, proc, imgs_all, TASK_MEDOID_INDUSTRY)
            industry_txt = _parse_json_value(out_ind, "industry")
        except Exception as e:
            print(f"[{creative.name}/MEDOID] industry ERROR: {e}")

        try:
            _, out_prod = run_task_multi(model, proc, imgs_all, TASK_MEDOID_PRODUCT)
            product_txt = _parse_json_value(out_prod, "product")
        except Exception as e:
            print(f"[{creative.name}/MEDOID] product ERROR: {e}")

        # NEW: whole-video brief description
        try:
            _, video_desc = run_task_multi(model, proc, imgs_all, TASK_VIDEO_DESC)
        except Exception as e:
            print(f"[{creative.name}/MEDOID] video-desc ERROR: {e}")
            video_desc = ""

        results.append({
            "file": "__ALL_MEDOIDS__",
            "source_set": "VID-FRAMES",
            "tags": "",
            "desc": video_desc,   # <--- brief video-level description stored here
            "emotion": "",
            "industry": industry_txt,
            "product": product_txt,
        })

        if VERBOSE:
            print(f"[{creative.name}/MEDOID] INDUSTRY: {industry_txt}")
            print(f"[{creative.name}/MEDOID] PRODUCT:  {product_txt}")
            print(f"[{creative.name}/MEDOID] DESC(len={len(video_desc)}): {video_desc[:120]}{'...' if len(video_desc) > 120 else ''}")

    # IMAGE summary over IMG-ORIG ∪ IMG-M2F (industry, product, overall desc)
    img_paths = [*paths_by_set.get("IMG-ORIG", []), *paths_by_set.get("IMG-M2F", [])]
    if img_paths:
        print(f"\n--- {creative.name} :: IMAGE-SUMMARY over {len(img_paths)} image(s) ---")
        img_eff = _subsample_uniform(img_paths, MAX_FRAMES_FOR_MEDOID_SUMMARY)
        imgs_all = [load_and_downscale(p, TARGET_MAX) for p in img_eff]

        industry_txt = "unknown"
        product_txt  = "unknown"
        image_desc   = ""

        try:
            _, out_ind = run_task_multi(model, processor, imgs_all, TASK_MEDOID_INDUSTRY)
            industry_txt = _parse_json_value(out_ind, "industry")
        except Exception as e:
            print(f"[{creative.name}/IMAGE] industry ERROR: {e}")

        try:
            _, out_prod = run_task_multi(model, processor, imgs_all, TASK_MEDOID_PRODUCT)
            product_txt = _parse_json_value(out_prod, "product")
        except Exception as e:
            print(f"[{creative.name}/IMAGE] product ERROR: {e}")

        try:
            _, image_desc = run_task_multi(model, processor, imgs_all, TASK_IMAGESET_DESC)
        except Exception as e:
            print(f"[{creative.name}/IMAGE] desc ERROR: {e}")

        results.append({
            "file": "__ALL_IMAGES__",
            "source_set": "IMG-ALL",
            "tags": "",
            "desc": image_desc,
            "emotion": "",
            "industry": industry_txt,
            "product": product_txt,
        })

        if VERBOSE:
            print(f"[{creative.name}/IMAGE] INDUSTRY: {industry_txt}")
            print(f"[{creative.name}/IMAGE] PRODUCT:  {product_txt}")
            print(f"[{creative.name}/IMAGE] DESC(len={len(image_desc)}): {image_desc[:120]}{'...' if len(image_desc) > 120 else ''}")

    # SAVE
    out_path = creative / "qwen_multi_prompts.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[save] Wrote {len(results)} record(s) → {out_path}")
    except Exception as e:
        print(f"[save] Failed to write JSON for {creative}: {e}")

# === PUBLIC ENTRYPOINT (generator) ===========================================
def process_all_input(
    report: Optional[Callable[[str], None]] = None,
    base_dir: Optional[Union[Path, str]] = None,
) -> Iterable[str]:
    """
    Run the full QWEN tagging/summary pipeline and yield status lines.
    - report: optional callable(str) for UI logging (e.g., Streamlit).
    - base_dir: override for data/input root; if None, uses CWD/data/input.
    """
    def _emit(msg: str):
        if report: report(msg)
        else: print(msg)

    data_input_root = Path(base_dir) if base_dir is not None else (Path.cwd() / "data" / "input")
    if not data_input_root.is_dir():
        msg = f"[error] Expected folder not found: {data_input_root}"
        _emit(msg); yield msg
        return

    grouped = collect_grouped_by_creative(data_input_root)
    for creative, paths_by_set in grouped.items():
        total = sum(len(v) for v in paths_by_set.values())
        hdr = f"\nCreative: {creative}  | total files: {total}"
        _emit(hdr); yield hdr
        for label, paths in paths_by_set.items():
            line = f"  {label}: {len(paths)}"
            _emit(line); yield line

    try:
        fp32_model, fp32_proc, fp32_build = build_fp32()
        msg = f"\n[FP32] build time: {fp32_build:.2f}s"
        _emit(msg); yield msg
    except Exception as e:
        err = f"[error] model build failed: {e}"
        _emit(err); yield err
        return

    for creative, paths_by_set in grouped.items():
        try:
            process_and_save_for_creative(creative, paths_by_set, fp32_model, fp32_proc)
            ok = f"[QWEN] {creative.name} → done"
            _emit(ok); yield ok
        except Exception as e:
            err = f"[QWEN] {creative.name} → ERROR: {e}"
            _emit(err); yield err

    done = "\n[done] All creatives processed."
    _emit(done); yield done

# ===== MAIN (standalone) =====
if __name__ == "__main__":
    root = Path.cwd() / "data" / "input"
    if not root.is_dir():
        raise SystemExit(f"[error] Expected folder not found: {root}")
    for _line in process_all_input(report=None, base_dir=root):
        pass
