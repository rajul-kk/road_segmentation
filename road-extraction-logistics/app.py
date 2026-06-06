"""
Road Segmentation — Gradio App

Four tabs:
  1. Inference    — upload satellite image → raw mask, cleaned mask, skeleton
  2. Pathfinding  — click two points on the image → A* path overlay
  3. Parameters   — live preview of CLAHE / threshold / kernel effects
  4. Batch Viewer — browse existing predicted masks from disk

Usage:
    pip install gradio
    python app.py
"""

import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import gradio as gr

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

import config as cfg
from models.architecture import load_model
from src.post_process import clean_mask, get_skeleton
from src.preprocessing import apply_clahe
from find_path import RoadPathfinder, pick_demo_endpoints
from src.pathfinder import AStarPathfinder
from src.path_utils import smooth_path, compute_path_distance


# ── Model (lazy-loaded on first inference call) ───────────────────────────────
_model       = None
_model_error = None
_to_tensor   = T.ToTensor()
_normalize   = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _ensure_model():
    global _model, _model_error
    if _model is not None:
        return _model, None
    if _model_error is not None:
        return None, _model_error
    try:
        _model = load_model()
        return _model, None
    except Exception as e:
        _model_error = str(e)
        return None, _model_error


# ── Pipeline helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_probs(pil_img: Image.Image) -> np.ndarray:
    """
    Run model inference. Returns raw road-class probability map (H, W) float32.
    Keeping raw probabilities allows live threshold adjustment without re-running
    the model.
    """
    model, err = _ensure_model()
    if model is None:
        raise RuntimeError(err)
    x      = _normalize(_to_tensor(pil_img.convert("RGB"))).unsqueeze(0).to(cfg.DEVICE)
    logits = model(x)["out"]                     # (1, 2, H, W)
    probs  = torch.softmax(logits, dim=1)[0, 1]  # road class (H, W)
    return probs.cpu().numpy().astype(np.float32)


def run_postprocess(probs: np.ndarray, threshold: float, kernel_size: int):
    """Apply threshold → morphological clean → skeletonize."""
    raw     = ((probs > threshold) * 255).astype(np.uint8)
    cleaned = clean_mask(raw, kernel_size=int(kernel_size))
    skel    = get_skeleton(cleaned)
    return Image.fromarray(raw), Image.fromarray(cleaned), Image.fromarray(skel)


# ── In-memory pathfinder adapter ──────────────────────────────────────────────
def _make_pathfinder(cleaned_arr: np.ndarray, road_threshold: int = 128) -> RoadPathfinder:
    """Build a RoadPathfinder from an in-memory cleaned mask (no disk I/O)."""
    pf              = object.__new__(RoadPathfinder)
    pf.mask_path    = None
    pf.road_threshold = road_threshold
    pf.use_skeleton = True
    pf.mask_array   = cleaned_arr
    pf.height, pf.width = cleaned_arr.shape
    pf.display_mask = cleaned_arr >= road_threshold
    skel            = get_skeleton(cleaned_arr)
    pf.road_mask    = skel > 127
    pf.pathfinder   = AStarPathfinder()
    return pf


# ── Visualization helpers ─────────────────────────────────────────────────────
def _overlay_path(path, sat_pil: Image.Image, display_mask: np.ndarray,
                  line_color=(255, 0, 0), line_width=3, opacity=0.7) -> Image.Image:
    """Draw road highlight + A* path on the satellite image (in memory)."""
    h, w = display_mask.shape
    sat  = sat_pil.convert("RGBA").resize((w, h))

    road_arr              = np.zeros((h, w, 4), dtype=np.uint8)
    road_arr[display_mask] = [255, 255, 0, int(255 * opacity * 0.3)]
    sat = Image.alpha_composite(sat, Image.fromarray(road_arr, "RGBA"))

    draw = ImageDraw.Draw(sat)
    if len(path) > 1:
        draw.line(path, fill=line_color + (255,), width=line_width)
    if len(path) >= 2:
        r = 8
        for pt, col in [(path[0], (0, 220, 0, 255)), (path[-1], (0, 100, 255, 255))]:
            draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=col)
    return sat.convert("RGB")


def _draw_markers(sat_pil: Image.Image, points: list) -> np.ndarray:
    """Draw start/end dots on a copy of the satellite image."""
    img  = sat_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    colors  = [(0, 220, 0), (0, 100, 255)]
    labels  = ["S", "E"]
    for i, pt in enumerate(points):
        r   = 10
        col = colors[i % 2]
        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r],
                     fill=col, outline=(255, 255, 255), width=2)
        draw.text((pt[0]+r+2, pt[1]-r), labels[i], fill=col)
    return np.array(img)


# ── Gradio app ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Road Segmentation", theme=gr.themes.Soft()) as demo:

    # ── Shared state ──────────────────────────────────────────────────────────
    sat_state      = gr.State(None)   # PIL.Image — current satellite image
    probs_state    = gr.State(None)   # np.ndarray (H,W) float32 — raw model output
    cleaned_state  = gr.State(None)   # np.ndarray (H,W) uint8 — cleaned binary mask
    click_state    = gr.State({"phase": "start", "start": None})
    batch_files    = gr.State([])
    batch_idx      = gr.State(0)

    gr.Markdown("# Road Segmentation")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 1 — Inference
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                t1_img    = gr.Image(type="pil", label="Satellite Image", sources=["upload"])
                t1_clahe  = gr.Checkbox(label="Apply CLAHE", value=cfg.USE_CLAHE)
                t1_clip   = gr.Slider(1.0, 8.0, value=2.0, step=0.5, label="CLAHE Clip Limit")
                t1_thresh = gr.Slider(0.1, 0.9, value=cfg.THRESHOLD, step=0.05, label="Threshold")
                t1_kernel = gr.Slider(3, 15, value=5, step=2, label="Morphology Kernel Size")
                t1_btn    = gr.Button("Run Inference", variant="primary")
                t1_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                with gr.Row():
                    t1_raw     = gr.Image(label="Raw Mask",     type="pil")
                    t1_cleaned = gr.Image(label="Cleaned Mask", type="pil")
                    t1_skel    = gr.Image(label="Skeleton",     type="pil")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 2 — Pathfinding
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Pathfinding"):
        with gr.Row():
            with gr.Column(scale=1):
                t2_thresh  = gr.Slider(0.1, 0.9, value=cfg.THRESHOLD, step=0.05, label="Threshold")
                t2_kernel  = gr.Slider(3, 15, value=5, step=2, label="Kernel Size")
                t2_status  = gr.Textbox(
                    value="Run inference in Tab 1, then click START point on the image.",
                    label="Status", interactive=False)
                t2_clear   = gr.Button("Clear Points")
                t2_stats   = gr.Textbox(label="Path Statistics", interactive=False, lines=4)
            with gr.Column(scale=2):
                t2_canvas  = gr.Image(label="Click to set path endpoints",
                                      type="numpy", interactive=True)
                t2_result  = gr.Image(label="Path Overlay", type="pil")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 3 — Parameters (live preview)
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Parameters"):
        gr.Markdown(
            "Adjust sliders to preview effects. "
            "The model only re-runs when you click **Run** or change CLAHE settings."
        )
        with gr.Row():
            with gr.Column(scale=1):
                t3_img    = gr.Image(type="pil", label="Satellite Image", sources=["upload"])
                t3_clahe  = gr.Checkbox(label="Apply CLAHE", value=cfg.USE_CLAHE)
                t3_clip   = gr.Slider(1.0, 8.0, value=2.0, step=0.5, label="CLAHE Clip Limit")
                t3_thresh = gr.Slider(0.1, 0.9, value=cfg.THRESHOLD, step=0.05, label="Threshold")
                t3_kernel = gr.Slider(3, 15, value=5, step=2, label="Kernel Size")
                t3_btn    = gr.Button("Run", variant="primary")
                t3_status = gr.Textbox(label="Status", interactive=False)
                t3_probs  = gr.State(None)
            with gr.Column(scale=2):
                with gr.Row():
                    t3_enhanced = gr.Image(label="CLAHE Enhanced", type="pil")
                    t3_raw      = gr.Image(label="Raw Mask",        type="pil")
                    t3_cleaned  = gr.Image(label="Cleaned Mask",    type="pil")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 4 — Batch Viewer
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Batch Viewer"):
        gr.Markdown("Browse predicted masks from disk and run automatic pathfinding.")
        with gr.Row():
            with gr.Column(scale=1):
                t4_folder  = gr.Textbox(value=cfg.MASK_DIR, label="Mask Folder")
                t4_load    = gr.Button("Load Masks", variant="primary")
                t4_drop    = gr.Dropdown(choices=[], label="Select Mask", interactive=True)
                with gr.Row():
                    t4_prev = gr.Button("◀ Prev")
                    t4_next = gr.Button("Next ▶")
                t4_path_btn = gr.Button("Find Auto Path")
                t4_status   = gr.Textbox(label="Status", interactive=False)
                t4_stats    = gr.Textbox(label="Path Statistics", interactive=False, lines=3)
            with gr.Column(scale=2):
                t4_mask   = gr.Image(label="Mask",         type="pil")
                t4_result = gr.Image(label="Path Overlay", type="pil")

    # ═══════════════════════════════════════════════════════════════════════════
    # Event handlers
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    def _tab1_run(img, use_clahe, clip_limit, threshold, kernel_size):
        if img is None:
            return (None, None, None, None, None, None,
                    "Upload an image first.", None)
        try:
            prepped     = apply_clahe(img, clip_limit=float(clip_limit)) if use_clahe else img
            probs       = predict_probs(prepped)
            raw, cleaned_pil, skel = run_postprocess(probs, threshold, int(kernel_size))
            cleaned_arr = np.array(cleaned_pil)
            coverage    = float((probs > threshold).mean()) * 100
            canvas      = np.array(img.convert("RGB"))
            return (raw, cleaned_pil, skel,
                    probs, cleaned_arr, img,
                    f"Done — road coverage: {coverage:.1f}%",
                    canvas)
        except Exception as e:
            return (None, None, None, None, None, None, f"Error: {e}", None)

    t1_btn.click(
        _tab1_run,
        inputs=[t1_img, t1_clahe, t1_clip, t1_thresh, t1_kernel],
        outputs=[t1_raw, t1_cleaned, t1_skel,
                 probs_state, cleaned_state, sat_state,
                 t1_status, t2_canvas],
    )

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    def _tab2_click(cs, sat, probs, threshold, kernel_size, evt: gr.SelectData):
        if sat is None or probs is None:
            return (None, cs,
                    "Run inference in Tab 1 first.", "", None)

        x, y = int(evt.index[0]), int(evt.index[1])

        if cs["phase"] == "start":
            new_cs  = {"phase": "end", "start": (x, y)}
            marked  = _draw_markers(sat, [(x, y)])
            return (marked, new_cs,
                    f"START set at ({x}, {y}). Now click END point.", "", None)

        # Second click — run pathfinding
        start = cs["start"]
        goal  = (x, y)
        marked = _draw_markers(sat, [start, goal])

        try:
            raw_bin     = ((probs > threshold) * 255).astype(np.uint8)
            cleaned_arr = clean_mask(raw_bin, kernel_size=int(kernel_size))
            pf          = _make_pathfinder(cleaned_arr)
            path        = pf.find_path(start, goal)

            if path:
                px, m    = compute_path_distance(path, cfg.PIXEL_RESOLUTION_METERS)
                euc      = float(np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2))
                dist_str = f"{m:.1f} m  ({px:.0f} px)" if m else f"{px:.0f} px"
                stats    = (f"Waypoints:  {len(path)}\n"
                            f"Distance:   {dist_str}\n"
                            f"Straight:   {euc:.0f} px\n"
                            f"Efficiency: {euc/px*100:.1f}%")
                result   = _overlay_path(path, sat, pf.display_mask)
                status   = f"Path found — {dist_str}"
            else:
                stats  = "No path found."
                result = None
                status = "No path found. Try different points or lower the threshold."
        except Exception as e:
            stats  = ""
            result = None
            status = f"Error: {e}"

        new_cs = {"phase": "start", "start": None}
        return (marked, new_cs, status, stats, result)

    def _tab2_clear(sat):
        canvas = np.array(sat.convert("RGB")) if sat is not None else None
        return (canvas,
                {"phase": "start", "start": None},
                "Points cleared. Click to set START point.", "", None)

    t2_canvas.select(
        _tab2_click,
        inputs=[click_state, sat_state, probs_state, t2_thresh, t2_kernel],
        outputs=[t2_canvas, click_state, t2_status, t2_stats, t2_result],
    )
    t2_clear.click(
        _tab2_clear,
        inputs=[sat_state],
        outputs=[t2_canvas, click_state, t2_status, t2_stats, t2_result],
    )

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    def _tab3_run(img, use_clahe, clip_limit, threshold, kernel_size):
        if img is None:
            return None, None, None, None, "Upload an image first."
        try:
            prepped = apply_clahe(img, clip_limit=float(clip_limit)) if use_clahe else img
            probs   = predict_probs(prepped)
            raw, cleaned, _ = run_postprocess(probs, threshold, int(kernel_size))
            return prepped, raw, cleaned, probs, "Done."
        except Exception as e:
            return None, None, None, None, f"Error: {e}"

    def _tab3_reprocess(probs, threshold, kernel_size):
        if probs is None:
            return None, None
        raw, cleaned, _ = run_postprocess(probs, threshold, int(kernel_size))
        return raw, cleaned

    t3_btn.click(
        _tab3_run,
        inputs=[t3_img, t3_clahe, t3_clip, t3_thresh, t3_kernel],
        outputs=[t3_enhanced, t3_raw, t3_cleaned, t3_probs, t3_status],
    )
    # Threshold and kernel sliders re-run only post-processing (no model call)
    t3_thresh.release(
        _tab3_reprocess,
        inputs=[t3_probs, t3_thresh, t3_kernel],
        outputs=[t3_raw, t3_cleaned],
    )
    t3_kernel.release(
        _tab3_reprocess,
        inputs=[t3_probs, t3_thresh, t3_kernel],
        outputs=[t3_raw, t3_cleaned],
    )

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    def _tab4_load(folder):
        if not os.path.isdir(folder):
            return [], gr.Dropdown(choices=[]), 0, "Folder not found.", None, None
        files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
        if not files:
            return [], gr.Dropdown(choices=[]), 0, "No PNG masks found.", None, None
        first = Image.open(os.path.join(folder, files[0])).convert("L")
        return (files, gr.Dropdown(choices=files, value=files[0]),
                0, f"Loaded {len(files)} masks.", first, None)

    def _tab4_show(files, idx, folder):
        if not files:
            return None
        return Image.open(os.path.join(folder, files[int(idx)])).convert("L")

    def _tab4_nav(files, idx, folder, delta):
        if not files:
            return idx, None, gr.Dropdown(), ""
        new_idx = int((int(idx) + delta) % len(files))
        img     = _tab4_show(files, new_idx, folder)
        return new_idx, img, gr.Dropdown(value=files[new_idx]), files[new_idx]

    def _tab4_select(choice, files, folder):
        if not files or choice not in files:
            return 0, None
        idx = files.index(choice)
        return idx, _tab4_show(files, idx, folder)

    def _tab4_autopath(files, idx, folder):
        if not files:
            return None, "Load masks first.", ""
        fname     = files[int(idx)]
        mask_path = os.path.join(folder, fname)
        try:
            pf            = RoadPathfinder(mask_path)
            start, goal   = pick_demo_endpoints(pf.display_mask)
            if start is None:
                return None, "Not enough road pixels in this mask.", ""
            path = pf.find_path(start, goal)
            if path is None:
                return None, "No path found.", ""

            px, m    = compute_path_distance(path, cfg.PIXEL_RESOLUTION_METERS)
            dist_str = f"{m:.1f} m  ({px:.0f} px)" if m else f"{px:.0f} px"
            stats    = f"Waypoints: {len(path)}\nDistance: {dist_str}"

            sat_name = fname.replace("_roadmask.png", "_sat.jpg")
            sat_path = os.path.join("data/raw/test", sat_name)
            if os.path.exists(sat_path):
                sat_img = Image.open(sat_path).convert("RGB")
                result  = _overlay_path(path, sat_img, pf.display_mask)
            else:
                result = pf.visualize_path(path)

            return result, f"Path found — {dist_str}", stats
        except Exception as e:
            return None, f"Error: {e}", ""

    t4_load.click(
        _tab4_load,
        inputs=[t4_folder],
        outputs=[batch_files, t4_drop, batch_idx, t4_status, t4_mask, t4_result],
    )
    t4_drop.change(
        _tab4_select,
        inputs=[t4_drop, batch_files, t4_folder],
        outputs=[batch_idx, t4_mask],
    )
    t4_prev.click(
        lambda f, i, folder: _tab4_nav(f, i, folder, -1),
        inputs=[batch_files, batch_idx, t4_folder],
        outputs=[batch_idx, t4_mask, t4_drop, t4_status],
    )
    t4_next.click(
        lambda f, i, folder: _tab4_nav(f, i, folder, +1),
        inputs=[batch_files, batch_idx, t4_folder],
        outputs=[batch_idx, t4_mask, t4_drop, t4_status],
    )
    t4_path_btn.click(
        _tab4_autopath,
        inputs=[batch_files, batch_idx, t4_folder],
        outputs=[t4_result, t4_status, t4_stats],
    )


if __name__ == "__main__":
    demo.launch()
