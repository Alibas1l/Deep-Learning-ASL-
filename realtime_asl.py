import argparse
import os
import sys
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time ASL sign classification from webcam using a Keras model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LSignLD.h5",
        help="Path to the trained Keras .h5/.keras model",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Path to labels file (one label per line). If missing, a default ASL list is used.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (0 is default camera)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Force camera width (0 = leave default)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Force camera height (0 = leave default)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip frame horizontally (mirror view)",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.3,
        help="Exponential smoothing factor for probabilities [0..1] (higher = smoother)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top predictions to display",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ASL Detector",
        help="Window title",
    )
    return parser.parse_args()


def try_load_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as exc:
        print(f"[ERROR] Failed to load model '{model_path}': {exc}")
        sys.exit(2)


def default_asl_labels() -> List[str]:
    # Common ASL Alphabet dataset classes
    # If your training set used a different order, provide a labels.txt
    letters = [chr(ord('A') + i) for i in range(26)]
    extra = ["del", "nothing", "space"]
    return letters + extra


def load_labels(labels_path: str, num_classes: Optional[int]) -> List[str]:
    labels: List[str] = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f.readlines() if line.strip()]
            labels = raw
    else:
        labels = default_asl_labels()

    if num_classes is None:
        return labels

    # Adjust length to match model outputs
    if len(labels) > num_classes:
        labels = labels[:num_classes]
    elif len(labels) < num_classes:
        # Pad with generic names
        for i in range(len(labels), num_classes):
            labels.append(f"class_{i}")
    return labels


def ensure_uint8_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_keep_aspect(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas[y: y + nh, x: x + nw] = resized
    return canvas


def draw_transparent_rect(frame: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], alpha: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_text(frame: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255), scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_topk_panel(
    frame: np.ndarray,
    labels: List[str],
    probs: np.ndarray,
    topk: int,
    panel_rect: Tuple[int, int, int, int],
) -> None:
    x, y, w, h = panel_rect
    draw_transparent_rect(frame, x, y, w, h, (30, 30, 30), 0.55)

    margin = 12
    inner_x = x + margin
    inner_y = y + margin
    inner_w = w - 2 * margin

    draw_text(frame, "Top predictions", (inner_x, inner_y + 10), (255, 255, 255), 0.7, 2)

    sorted_idx = np.argsort(probs)[::-1][:topk]
    bar_top = inner_y + 34
    bar_h = 28
    bar_gap = 12
    max_bar_w = inner_w

    for rank, cls_idx in enumerate(sorted_idx):
        label = labels[int(cls_idx)] if int(cls_idx) < len(labels) else str(cls_idx)
        score = float(probs[int(cls_idx)])
        bar_w = int(max_bar_w * max(0.0, min(1.0, score)))
        y0 = bar_top + rank * (bar_h + bar_gap)

        # Bar background
        cv2.rectangle(frame, (inner_x, y0), (inner_x + max_bar_w, y0 + bar_h), (70, 70, 70), -1)
        # Bar fill
        cv2.rectangle(frame, (inner_x, y0), (inner_x + bar_w, y0 + bar_h), (80, 180, 60), -1)

        # Label and perc
        perc_text = f"{label}: {int(score * 100):d}%"
        draw_text(frame, perc_text, (inner_x + 6, y0 + bar_h - 6), (255, 255, 255), 0.6, 1)


def compute_fps(t_prev: float, smoothed: float, smoothing: float = 0.9) -> Tuple[float, float]:
    now = time.time()
    dt = max(1e-6, now - t_prev)
    instant = 1.0 / dt
    smoothed = smoothing * smoothed + (1.0 - smoothing) * instant
    return now, smoothed


def main() -> None:
    set_seed()
    args = parse_arguments()

    model = try_load_model(args.model)

    # Infer image size from model input
    try:
        input_shape = model.input_shape
        image_h = int(input_shape[1])
        image_w = int(input_shape[2])
        image_size = (image_h, image_w)
    except Exception:
        image_size = (64, 64)

    # Infer number of classes
    try:
        num_classes = int(model.output_shape[-1])
    except Exception:
        num_classes = None

    labels = load_labels(args.labels, num_classes)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}")
        sys.exit(3)

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_title = args.title
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    # Warmup model
    dummy = np.zeros((1, image_size[0], image_size[1], 3), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    ema_alpha = float(np.clip(args.ema, 0.0, 1.0))
    prob_smooth: Optional[np.ndarray] = None
    t_prev = time.time()
    fps_smoothed = 0.0

    paused = False

    try:
        while True:
            if not paused:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("[WARN] Failed to read frame from camera.")
                    break

                frame_bgr = ensure_uint8_bgr(frame_bgr)
                if args.flip:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                h, w = frame_bgr.shape[:2]

                # ROI on right side
                roi_w = int(0.36 * w)
                roi_h = int(0.72 * h)
                roi_x = max(12, w - roi_w - 18)
                roi_y = max(12, (h - roi_h) // 2)

                # Panel on left
                panel_w = int(0.32 * w)
                panel_h = roi_h
                panel_x = 18
                panel_y = roi_y

                # Draw ROI frame
                cv2.rectangle(frame_bgr, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (50, 220, 120), 2)
                draw_text(frame_bgr, "Place hand in the box", (roi_x + 10, roi_y - 10), (240, 240, 240), 0.7, 2)

                roi = frame_bgr[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_resized = cv2.resize(roi_rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
                    inp = roi_resized.astype(np.float32)[None, ...]  # model contains its own Rescaling layer

                    preds = model.predict(inp, verbose=0)
                    prob = preds[0].astype(np.float32)
                    # Normalize to be robust against any numeric drift
                    s = float(prob.sum())
                    if s > 1e-6:
                        prob = prob / s

                    if prob_smooth is None:
                        prob_smooth = prob
                    else:
                        prob_smooth = (1.0 - ema_alpha) * prob_smooth + ema_alpha * prob
                        s2 = float(prob_smooth.sum())
                        if s2 > 1e-6:
                            prob_smooth = prob_smooth / s2

                    # Draw predictions panel
                    top_probs = prob_smooth.copy() if prob_smooth is not None else prob
                    draw_topk_panel(frame_bgr, labels, top_probs, args.topk, (panel_x, panel_y, panel_w, panel_h))

                    # Headline predicted label
                    best_idx = int(np.argmax(top_probs))
                    best_label = labels[best_idx] if best_idx < len(labels) else str(best_idx)
                    best_conf = float(top_probs[best_idx])
                    draw_transparent_rect(frame_bgr, roi_x, roi_y - 46, roi_w, 36, (30, 30, 30), 0.7)
                    draw_text(
                        frame_bgr,
                        f"Predicted: {best_label}  ({int(best_conf * 100)}%)",
                        (roi_x + 10, roi_y - 20),
                        (255, 255, 255),
                        0.8,
                        2,
                    )

                # Footer controls and FPS
                t_prev, fps_smoothed = compute_fps(t_prev, fps_smoothed)
                footer_text = f"FPS: {fps_smoothed:4.1f}   [q]=quit  [p]=pause/resume"
                draw_transparent_rect(frame_bgr, 10, h - 40, w - 20, 30, (30, 30, 30), 0.6)
                draw_text(frame_bgr, footer_text, (20, h - 18), (255, 255, 255), 0.7, 2)

                cv2.imshow(window_title, frame_bgr)
            else:
                # When paused, still show the last frame with a paused badge
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_text(frame, "Paused - press 'p' to resume", (40, 80), (255, 255, 255), 1.0, 2)
                cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('p'):
                paused = not paused

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
