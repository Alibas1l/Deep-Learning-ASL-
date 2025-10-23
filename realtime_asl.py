#!/usr/bin/env python3
import argparse
import os
import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_LABELS_29 = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "del","nothing","space"
]


class ASLRealtimeApp:
    def __init__(
        self,
        model_path: str,
        labels: List[str],
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        topk: int = 5,
        smooth_window: int = 8,
        use_roi: bool = True,
        roi_rel_size: float = 0.6,
        mirror: bool = True,
        min_confidence_to_show: float = 0.15,
    ) -> None:
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Determine model input size (H, W)
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if len(input_shape) == 4:
            self.input_h = int(input_shape[1])
            self.input_w = int(input_shape[2])
        else:
            # Fallback
            self.input_h = 64
            self.input_w = 64

        self.labels = labels
        self.num_classes = len(self.labels)
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.topk = max(1, min(topk, self.num_classes))
        self.smooth_window = max(1, smooth_window)
        self.scores_buffer: deque[np.ndarray] = deque(maxlen=self.smooth_window)
        self.use_roi = use_roi
        self.roi_rel_size = float(np.clip(roi_rel_size, 0.2, 0.95))
        self.mirror = mirror
        self.min_confidence_to_show = float(np.clip(min_confidence_to_show, 0.0, 1.0))

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_roi_rect: Tuple[int, int, int, int] | None = None

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        src = frame_bgr
        if self.use_roi:
            side = int(min(h, w) * self.roi_rel_size)
            x1 = w // 2 - side // 2
            y1 = h // 2 - side // 2
            x2 = x1 + side
            y2 = y1 + side
            self.last_roi_rect = (x1, y1, x2, y2)
            src = frame_bgr[y1:y2, x1:x2]
        else:
            self.last_roi_rect = None

        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32)  # Model contains Rescaling, so raw [0..255] is fine
        x = np.expand_dims(x, axis=0)   # (1, H, W, 3)
        return x, src

    def _predict(self, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
        probs = self.model.predict(x, verbose=0)[0]
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 1:
            probs = probs.reshape(-1)

        self.scores_buffer.append(probs)
        if len(self.scores_buffer) > 1:
            smoothed = np.mean(self.scores_buffer, axis=0)
        else:
            smoothed = probs

        smoothed = smoothed / (np.sum(smoothed) + 1e-8)
        top_idx = int(np.argmax(smoothed))
        top_conf = float(smoothed[top_idx])
        return top_idx, top_conf, smoothed

    @staticmethod
    def _draw_filled_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], alpha: float = 0.4) -> None:
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    def _draw_ui(
        self,
        frame_bgr: np.ndarray,
        roi_bgr: np.ndarray,
        top_idx: int,
        top_conf: float,
        probs: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr

        # ROI rectangle on main frame
        if self.last_roi_rect is not None:
            x1, y1, x2, y2 = self.last_roi_rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 200, 60), 2)

        # Header bar
        self._draw_filled_rect(canvas, (0, 0), (w, 60), (0, 0, 0), alpha=0.5)
        title = "ASL Detector"
        cv2.putText(canvas, title, (16, 40), self.font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        # Main prediction display
        pred_label = self.labels[top_idx]
        display_label = {
            "nothing": "Nothing",
            "space": "Space",
            "del": "Del",
        }.get(pred_label.lower(), pred_label.upper())

        pred_text = f"{display_label}  {int(round(top_conf * 100))}%"
        cv2.putText(canvas, pred_text, (w // 2 - 120, 40), self.font, 1.0, (80, 255, 140), 2, cv2.LINE_AA)

        # Side panel for top-k bars
        panel_w = 320
        self._draw_filled_rect(canvas, (0, 60), (panel_w, h), (0, 0, 0), alpha=0.35)

        order = np.argsort(probs)[::-1]
        bar_x = 16
        bar_y = 90
        bar_h = 24
        gap = 10
        bar_w_max = panel_w - 2 * bar_x - 60

        for i in range(min(self.topk, len(order))):
            idx = int(order[i])
            label = self.labels[idx]
            label = {"nothing": "Nothing", "space": "Space", "del": "Del"}.get(label.lower(), label.upper())
            p = float(probs[idx])
            pw = int(bar_w_max * p)
            y1 = bar_y + i * (bar_h + gap)
            y2 = y1 + bar_h
            # Bar background
            cv2.rectangle(canvas, (bar_x, y1), (bar_x + bar_w_max, y2), (60, 60, 60), 1)
            # Bar fill
            cv2.rectangle(canvas, (bar_x + 1, y1 + 1), (bar_x + pw, y2 - 1), (90, 200, 255), -1)
            # Text
            txt = f"{label}"
            cv2.putText(canvas, txt, (bar_x, y1 - 6), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{int(p * 100)}%", (bar_x + bar_w_max + 6, y2 - 4), self.font, 0.6, (200, 255, 200), 1, cv2.LINE_AA)

        # ROI preview box (top-right)
        preview_size = 180
        px2 = w - preview_size - 16
        py2 = 76
        self._draw_filled_rect(canvas, (px2 - 4, py2 - 24), (px2 + preview_size + 4, py2 + preview_size + 4), (0, 0, 0), alpha=0.4)
        cv2.putText(canvas, "ROI", (px2, py2 - 6), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if roi_bgr.size > 0:
            roi_preview = cv2.resize(roi_bgr, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
            canvas[py2:py2 + preview_size, px2:px2 + preview_size] = roi_preview

        # Footer: controls and FPS
        controls = "q: quit  r: toggle ROI  m: mirror"
        self._draw_filled_rect(canvas, (0, h - 36), (w, h), (0, 0, 0), alpha=0.5)
        cv2.putText(canvas, controls, (16, h - 12), self.font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 140, h - 12), self.font, 0.6, (180, 255, 180), 1, cv2.LINE_AA)

        # If confidence is low, subtly indicate uncertainty
        if top_conf < self.min_confidence_to_show:
            cv2.putText(canvas, "Low confidence", (w // 2 - 80, 20), self.font, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

        return canvas

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.frame_height))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                x, roi_bgr = self._preprocess(frame)
                top_idx, top_conf, probs = self._predict(x)

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps > 0 else (1.0 / dt)

                vis = self._draw_ui(frame, roi_bgr, top_idx, top_conf, probs, fps)

                cv2.imshow("ASL Detector", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.use_roi = not self.use_roi
                elif key == ord('m'):
                    self.mirror = not self.mirror
        finally:
            cap.release()
            cv2.destroyAllWindows()


def load_labels(path: str | None) -> List[str]:
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    return DEFAULT_LABELS_29


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time ASL detector using a Keras model and your webcam")
    parser.add_argument("--model", type=str, default="LSignLD.h5", help="Path to .h5 or SavedModel directory")
    parser.add_argument("--labels", type=str, default=None, help="Optional path to labels file (one per line)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--topk", type=int, default=5, help="Show top-K classes")
    parser.add_argument("--smooth", type=int, default=8, help="Temporal smoothing window size")
    parser.add_argument("--no-roi", action="store_true", help="Disable center-square ROI; use full frame")
    parser.add_argument("--roi-size", type=float, default=0.6, help="Relative ROI square size (0.2..0.95)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror (flip) effect")
    parser.add_argument("--min-conf", type=float, default=0.15, help="Show low-confidence hint below this threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)

    app = ASLRealtimeApp(
        model_path=args.model,
        labels=labels,
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        topk=args.topk,
        smooth_window=args.smooth,
        use_roi=(not args.no_roi),
        roi_rel_size=args.roi_size,
        mirror=(not args.no_mirror),
        min_confidence_to_show=args.min_conf,
    )
    app.run()


if __name__ == "__main__":
    main()
