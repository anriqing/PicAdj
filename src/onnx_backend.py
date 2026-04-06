"""
Optional ONNX Runtime (CPU) segmentation -> quadrilateral fallback.

Place a compatible model at models/doc_segment.onnx or set PICADJ_ONNX_MODEL.

Expected model (flexible shapes supported):
- Input: float32 image, either NCHW [1,3,H,W] or NHWC [1,H,W,3], RGB, values in [0,1].
- Output: single tensor — per-pixel class / foreground score map
  [1,1,h,w], [1,h,w], or [1,H,W,3] (uses first channel). Values in [0,1] or logits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detect_document import (
    DetectionResult,
    MAX_SIDE_DEFAULT,
    _find_quads_in_edges,
    _score_candidate,
    detect_quad_opencv,
    order_corners_tl_tr_br_bl,
)


DEFAULT_MODEL_RELATIVE = os.path.join("models", "doc_segment.onnx")
CONFIDENCE_TRY_ONNX_BELOW = 0.52


@dataclass
class OnnxSessionHolder:
    session: object
    input_name: str
    input_shape: Sequence[Optional[int]]  # may contain None for dynamic dim
    nchw: bool


def _app_root() -> str:
    import sys

    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _resolve_model_path(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    env = os.environ.get("PICADJ_ONNX_MODEL", "").strip()
    if env and os.path.isfile(env):
        return env
    p = os.path.join(_app_root(), DEFAULT_MODEL_RELATIVE)
    if os.path.isfile(p):
        return p
    return None


def _create_session(model_path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def _parse_input_layout(sess) -> Tuple[str, List[Optional[int]], bool]:
    inp = sess.get_inputs()[0]
    name = inp.name
    clean: List[Optional[int]] = []
    for d in inp.shape:
        if d is None or (isinstance(d, str)):
            clean.append(None)
        else:
            try:
                di = int(d)
                clean.append(di if di > 0 else None)
            except (TypeError, ValueError):
                clean.append(None)

    if len(clean) == 4 and clean[1] in (1, 3) and clean[-1] not in (1, 3):
        nchw = True
    elif len(clean) == 4 and clean[-1] == 3:
        nchw = False
    else:
        # Guess from layout: if dim 1 is 3, NCHW
        nchw = len(clean) == 4 and clean[1] == 3
    return name, clean, nchw


def _pick_spatial_size(
    shape: List[Optional[int]], ih: int, iw: int
) -> Tuple[int, int]:
    h_in = None
    w_in = None
    if len(shape) == 4:
        if shape[2] is not None and shape[3] is not None:
            h_in, w_in = shape[2], shape[3]
        elif shape[1] is not None and shape[2] is not None and shape[1] == 3:
            h_in, w_in = shape[2], shape[3]
    elif len(shape) == 3:
        if shape[1] is not None and shape[2] is not None:
            h_in, w_in = shape[1], shape[2]
    if h_in is None or w_in is None:
        return min(512, ih), min(512, iw)
    return int(h_in), int(w_in)


def _prepare_input_blob(
    rgb: np.ndarray, nchw: bool, out_h: int, out_w: int
) -> np.ndarray:
    resized = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    if nchw:
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
    else:
        x = np.expand_dims(x, 0)
    return np.ascontiguousarray(x)


def _mask_from_output(out: np.ndarray) -> np.ndarray:
    arr = np.asarray(out)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[-1] if arr.shape[0] <= 4 else arr[:, :, 0]
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError("Unexpected ONNX output rank for mask")
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    mask = (arr > 0.5).astype(np.uint8) * 255
    return mask


def _quad_from_binary_mask(
    mask_255: np.ndarray, orig_shape: Tuple[int, int, int]
) -> Optional[DetectionResult]:
    oh, ow = orig_shape[0], orig_shape[1]
    mh, mw = mask_255.shape[:2]
    edges = cv2.Canny(mask_255, 50, 150)
    gray_aux = mask_255

    best_conf = -1.0
    best_ordered: Optional[np.ndarray] = None
    best_tag = ""
    for retr in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
        for conf, ordered, full_tag, _cnt in _find_quads_in_edges(
            edges, gray_aux, "onnx_mask", retr
        ):
            if conf > best_conf:
                best_conf = conf
                best_ordered = ordered
                best_tag = full_tag

    if best_ordered is None:
        cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        for eps_frac in (0.01, 0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                ordered = order_corners_tl_tr_br_bl(quad)
                conf = _score_candidate(
                    ordered, mw, mh, cnt, "onnx_contour_fallback", mask_255
                )
                best_conf = conf * 0.92
                best_ordered = ordered
                best_tag = "onnx_contour_fallback"
                break
        if best_ordered is None:
            return None

    sx = ow / float(mw)
    sy = oh / float(mh)
    corners = best_ordered.copy()
    corners[:, 0] *= sx
    corners[:, 1] *= sy
    conf = max(0.0, min(1.0, best_conf))
    return DetectionResult(
        corners=corners.astype(np.float32),
        confidence=conf,
        scale=1.0,
        method=best_tag + "_onnx",
    )


_holder: Optional[OnnxSessionHolder] = None
_model_path_loaded: Optional[str] = None


def get_onnx_session(model_path: Optional[str] = None) -> Optional[OnnxSessionHolder]:
    global _holder, _model_path_loaded
    path = _resolve_model_path(model_path)
    if path is None:
        return None
    if _holder is not None and _model_path_loaded == path:
        return _holder
    sess = _create_session(path)
    in_name, shape, nchw = _parse_input_layout(sess)
    _holder = OnnxSessionHolder(session=sess, input_name=in_name, input_shape=shape, nchw=nchw)
    _model_path_loaded = path
    return _holder


def run_segmentation_quad(
    image_bgr: np.ndarray,
    model_path: Optional[str] = None,
) -> Optional[DetectionResult]:
    holder = get_onnx_session(model_path)
    if holder is None:
        return None

    ih, iw = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    oh, ow = _pick_spatial_size(list(holder.input_shape), ih, iw)
    blob = _prepare_input_blob(rgb, holder.nchw, oh, ow)

    out = holder.session.run(None, {holder.input_name: blob})[0]
    mask_small = _mask_from_output(out)
    mask_small = cv2.resize(mask_small, (iw, ih), interpolation=cv2.INTER_NEAREST)
    return _quad_from_binary_mask(mask_small, image_bgr.shape)


def detect_quad_fused(
    image_bgr: np.ndarray,
    max_side: int = MAX_SIDE_DEFAULT,
    onnx_path: Optional[str] = None,
    try_onnx_below: float = CONFIDENCE_TRY_ONNX_BELOW,
) -> Tuple[Optional[DetectionResult], str]:
    """
    OpenCV first; if confidence < try_onnx_below and an ONNX model is available,
    run segmentation fallback and pick the higher-confidence result.
    """
    opencv_result, _ = detect_quad_opencv(image_bgr, max_side=max_side)
    notes: List[str] = []
    if opencv_result:
        notes.append(f"opencv:{opencv_result.method}:{opencv_result.confidence:.2f}")

    onnx_result = None
    if _resolve_model_path(onnx_path) is not None:
        need_onnx = opencv_result is None or opencv_result.confidence < try_onnx_below
        if need_onnx:
            try:
                onnx_result = run_segmentation_quad(image_bgr, onnx_path)
                if onnx_result:
                    notes.append(
                        f"onnx:{onnx_result.method}:{onnx_result.confidence:.2f}"
                    )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"onnx_error:{exc}")

    if opencv_result is None and onnx_result is None:
        return None, "; ".join(notes) if notes else "no_detection"

    if opencv_result is None:
        return onnx_result, "; ".join(notes)
    if onnx_result is None:
        return opencv_result, "; ".join(notes)
    if onnx_result.confidence > opencv_result.confidence + 0.05:
        return onnx_result, "; ".join(notes)
    return opencv_result, "; ".join(notes)
