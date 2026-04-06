"""Perspective unwrap: sorted corners -> warped document image."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .detect_document import order_corners_tl_tr_br_bl


def destination_size_from_corners(corners_tl_tr_br_bl: np.ndarray) -> Tuple[int, int]:
    """
    Output (width, height) from edge lengths: max of top/bottom widths,
    max of left/right heights.
    """
    pts = np.asarray(corners_tl_tr_br_bl, dtype=np.float32).reshape(4, 2)
    (tl, tr, br, bl) = pts
    wa = np.linalg.norm(tr - tl)
    wb = np.linalg.norm(br - bl)
    ha = np.linalg.norm(bl - tl)
    hb = np.linalg.norm(br - tr)
    w = int(max(round(wa), round(wb)))
    h = int(max(round(ha), round(hb)))
    w = max(w, 1)
    h = max(h, 1)
    return w, h


def warp_document(
    image_bgr: np.ndarray,
    corners_tl_tr_br_bl: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Apply perspective transform. corners order must be tl, tr, br, bl in
    original image pixel coordinates.
    """
    img = image_bgr
    pts = order_corners_tl_tr_br_bl(corners_tl_tr_br_bl)
    if output_size is None:
        out_w, out_h = destination_size_from_corners(pts)
    else:
        out_w, out_h = output_size

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(
        img,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def optional_clahe_bgr(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Mild contrast normalize in LAB L channel (optional post-process)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def save_image(path: str, image_bgr: np.ndarray, jpeg_quality: int = 95) -> None:
    ext = path.lower().split(".")[-1]
    if ext in ("jpg", "jpeg"):
        cv2.imwrite(path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    else:
        cv2.imwrite(path, image_bgr)
