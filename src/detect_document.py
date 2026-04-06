"""Document quadrilateral detection using OpenCV (CPU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Quadrilateral in original image coordinates, ordered tl, tr, br, bl."""

    corners: np.ndarray  # (4, 2) float32, original pixel coords
    confidence: float  # 0..1
    scale: float  # processing scale vs original (1.0 = full res)
    method: str  # which internal path succeeded


MAX_SIDE_DEFAULT = 1200
MIN_QUAD_AREA_RATIO = 0.12
MAX_QUAD_AREA_RATIO = 0.95
MIN_ASPECT = 0.2
MAX_ASPECT = 5.0

EPS_FRACS = (
    0.002,
    0.004,
    0.006,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.07,
    0.09,
    0.12,
)


def order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Sort four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_area(pts: np.ndarray) -> float:
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def _quad_is_convex_ordered(ordered: np.ndarray) -> bool:
    pts = ordered.reshape(4, 2)
    signs: List[int] = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        signs.append(1 if cross > 1e-6 else (-1 if cross < -1e-6 else 0))
    nonzero = [s for s in signs if s != 0]
    if len(nonzero) < 3:
        return True
    return len(set(nonzero)) == 1


def _quad_compactness(quad: np.ndarray) -> float:
    """Area(quad) / Area(minAreaRect); in (0,1], 1 = perfectly aligned rectangle."""
    qa = _quad_area(quad)
    if qa < 1.0:
        return 0.0
    box = cv2.minAreaRect(quad.reshape(-1, 1, 2).astype(np.float32))
    bw, bh = box[1]
    ra = float(bw * bh)
    if ra < 1.0:
        return 0.0
    return float(np.clip(qa / ra, 0.0, 1.0))


def _interior_angle_deg(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    """Interior angle at vertex p for a convex polygon (walking p_prev -> p -> p_next)."""
    v1 = np.asarray(p_prev, dtype=np.float64) - p
    v2 = np.asarray(p_next, dtype=np.float64) - p
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _interior_angles_tl_tr_br_bl(ordered: np.ndarray) -> List[float]:
    """Four interior angles at tl, tr, br, bl."""
    p = ordered.reshape(4, 2)
    tl, tr, br, bl = p[0], p[1], p[2], p[3]
    return [
        _interior_angle_deg(bl, tl, tr),
        _interior_angle_deg(tl, tr, br),
        _interior_angle_deg(tr, br, bl),
        _interior_angle_deg(br, bl, tl),
    ]


def _line_intersection_xy(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> Optional[np.ndarray]:
    """Infinite line through a–b and c–d; None if parallel."""
    x1, y1 = float(a[0]), float(a[1])
    x2, y2 = float(b[0]), float(b[1])
    x3, y3 = float(c[0]), float(c[1])
    x4, y4 = float(d[0]), float(d[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-4:
        return None
    cx = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    cy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    return np.array([cx, cy], dtype=np.float64)


def _rectangle_plane_prior(ordered: np.ndarray, img_w: int, img_h: int) -> float:
    """
    Document pages are planar rectangles: convex quad with sane perspective, opposite
    sides meeting at plausible vanishing points (not inside the page region).
    """
    p = ordered.reshape(4, 2)
    tl, tr, br, bl = p[0], p[1], p[2], p[3]

    angs = _interior_angles_tl_tr_br_bl(ordered)
    ang_score = 1.0
    for a in angs:
        if a < 20.0 or a > 160.0:
            return 0.04
        if a < 34.0:
            ang_score *= 0.55 + 0.45 * (a - 20.0) / 14.0
        elif a > 146.0:
            ang_score *= 0.55 + 0.45 * (160.0 - a) / 14.0
    ang_score = float(max(0.15, min(1.0, ang_score)))

    lt = float(np.linalg.norm(tr - tl))
    lb = float(np.linalg.norm(br - bl))
    ll = float(np.linalg.norm(bl - tl))
    lr = float(np.linalg.norm(br - tr))
    r_tb = min(lt, lb) / max(lt, lb, 1e-6)
    r_lr = min(ll, lr) / max(ll, lr, 1e-6)
    side_score = float(0.45 * r_tb + 0.55 * r_lr)
    side_score = max(0.4, min(1.0, side_score ** 0.75))

    poly = ordered.reshape(-1, 1, 2).astype(np.float32)

    def _inside_page(pt: Optional[np.ndarray]) -> bool:
        if pt is None:
            return False
        v = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
        return v > 0.0

    vp_h = _line_intersection_xy(tl, tr, bl, br)
    vp_v = _line_intersection_xy(tl, bl, tr, br)
    vp_score = 1.0
    if _inside_page(vp_h):
        vp_score *= 0.62
    if _inside_page(vp_v):
        vp_score *= 0.62
    if vp_h is not None and vp_v is not None:
        diag = float((img_w ** 2 + img_h ** 2) ** 0.5)
        d_vps = float(np.linalg.norm(vp_h - vp_v))
        if d_vps < 0.06 * max(diag, 1.0):
            vp_score *= 0.88

    return float(max(0.08, min(1.0, ang_score * (0.62 + 0.38 * side_score) * vp_score)))


def _document_interior_margin_factor(
    ordered: np.ndarray, img_w: int, img_h: int
) -> float:
    """
    Physical pages are usually framed by background; true corners are rarely glued
    to the sensor crop. Strongly down-rank quads with multiple corners on/near the
    image border (a common failure mode when edges latch onto the full frame).
    """
    iw, ih = int(img_w), int(img_h)
    if iw < 32 or ih < 32:
        return 1.0
    min_dim = float(min(iw, ih))
    margin = max(10.0, 0.0175 * min_dim)
    prod = 1.0
    for x, y in ordered.reshape(4, 2):
        xf, yf = float(x), float(y)
        d = min(xf, float(iw - 1) - xf, yf, float(ih - 1) - yf)
        if d < margin:
            t = max(0.0, d / margin)
            prod *= 0.35 + 0.65 * t
    return float(max(0.08, min(1.0, prod)))


def _edge_consistency_score(gray: np.ndarray, ordered: np.ndarray) -> float:
    """Mean gradient along perimeter normals (higher = stronger edges on boundary)."""
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0.5
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy + 1e-6)
    pts = ordered.reshape(4, 2)
    samples = []
    steps = 12
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        edge = p1 - p0
        ln = float(np.linalg.norm(edge))
        if ln < 1e-3:
            continue
        tang = edge / ln
        nx, ny = -tang[1], tang[0]
        for t in np.linspace(0.05, 0.95, steps):
            x = p0[0] * (1 - t) + p1[0] * t
            y = p0[1] * (1 - t) + p1[1] * t
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            samples.append(float(mag[yi, xi]))
    if not samples:
        return 0.5
    m = float(np.median(samples))
    # Normalize roughly; gradient scale varies by image
    return float(np.clip(m / 80.0, 0.0, 1.0))


def _score_candidate(
    quad_raw: np.ndarray,
    img_w: int,
    img_h: int,
    contour: Optional[np.ndarray],
    tag: str,
    gray_work: Optional[np.ndarray],
) -> float:
    """Composite score 0..1; perspective quads need not have 90° corners in the image."""
    ordered = order_corners_tl_tr_br_bl(quad_raw)
    if not _quad_is_convex_ordered(ordered):
        return 0.0

    w, h = float(img_w), float(img_h)
    img_area = w * h
    qa = _quad_area(ordered)
    area_ratio = qa / img_area if img_area > 0 else 0.0
    if area_ratio < MIN_QUAD_AREA_RATIO or area_ratio > MAX_QUAD_AREA_RATIO:
        return 0.0

    rect_w = np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])
    rect_h = np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])
    aspect = (rect_w / 2.0) / max(rect_h / 2.0, 1e-6)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return 0.0

    # Prefer large, document-sized quads
    area_score = min(
        1.0,
        (area_ratio - MIN_QUAD_AREA_RATIO) / max(0.35 - MIN_QUAD_AREA_RATIO, 1e-6),
    )
    area_score = 0.2 + 0.8 * area_score

    compact = _quad_compactness(ordered)
    compact_score = 0.4 + 0.6 * compact

    # Contour / hull vs quad: good approximations align areas (both inputs convex-safe)
    fill_score = 0.75
    if contour is not None:
        ca = abs(cv2.contourArea(contour))
        ha = ca
        if len(contour) >= 3:
            hull = cv2.convexHull(contour)
            ha = abs(cv2.contourArea(hull))
        r_ca = min(ca, qa) / max(ca, qa, 1.0)
        r_ha = min(ha, qa) / max(ha, qa, 1.0)
        fill_score = float(0.45 * r_ca + 0.55 * r_ha)
        fill_score = max(0.25, min(1.0, fill_score))

    method_weight = 1.0
    if "minrect" in tag:
        method_weight = 0.55
    elif "hull" in tag:
        method_weight = 1.0
    elif "contour" in tag:
        method_weight = 0.92

    grad_score = 0.65
    if gray_work is not None:
        grad_score = 0.45 + 0.55 * _edge_consistency_score(gray_work, ordered)

    # Planar rectangle in perspective: tightens corner choice vs random quadrilaterals
    doc_rect_prior = _rectangle_plane_prior(ordered, img_w, img_h)

    base = (
        area_score * 0.26
        + compact_score * 0.26
        + fill_score * 0.20
        + grad_score * 0.28
    )
    margin_f = _document_interior_margin_factor(ordered, img_w, img_h)
    combined = base * method_weight * doc_rect_prior * margin_f
    return float(max(0.0, min(1.0, combined)))


def _approx_quad_on_poly(poly: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(poly, True)
    if peri < 1e-6:
        return None
    for ef in EPS_FRACS:
        a = cv2.approxPolyDP(poly, ef * peri, True)
        if len(a) == 4:
            return a.reshape(4, 2).astype(np.float32)
    return None


def _quads_from_contour(cnt: np.ndarray) -> List[Tuple[np.ndarray, str, np.ndarray]]:
    """Multiple extraction strategies; return (quad, tag, contour_used_for_scoring)."""
    out: List[Tuple[np.ndarray, str, np.ndarray]] = []

    hull = cv2.convexHull(cnt)
    if hull is None or len(hull) < 3:
        return out

    q = _approx_quad_on_poly(hull)
    if q is not None:
        out.append((q, "hull_approx", cnt))

    q2 = _approx_quad_on_poly(cnt)
    if q2 is not None:
        if not out or np.max(np.linalg.norm(q2 - out[0][0], axis=1)) > 2.0:
            out.append((q2, "contour_approx", cnt))

    box = cv2.minAreaRect(hull)
    pts = cv2.boxPoints(box).astype(np.float32)
    out.append((pts, "minrect_hull", cnt))

    return out


def _find_quads_in_edges(
    edges: np.ndarray,
    gray_work: np.ndarray,
    method_name: str,
    retrieval: int,
) -> List[Tuple[float, np.ndarray, str, np.ndarray]]:
    """Return list of (score, ordered_quad, method_name, contour)."""
    h, w = edges.shape[:2]
    cnts, _ = cv2.findContours(edges, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)
    scored: List[Tuple[float, np.ndarray, str, np.ndarray]] = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < img_area * (MIN_QUAD_AREA_RATIO * 0.85):
            continue

        for quad, tag, c in _quads_from_contour(cnt):
            conf = _score_candidate(quad, w, h, c, tag, gray_work)
            if conf <= 0:
                continue
            ordered = order_corners_tl_tr_br_bl(quad)
            full_tag = f"{method_name}/{tag}"
            scored.append((conf, ordered, full_tag, cnt))

    return scored


def _close_canny_gaps(edges: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    e = cv2.dilate(edges, k, iterations=1)
    return cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=2)


def _preprocess_variants(
    small_bgr: np.ndarray,
) -> List[Tuple[str, np.ndarray, bool]]:
    """
    Return (name, edge_or_binary, use_as_edges_directly).
    If use_as_edges_directly, matrix is already edge-like; else run Canny inside caller.
    """
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    blur5 = cv2.GaussianBlur(gray, (5, 5), 0)
    blur3 = cv2.GaussianBlur(gray, (3, 3), 0)

    variants: List[Tuple[str, np.ndarray, bool]] = []

    med = float(np.median(blur5))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    variants.append(("canny_median", cv2.Canny(blur5, lo, hi), True))

    variants.append(("canny_50_150", cv2.Canny(blur5, 50, 150), True))

    bil = cv2.bilateralFilter(gray, 9, 80, 80)
    variants.append(("canny_bilateral", cv2.Canny(bil, 30, 90), True))

    gx = cv2.Sobel(blur3, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur3, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_n = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sobel_e = cv2.threshold(mag_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("sobel_otsu", sobel_e, True))

    th = cv2.adaptiveThreshold(
        blur5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )
    th = cv2.medianBlur(th, 3)
    variants.append(("adaptive_canny", th, False))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ce = clahe.apply(blur5)
    th_clahe = cv2.adaptiveThreshold(
        l_ce, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )
    th_clahe = cv2.medianBlur(th_clahe, 3)
    variants.append(("adaptive_clahe_canny", th_clahe, False))

    sharp = cv2.addWeighted(
        blur5, 1.5, cv2.GaussianBlur(blur5, (0, 0), 3.0), -0.5, 0.0
    )
    th_sharp = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )
    th_sharp = cv2.medianBlur(th_sharp, 3)
    variants.append(("adaptive_sharp_canny", th_sharp, False))

    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k5, iterations=2)
    variants.append(("adaptive_closed_canny", closed, False))

    _, otsu = cv2.threshold(blur5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_canny", otsu, False))

    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s = cv2.GaussianBlur(s, (5, 5), 0)
    variants.append(("S_channel_canny", s, False))

    return variants


def _refine_corners_subpix(gray_full: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Refine corners on full-resolution grayscale."""
    c = corners.reshape(-1, 1, 2).astype(np.float32)
    h, w = gray_full.shape[:2]
    win = max(5, min(15, int(round(min(h, w) * 0.02)) | 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    try:
        cv2.cornerSubPix(gray_full, c, (win, win), (-1, -1), criteria)
    except cv2.error:
        pass
    out = c.reshape(4, 2)
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out.astype(np.float32)


def refine_corners_on_original(
    image_bgr: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Public: sub-pixel snap on original image (BGR)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return _refine_corners_subpix(gray, corners)


def _distinct_preprocess_support(
    ordq: np.ndarray,
    all_scored: List[Tuple[float, np.ndarray, str]],
    dist_px: float = 22.0,
) -> int:
    """
    Count how many independent preprocess branches (e.g. otsu_canny+closed vs
    S_channel_canny+closed) produced a quad congruent with ordq. Isolated high-score
    hits from a single branch are often spurious when other branches agree elsewhere.
    """
    ov = ordq.reshape(4, 2)
    paths: set[str] = set()
    for _c, o2, tag in all_scored:
        if (
            float(np.max(np.linalg.norm(ov - o2.reshape(4, 2), axis=1))) >= dist_px
        ):
            continue
        paths.add(tag.split("/")[0])
    return len(paths)


def _dedupe_quad_candidates(
    raw: List[Tuple[float, np.ndarray, str]],
    corner_dist_work: float = 16.0,
    max_keep: int = 48,
) -> List[Tuple[float, np.ndarray, str]]:
    """Keep best-scoring quad per cluster of similar corner sets (work coordinates)."""
    sorted_r = sorted(raw, key=lambda x: -x[0])
    kept: List[Tuple[float, np.ndarray, str]] = []
    for conf, ordq, tag in sorted_r:
        ov = ordq.reshape(4, 2)
        is_dup = False
        for _c, prev, _t in kept:
            pv = prev.reshape(4, 2)
            if float(np.max(np.linalg.norm(ov - pv, axis=1))) < corner_dist_work:
                is_dup = True
                break
        if not is_dup:
            kept.append((conf, ordq.copy(), tag))
        if len(kept) >= max_keep:
            break
    return kept


def _resolve_top_candidate(
    candidates: List[Tuple[float, np.ndarray, str]],
    gray_work: np.ndarray,
    all_scored: Optional[List[Tuple[float, np.ndarray, str]]] = None,
    score_slack: float = 0.075,
    max_pool: int = 14,
    multi_path_conf_proximity: float = 0.03,
) -> Tuple[float, np.ndarray, str]:
    """
    Primary sort: detection score. Among scores within score_slack of the best,
    prefer multi-path agreement (see _distinct_preprocess_support), then boundary
    gradients + compactness.
    """
    if not candidates:
        raise ValueError("candidates empty")
    by_score = sorted(candidates, key=lambda x: -x[0])
    top_conf = by_score[0][0]
    pool = [c for c in by_score if c[0] >= top_conf - score_slack][:max_pool]

    if all_scored is not None and len(all_scored) > 0:
        annotated: List[Tuple[float, np.ndarray, str, int]] = []
        for conf, ordq, tag in pool:
            sup = _distinct_preprocess_support(ordq, all_scored)
            annotated.append((conf, ordq, tag, sup))
        max_sup = max(a[3] for a in annotated)
        if max_sup >= 2:
            filtered: List[Tuple[float, np.ndarray, str, int]] = []
            for conf, ordq, tag, sup in annotated:
                if sup >= 2:
                    filtered.append((conf, ordq, tag, sup))
                    continue
                has_rival = any(
                    sup2 >= 2 and float(c2) >= float(conf) - multi_path_conf_proximity
                    for c2, _o2, _t2, sup2 in annotated
                )
                if not has_rival:
                    filtered.append((conf, ordq, tag, sup))
            if filtered:
                pool = [(c, o, t) for c, o, t, _ in filtered]

    best_c = pool[0]
    best_rank = -1.0
    for conf, ordq, tag in pool:
        ed = _edge_consistency_score(gray_work, ordq)
        comp = _quad_compactness(ordq)
        if all_scored is not None and len(all_scored) > 0:
            sup = _distinct_preprocess_support(ordq, all_scored)
        else:
            sup = 1
        sup_norm = float(min(1.0, max(0.0, (sup - 1) / 3.0)))
        rank = (
            float(conf) * 0.58
            + ed * 0.20
            + comp * 0.07
            + sup_norm * 0.15
        )
        if rank > best_rank:
            best_rank = rank
            best_c = (conf, ordq.copy(), tag)
    return best_c


def detect_quad_opencv(
    image_bgr: np.ndarray,
    max_side: int = MAX_SIDE_DEFAULT,
) -> Tuple[Optional[DetectionResult], float]:
    """
    Detect document quadrilateral. If image is downscaled internally, corners are
    mapped back to full image coordinates and refined with cornerSubPix.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, 1.0

    orig_h, orig_w = image_bgr.shape[:2]
    scale = 1.0
    work = image_bgr

    m = max(orig_h, orig_w)
    if m > max_side:
        scale = max_side / float(m)
        work = cv2.resize(
            image_bgr,
            (int(round(orig_w * scale)), int(round(orig_h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray_work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    all_scored: List[Tuple[float, np.ndarray, str]] = []

    retrievals = (cv2.RETR_EXTERNAL, cv2.RETR_LIST)

    for name, raw, is_edges in _preprocess_variants(work):
        if is_edges:
            edged = raw
        else:
            edged = cv2.Canny(raw, 40, 120)

        for variant_edges, suffix in (
            (edged, ""),
            (_close_canny_gaps(edged), "+closed"),
        ):
            for retr in retrievals:
                scored = _find_quads_in_edges(
                    variant_edges, gray_work, name + suffix, retr
                )
                for conf, ordered, full_tag, _cnt in scored:
                    all_scored.append((conf, ordered.copy(), full_tag + f"/{retr}"))

    if not all_scored:
        return None, scale

    unique = _dedupe_quad_candidates(all_scored)
    conf, ordered, method = _resolve_top_candidate(unique, gray_work, all_scored)
    # Sub-pixel refine on the same scale as detection. Refining directly on full-res
    # often snaps corners to interior text/texture (especially on 3–4K phone images).
    refined_work = _refine_corners_subpix(gray_work, ordered.astype(np.float32))
    inv = 1.0 / scale
    mapped = (refined_work * inv).astype(np.float32)
    # Clamp to full image (mapping can drift by fractions of a pixel)
    mapped[:, 0] = np.clip(mapped[:, 0], 0, float(orig_w - 1))
    mapped[:, 1] = np.clip(mapped[:, 1], 0, float(orig_h - 1))

    return (
        DetectionResult(
            corners=mapped,
            confidence=float(conf),
            scale=scale,
            method=method,
        ),
        scale,
    )


def default_corners_image(
    image_bgr: np.ndarray, margin_ratio: float = 0.08
) -> np.ndarray:
    """Full-image inset rectangle when detection fails (for manual adjust)."""
    h, w = image_bgr.shape[:2]
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    tl = np.array([mx, my], dtype=np.float32)
    tr = np.array([w - 1 - mx, my], dtype=np.float32)
    br = np.array([w - 1 - mx, h - 1 - my], dtype=np.float32)
    bl = np.array([mx, h - 1 - my], dtype=np.float32)
    return order_corners_tl_tr_br_bl(np.stack([tl, tr, br, bl], axis=0))
