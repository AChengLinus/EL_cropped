import sys, os, json, threading

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET = os.path.join(_ROOT, 'runtime', 'Lib', 'site-packages')
if os.path.isdir(_TARGET) and _TARGET not in sys.path:
    sys.path.insert(0, _TARGET)
_PYDIR = os.path.join(_ROOT, 'runtime')
for _sp in ['Lib\\site-packages', 'site-packages']:
    _p = os.path.join(_PYDIR, _sp)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
except ImportError:
    print('[ERROR] numpy not found.')
    input('Press Enter...'); sys.exit(1)
try:
    import cv2
except ImportError:
    print('[ERROR] opencv not found.')
    input('Press Enter...'); sys.exit(1)
try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print('[ERROR] flask not found.')
    input('Press Enter...'); sys.exit(1)

import base64, time, traceback
import concurrent.futures

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
HTML    = os.path.join(BASE, 'index.html')

USE_GPU = cv2.ocl.haveOpenCL()
if USE_GPU:
    cv2.ocl.setUseOpenCL(True)
WORKERS = min(os.cpu_count() or 4, 8)
pool    = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
app     = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# ================================================================
# 检测内核 v4.0 — 纯基础算法 (Hough + 亮度阈值)
# 无深度学习 · 无学习系统 · 离线可用
# ================================================================

def order_corners(pts):
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

def _expand_corners(pts, h_pad, v_pad=None):
    if v_pad is None:
        v_pad = h_pad
    cx, cy = pts.mean(axis=0)
    h_side = max(np.linalg.norm(pts[3] - pts[0]),
                  np.linalg.norm(pts[2] - pts[1]))
    expand_x = round(h_side * h_pad)
    expand_y = round(h_side * v_pad)
    result = pts.copy()
    for i in range(4):
        dx, dy = result[i] - np.array([cx, cy])
        ln = max(np.hypot(dx, dy), 1e-6)
        result[i] = result[i] + np.array([dx / ln * expand_x, dy / ln * expand_y])
    return result

def _hull_to_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)

def _clean(gray, top_frac=0.03):
    sh, sw = gray.shape
    g = gray.copy()
    g[:int(sh * top_frac), :] = 0
    if np.any(g > 20):
        p98 = float(np.percentile(g[g > 0], 98.5))
        hot = (g > max(p98 * 0.92, 20)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(hot)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < sw * sh * 0.008:
                g[labels == i] = 0
    return g


# ================================================================
# Hough 直线检测法（主检测方法）
# ================================================================

def _method_hough(gray, sh, sw, bcx, bcy):
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 20, 60)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    min_ll = int(min(sw, sh) * 0.10)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=min_ll, maxLineGap=25)
    if lines is None or len(lines) < 4:
        return None

    lines_arr = lines.reshape(-1, 4)
    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lines_arr:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.hypot(x2 - x1, y2 - y1)
        if angle < 20 or angle > 160:
            h_lines.append(((y1 + y2) / 2, length, x1, y1, x2, y2))
        elif 70 < angle < 110:
            v_lines.append(((x1 + x2) / 2, length, x1, y1, x2, y2))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    long_h = sw * 0.35
    long_v = sh * 0.15

    bot_sorted = sorted(
        [t for t in h_lines if t[0] > bcy + sh * 0.05],
        key=lambda t: t[0])
    bot_line = None
    for t in bot_sorted:
        if t[1] >= long_h:
            bot_line = t; break
    if bot_line is None:
        bot_med = [t for t in bot_sorted if t[1] >= min_ll]
        bot_line = bot_med[0] if bot_med else None
    if bot_line is None:
        return None

    top_sorted = sorted(
        [t for t in h_lines if t[0] < bcy - sh * 0.05],
        key=lambda t: -t[0])
    top_line = None
    for t in top_sorted:
        if t[1] >= long_h:
            top_line = t; break
    if top_line is None:
        top_med = [t for t in top_sorted if t[1] >= min_ll]
        top_line = top_med[0] if top_med else None
    if top_line is None:
        return None

    left_sorted = sorted(
        [t for t in v_lines if t[0] < bcx - sw * 0.05],
        key=lambda t: t[0])
    left_line = None
    for t in left_sorted:
        if t[1] >= long_v:
            left_line = t; break
    if left_line is None:
        return None

    right_sorted = sorted(
        [t for t in v_lines if t[0] > bcx + sw * 0.05],
        key=lambda t: -t[0])
    right_line = None
    for t in right_sorted:
        if t[1] >= long_v:
            right_line = t; break
    if right_line is None:
        return None

    def seg_to_line(x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        ln = max(np.hypot(vx, vy), 1e-6)
        return [vx / ln, vy / ln, (x1 + x2) / 2, (y1 + y2) / 2]

    def line_intersect(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    tl = seg_to_line(*top_line[2:])
    bl = seg_to_line(*bot_line[2:])
    ll = seg_to_line(*left_line[2:])
    rl = seg_to_line(*right_line[2:])

    corners = [
        line_intersect(tl, ll),
        line_intersect(tl, rl),
        line_intersect(bl, rl),
        line_intersect(bl, ll),
    ]
    if any(c is None for c in corners):
        return None

    pts = order_corners(np.array(corners, dtype=np.float32))

    w1 = np.linalg.norm(pts[1] - pts[0])
    h1 = np.linalg.norm(pts[3] - pts[0])
    ar = w1 / max(h1, 1e-6)
    if ar < 1.4 or ar > 3.5:
        return None

    return pts


# ================================================================
# 备用检测方法 — 亮度 Otsu 阈值法
# ================================================================

def _method_brightness(gray, sh, sw, gray_blur=None):
    blur = gray_blur if gray_blur is not None else cv2.GaussianBlur(_clean(gray), (31, 31), 0)
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    best_pts, best_score = None, -1
    for frac in [1.0, 0.90, 0.82, 0.74, 0.66, 0.58, 0.50, 0.42, 0.35]:
        thresh = max(8, otsu_val * frac)
        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.zeros_like(mask)
        m[labels == largest] = 255
        ys, xs = np.where(m > 0)
        if len(xs) < 40:
            continue
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32))
        area = cv2.contourArea(hull)
        if area < sw * sh * 0.08 or area > sw * sh * 0.92:
            continue
        pts = _hull_to_quad(hull)
        o = order_corners(pts)
        w1 = np.linalg.norm(o[1] - o[0])
        w2 = np.linalg.norm(o[2] - o[3])
        h1 = np.linalg.norm(o[3] - o[0])
        h2 = np.linalg.norm(o[2] - o[1])
        ar = max(w1, w2) / max(max(h1, h2), 1e-6)
        if ar < 1.4 or ar > 3.5:
            continue
        rect = (min(w1, w2) / max(w1, w2)) * (min(h1, h2) / max(h1, h2))
        score = area * rect * max(1.0 - abs(ar - 2.1) / 2.0, 0.3)
        if score > best_score:
            best_score, best_pts = score, pts

    return order_corners(best_pts) if best_pts is not None else None


def _refine_corners_by_lines(gray, rough_pts, sw, sh):
    blur = cv2.GaussianBlur(_clean(gray), (5, 5), 0)
    edges = cv2.Canny(blur, 25, 75)
    margin_n = max(14, int(min(sw, sh) * 0.028))
    quad_mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(quad_mask, [rough_pts.astype(np.int32)], 255)
    expand_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (margin_n * 2 + 1, margin_n * 2 + 1))
    quad_mask = cv2.dilate(quad_mask, expand_k, iterations=1)
    edges = cv2.bitwise_and(edges, quad_mask)
    ys, xs = np.where(edges > 0)
    if len(xs) < 40:
        return rough_pts

    pts_arr = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    o = order_corners(rough_pts)
    fitted = []
    for i in range(4):
        p1, p2 = o[i], o[(i + 1) % 4]
        dx, dy = p2 - p1
        length = max(np.hypot(dx, dy), 1e-6)
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        rel = pts_arr - p1
        d_n = rel[:, 0] * nx + rel[:, 1] * ny
        d_t = rel[:, 0] * tx + rel[:, 1] * ty
        margin_t = length * 0.10
        sel = (np.abs(d_n) < margin_n) & \
              (d_t > -margin_t) & (d_t < length + margin_t)
        side_pts = pts_arr[sel]
        if len(side_pts) < 6:
            fitted.append(None)
            continue
        line = cv2.fitLine(side_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        fitted.append(line)

    def _line_intersect(l1, l2):
        if l1 is None or l2 is None:
            return None
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    refined = []
    max_shift = min(sw, sh) * 0.07
    for i in range(4):
        pt = _line_intersect(fitted[(i - 1) % 4], fitted[i])
        if pt is not None:
            dist = np.hypot(pt[0] - o[i][0], pt[1] - o[i][1])
            if dist < max_shift:
                refined.append(pt)
                continue
        refined.append(o[i])
    return np.array(refined, dtype=np.float32)


# ================================================================
# 主检测入口（纯基础算法）
# ================================================================

def detect_corners_v2(img, pad_pct=0.005, v_pad=None):
    if v_pad is None:
        v_pad = pad_pct
    H, W = img.shape[:2]
    sc = min(1.0, 1200 / max(W, H))
    sm = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)

    cleaned = _clean(gray)
    gray_blur = cv2.GaussianBlur(cleaned, (31, 31), 0)
    otsu_val, _ = cv2.threshold(gray_blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bright_mask = cv2.threshold(gray_blur, otsu_val * 0.6, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(bright_mask > 0)
    if len(xs) < 100:
        return None
    bcx, bcy = float(np.mean(xs)), float(np.mean(ys))

    best_pts = _method_hough(gray, sh, sw, bcx, bcy)

    if best_pts is None:
        best_pts = _method_brightness(gray, sh, sw, gray_blur)

    if best_pts is None:
        return None

    best_pts = _refine_corners_by_lines(gray, order_corners(best_pts), sw, sh)
    best_pts = order_corners(best_pts)

    inv = 1.0 / sc
    pts = order_corners(best_pts * inv)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    best_pts = _expand_corners(pts, pad_pct, v_pad)
    for i in range(4):
        best_pts[i, 0] = np.clip(best_pts[i, 0], 0, W - 1)
        best_pts[i, 1] = np.clip(best_pts[i, 1], 0, H - 1)

    return order_corners(best_pts)


def warp_image(img, pts, target_w=None, target_h=None):
    tl, tr, br, bl = pts
    if target_w and target_h:
        dw, dh = int(target_w), int(target_h)
    else:
        dw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        dh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if dw < 10 or dh < 10:
        raise ValueError('Invalid corners')
    dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    if USE_GPU:
        return cv2.warpPerspective(cv2.UMat(img), M, (dw, dh)).get()
    return cv2.warpPerspective(img, M, (dw, dh))


def process_image(img_bytes, pad_pct, forced=None, img_name=None, v_pad=None, target_w=None, target_h=None):
    """处理流程：forced 手动角点 → 自动检测 (Hough + 亮度备用)"""
    if v_pad is None:
        v_pad = pad_pct
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None, 'Decode failed'
        H, W = img.shape[:2]

        if forced is not None:
            pts = np.array(forced, dtype=np.float32)
        else:
            pts = detect_corners_v2(img, pad_pct, v_pad)
            if pts is None:
                return None, None, 'Panel not detected'

        warped = warp_image(img, pts, target_w, target_h)
        ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (buf.tobytes(), pts.tolist(), None) if ok else (None, None, 'Encode failed')
    except Exception as e:
        return None, None, str(e)


# ================================================================
# 无人机拍摄模式
# ================================================================

def _drone_find_dominant_row(gray, x_left, x_right, y_top, y_bot):
    col_strip = gray[y_top:y_bot, x_left:x_right]
    if col_strip.shape[0] < 50:
        return y_top, y_bot
    row_mean = col_strip.astype(np.float32).mean(axis=1)
    ksize = max(31, int(len(row_mean) * 0.02)) | 1
    row_smooth = cv2.GaussianBlur(row_mean.reshape(-1, 1), (1, ksize), 0).flatten()
    max_b = float(row_smooth.max())
    th = max_b * 0.55
    n = len(row_smooth)
    is_bright = row_smooth > th
    segs = []
    start = None
    min_seg = max(100, int(n * 0.08))
    for i in range(n):
        if is_bright[i] and start is None:
            start = i
        elif not is_bright[i] and start is not None:
            if i - start >= min_seg:
                seg_mean = float(row_smooth[start:i].mean())
                segs.append((start, i, seg_mean, i - start))
            start = None
    if start is not None and n - start >= min_seg:
        seg_mean = float(row_smooth[start:n].mean())
        segs.append((start, n, seg_mean, n - start))
    if not segs:
        return y_top, y_bot
    segs.sort(key=lambda s: -s[3])
    best = segs[0]
    return y_top + best[0], y_top + best[1]


def _drone_find_panel_region(gray):
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(blur, max(otsu_val * 0.5, 15), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == largest)
    if len(xs) < 100:
        return None
    x_left = int(xs.min())
    x_right = int(xs.max())
    x_mid_s = x_left + int((x_right - x_left) * 0.2)
    x_mid_e = x_left + int((x_right - x_left) * 0.8)
    strip = gray[:, x_mid_s:x_mid_e]
    row_mean = strip.astype(np.float32).mean(axis=1)
    ksize = max(21, int(len(row_mean) * 0.01)) | 1
    rs = cv2.GaussianBlur(row_mean.reshape(-1, 1), (1, ksize), 0).flatten()
    max_b = float(rs.max())
    bright_th = max_b * 0.65
    bands = []
    in_b = False
    bs = 0
    for i in range(len(rs)):
        if rs[i] > bright_th and not in_b:
            bs = i; in_b = True
        elif rs[i] <= bright_th and in_b:
            bands.append([bs, i])
            in_b = False
    if in_b:
        bands.append([bs, len(rs)])
    if not bands:
        return (int(ys.min()), int(ys.max()), x_left, x_right)
    merged = [bands[0]]
    for b in bands[1:]:
        gap_start = merged[-1][1]
        gap_end = b[0]
        gap_w = gap_end - gap_start
        if gap_w == 0:
            merged[-1][1] = b[1]
            continue
        gap_min = float(rs[gap_start:gap_end].min())
        if gap_w < H * 0.03 and gap_min > max_b * 0.55:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))
    merged = [m for m in merged if (m[1] - m[0]) > H * 0.25]
    if not merged:
        return (int(ys.min()), int(ys.max()), x_left, x_right)
    merged.sort(key=lambda b: -(b[1] - b[0]))
    best = merged[0]
    return (best[0], best[1], x_left, x_right)


def _drone_detect_n_panels(gray, region, max_panels=3):
    y_top, y_bot, x_left, x_right = region
    h = y_bot - y_top
    r_top = y_top + int(h * 0.2)
    r_bot = y_bot - int(h * 0.2)
    roi = gray[r_top:r_bot, x_left:x_right]
    if roi.shape[0] < 10 or roi.shape[1] < 10:
        return []
    col_mean = roi.astype(np.float32).mean(axis=0)
    ksize = max(21, int(len(col_mean) * 0.02)) | 1
    col_smooth = cv2.GaussianBlur(col_mean.reshape(1, -1), (ksize, 1), 0).flatten()
    w = len(col_smooth)
    max_b = float(col_smooth.max())
    valleys = []
    depth_min = max_b * 0.08
    for i in range(20, w - 20):
        if col_smooth[i] < col_smooth[i - 15] and col_smooth[i] < col_smooth[i + 15]:
            left_max = col_smooth[max(0, i - 200):i].max()
            right_max = col_smooth[i:min(w, i + 200)].max()
            depth = min(left_max, right_max) - col_smooth[i]
            if depth > depth_min:
                valleys.append((i, col_smooth[i], depth))
    valleys.sort(key=lambda v: -v[2])
    min_sep = max(80, int(w * 0.10))
    deduped = []
    for v in valleys:
        if all(abs(v[0] - sv[0]) > min_sep for sv in deduped):
            deduped.append(v)
    deduped.sort(key=lambda v: v[0])
    split_positions = [v[0] for v in deduped]
    W_img = gray.shape[1]
    edge_tol = max(4, int(W_img * 0.01))
    x_bounds = [0] + split_positions + [w]
    raw_segs = []
    for i in range(len(x_bounds) - 1):
        s, e = x_bounds[i], x_bounds[i + 1]
        seg_w = e - s
        if seg_w < w * 0.08:
            continue
        region_mean = float(col_smooth[s:e].mean())
        if region_mean <= max_b * 0.55:
            continue
        abs_xl = x_left + s
        abs_xr = x_left + e
        on_left_edge = abs_xl <= edge_tol
        on_right_edge = abs_xr >= W_img - edge_tol
        raw_segs.append({'xl': abs_xl, 'xr': abs_xr, 'w': seg_w,
                         'on_edge': on_left_edge or on_right_edge})
    if not raw_segs:
        return []
    inner_widths = [s['w'] for s in raw_segs if not s['on_edge']]
    if inner_widths:
        ref_w = float(np.median(inner_widths))
        kept = []
        for s in raw_segs:
            if s['on_edge'] and s['w'] < ref_w * 0.75:
                continue
            kept.append(s)
        raw_segs = kept
    else:
        raw_segs = [s for s in raw_segs if s['w'] >= w * 0.15]
    candidates = [(s['xl'], s['xr']) for s in raw_segs]
    return candidates[:max_panels]


def _refine_panel_corners(edges, rough_pts, sw, sh):
    pts = order_corners(rough_pts)
    def fit_edge_points(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max(np.hypot(dx, dy), 1e-6)
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        margin_n = max(8, int(min(sw, sh) * 0.015))
        margin_t = length * 0.03
        ys, xs = np.where(edges > 0)
        if len(xs) < 10:
            return None
        ptsE = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        rel = ptsE - np.array([p1[0], p1[1]], dtype=np.float32)
        d_n = rel[:, 0] * nx + rel[:, 1] * ny
        d_t = rel[:, 0] * tx + rel[:, 1] * ty
        sel = (np.abs(d_n) < margin_n) & (d_t > -margin_t) & (d_t < length + margin_t)
        side_pts = ptsE[sel]
        if len(side_pts) < 10:
            return None
        line = cv2.fitLine(side_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
    edges_lines = [
        fit_edge_points(pts[0], pts[1]), fit_edge_points(pts[1], pts[2]),
        fit_edge_points(pts[2], pts[3]), fit_edge_points(pts[3], pts[0])]
    def line_intersect2(l1, l2):
        if l1 is None or l2 is None:
            return None
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)
    refined = []
    max_shift = min(sw, sh) * 0.05
    for i in range(4):
        new_pt = line_intersect2(edges_lines[(i - 1) % 4], edges_lines[i])
        if new_pt is not None:
            dist = np.hypot(new_pt[0] - pts[i][0], new_pt[1] - pts[i][1])
            if dist < max_shift:
                refined.append(new_pt)
                continue
        refined.append(pts[i])
    return np.array(refined, dtype=np.float32)


def _drone_detect_single_panel(gray_roi, pad_pct=0.005):
    sh, sw = gray_roi.shape[:2]
    def seg_to_line(x1, y1, x2, y2):
        vx, vy = float(x2 - x1), float(y2 - y1)
        ln = max(np.hypot(vx, vy), 1e-6)
        return [vx / ln, vy / ln, (x1 + x2) / 2.0, (y1 + y2) / 2.0]
    def line_intersect(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)
    blur_otsu = cv2.GaussianBlur(gray_roi, (25, 25), 0)
    otsu_val, _ = cv2.threshold(blur_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bright_mask = cv2.threshold(blur_otsu, max(otsu_val * 0.5, 15), 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, k3, iterations=3)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, k3, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    panel_top, panel_bot, panel_left, panel_right = 0, sh - 1, 0, sw - 1
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pl, pt = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP]
        pw, ph = stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]
        if pw * ph > sw * sh * 0.15:
            panel_left, panel_top = pl, pt
            panel_right, panel_bot = pl + pw - 1, pt + ph - 1
    blur = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 80)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    min_ll = int(min(sw, sh) * 0.15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_ll, maxLineGap=15)
    top_line = bot_line = left_line = right_line = None
    if lines is not None and len(lines) >= 2:
        lines_arr = lines.reshape(-1, 4)
        h_lines, v_lines = [], []
        for x1, y1, x2, y2 in lines_arr:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.hypot(x2 - x1, y2 - y1)
            if angle < 20 or angle > 160:
                h_lines.append(((y1 + y2) / 2, length, x1, y1, x2, y2))
            elif 70 < angle < 110:
                v_lines.append(((x1 + x2) / 2, length, x1, y1, x2, y2))
        long_h, long_v = sw * 0.30, sh * 0.20
        tol = min(sw, sh) * 0.08
        top_cands = [t for t in h_lines if t[1] >= long_h and abs(t[0] - panel_top) < tol]
        if top_cands: top_line = min(top_cands, key=lambda t: abs(t[0] - panel_top))
        bot_cands = [t for t in h_lines if t[1] >= long_h and abs(t[0] - panel_bot) < tol]
        if bot_cands: bot_line = min(bot_cands, key=lambda t: abs(t[0] - panel_bot))
        left_cands = [t for t in v_lines if t[1] >= long_v and abs(t[0] - panel_left) < tol]
        if left_cands: left_line = min(left_cands, key=lambda t: abs(t[0] - panel_left))
        right_cands = [t for t in v_lines if t[1] >= long_v and abs(t[0] - panel_right) < tol]
        if right_cands: right_line = min(right_cands, key=lambda t: abs(t[0] - panel_right))
    all_hough_found = (top_line is not None and bot_line is not None and
                       left_line is not None and right_line is not None)
    m = 2
    if top_line is None: top_line = (max(m, panel_top), sw, 0, max(m, panel_top), sw - 1, max(m, panel_top))
    if bot_line is None: bot_line = (min(sh - m, panel_bot), sw, 0, min(sh - m, panel_bot), sw - 1, min(sh - m, panel_bot))
    if left_line is None: left_line = (max(m, panel_left), sh, max(m, panel_left), 0, max(m, panel_left), sh - 1)
    if right_line is None: right_line = (min(sw - m, panel_right), sh, min(sw - m, panel_right), 0, min(sw - m, panel_right), sh - 1)
    if abs(top_line[0] - bot_line[0]) > sh * 0.3 and abs(left_line[0] - right_line[0]) > sw * 0.3:
        tl = seg_to_line(*top_line[2:])
        bl = seg_to_line(*bot_line[2:])
        ll = seg_to_line(*left_line[2:])
        rl = seg_to_line(*right_line[2:])
        corners = [line_intersect(tl, ll), line_intersect(tl, rl),
                   line_intersect(bl, rl), line_intersect(bl, ll)]
        if all(c is not None for c in corners):
            pts = order_corners(np.array(corners, dtype=np.float32))
            w1 = np.linalg.norm(pts[1] - pts[0])
            h1 = np.linalg.norm(pts[3] - pts[0])
            ar = w1 / max(h1, 1e-6)
            if 0.15 < ar < 5.0:
                pts = _refine_panel_corners(edges, pts, sw, sh)
                if pad_pct > 0: pts = _expand_corners(pts, pad_pct)
                for i in range(4):
                    pts[i, 0] = np.clip(pts[i, 0], 0, sw - 1)
                    pts[i, 1] = np.clip(pts[i, 1], 0, sh - 1)
                return pts, all_hough_found
    blur2 = cv2.GaussianBlur(gray_roi, (31, 31), 0)
    otsu_val, _ = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best_pts, best_score = None, -1
    for frac in [1.0, 0.90, 0.82, 0.74, 0.66, 0.58, 0.50, 0.42, 0.35]:
        thresh = max(8, otsu_val * frac)
        _, mask = cv2.threshold(blur2, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ke, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ke, iterations=3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n < 2: continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.zeros_like(mask); m[labels == largest] = 255
        ys, xs = np.where(m > 0)
        if len(xs) < 40: continue
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32))
        area = cv2.contourArea(hull)
        if area < sw * sh * 0.08 or area > sw * sh * 0.98: continue
        pts = _hull_to_quad(hull)
        o = order_corners(pts)
        w1, w2 = np.linalg.norm(o[1] - o[0]), np.linalg.norm(o[2] - o[3])
        h1, h2 = np.linalg.norm(o[3] - o[0]), np.linalg.norm(o[2] - o[1])
        ar = max(w1, w2) / max(max(h1, h2), 1e-6)
        if ar < 0.2 or ar > 4.0: continue
        rect = (min(w1, w2) / max(w1, w2)) * (min(h1, h2) / max(h1, h2))
        score = area * rect
        if score > best_score: best_score, best_pts = score, pts
    if best_pts is None: return None, False
    pts = order_corners(best_pts)
    pts = _expand_corners(pts, pad_pct)
    for i in range(4):
        pts[i, 0] = np.clip(pts[i, 0], 0, sw - 1)
        pts[i, 1] = np.clip(pts[i, 1], 0, sh - 1)
    return pts, False


def _panel_is_complete(pts_full, W, H, margin_frac=0.02):
    margin_x = W * margin_frac
    margin_y = H * margin_frac
    for x, y in pts_full:
        if x < margin_x or x > W - margin_x: return False
        if y < margin_y or y > H - margin_y: return False
    return True


def process_drone_image(img_bytes, pad_pct=0.014, v_pad=None, force_w=None, force_h=None):
    if v_pad is None: v_pad = pad_pct
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None, 'Decode failed'
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        region = _drone_find_panel_region(gray)
        if region is None: return None, 'Panel region not detected'
        y_top, y_bot, x_left, x_right = region
        panel_ranges = _drone_detect_n_panels(gray, region, max_panels=3)
        if not panel_ranges: return None, 'No panels detected'
        n_panels = len(panel_ranges)
        all_labels = ['左', '中', '右'][:n_panels]
        valid_panels, skipped = [], []
        for pi in range(n_panels):
            xl_p, xr_p = panel_ranges[pi]
            pw = xr_p - xl_p
            margin = int(pw * 0.03)
            left_hard = (panel_ranges[pi - 1][1] + xl_p) // 2 if pi > 0 else 0
            right_hard = (xr_p + panel_ranges[pi + 1][0] + 1) // 2 if pi < n_panels - 1 else W
            xl = max(left_hard, xl_p - margin)
            xr = min(right_hard, xr_p + margin)
            margin_y = int(pw * 0.03)
            yt, yb = max(0, y_top - margin_y), min(H, y_bot + margin_y)
            label = all_labels[pi] if pi < len(all_labels) else str(pi)
            gray_roi = gray[yt:yb, xl:xr]
            pts_roi, hough_complete = _drone_detect_single_panel(gray_roi, 0)
            if pts_roi is None:
                skipped.append(label + '(检测失败)')
                continue
            pts_full = pts_roi.copy()
            pts_full[:, 0] += xl; pts_full[:, 1] += yt
            neighbor_tol = pw * 0.01
            if pi > 0:
                pts_full[0, 0] = max(pts_full[0, 0], xl_p - neighbor_tol)
                pts_full[3, 0] = max(pts_full[3, 0], xl_p - neighbor_tol)
            if pi < n_panels - 1:
                pts_full[1, 0] = min(pts_full[1, 0], xr_p + neighbor_tol)
                pts_full[2, 0] = min(pts_full[2, 0], xr_p + neighbor_tol)
            if pad_pct > 0:
                h_side = max(np.linalg.norm(pts_full[3] - pts_full[0]),
                              np.linalg.norm(pts_full[2] - pts_full[1]))
                cx, cy = pts_full.mean(axis=0)
                expand_h, expand_v = round(h_side * pad_pct), round(h_side * v_pad)
                for j in range(4):
                    dx, dy = pts_full[j] - np.array([cx, cy])
                    d = max(np.hypot(dx, dy), 1e-6)
                    pts_full[j] += np.array([dx / d * expand_h, dy / d * expand_v])
                pts_full[:, 0] = np.clip(pts_full[:, 0], 0, W - 1)
                pts_full[:, 1] = np.clip(pts_full[:, 1], 0, H - 1)
            if pad_pct <= 0:
                if not _panel_is_complete(pts_full, W, H, margin_frac=0.02):
                    skipped.append(label + '(超出画幅)')
                    continue
            valid_panels.append((pi, label, pts_full))
        if not valid_panels:
            reason = '、'.join(skipped) if skipped else '无候选'
            return None, '未检测到完整面板（' + reason + '）'
        panel_corners_list = [pts for _, _, pts in valid_panels]
        widths, heights = [], []
        for pts in panel_corners_list:
            widths.append(int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))))
            heights.append(int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))))
        target_w = max(int(np.median(widths)), 10)
        target_h = max(int(np.median(heights)), 10)
        if force_w is not None and force_w > 0: target_w = int(force_w)
        if force_h is not None and force_h > 0: target_h = int(force_h)
        results = []
        for pi, label, pts in valid_panels:
            pts = pts.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            dst = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(img, M, (target_w, target_h), borderMode=cv2.BORDER_REPLICATE)
            ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok: return None, 'Encode failed for panel ' + label
            results.append((buf.tobytes(), label, pts.tolist(), None))
        meta = {'region': list(region), 'panel_ranges': panel_ranges,
                'n_panels': len(valid_panels), 'skipped': skipped,
                'img_w': W, 'img_h': H, 'target_w': target_w, 'target_h': target_h}
        return results, meta
    except Exception as e:
        return None, str(e)


def process_drone_single(img_bytes, corners_full, target_w=0, target_h=0):
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None, 'Decode failed'
        H, W = img.shape[:2]
        pts = np.array(corners_full, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        if target_w <= 0 or target_h <= 0:
            target_w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
            target_h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
        dst = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (target_w, target_h), borderMode=cv2.BORDER_REPLICATE)
        ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok: return None, 'Encode failed'
        return {'jpeg': buf.tobytes(), 'corners': pts.tolist()}, None
    except Exception as e:
        return None, str(e)


# ================================================================
# Flask routes — 静态文件
# ================================================================

@app.route('/')
def index():
    with open(HTML, 'r', encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html; charset=utf-8')

@app.route('/logo')
def logo():
    for name, mime in [('logo.png', 'image/png'), ('logo.ico', 'image/x-icon')]:
        path = os.path.join(BASE, name)
        if os.path.exists(path):
            resp = Response(open(path, 'rb').read(), mimetype=mime)
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
    return Response(b'', status=404)

@app.route('/restore_icon')
def restore_icon():
    path = os.path.join(BASE, 'restore_icon.png')
    if os.path.exists(path):
        resp = Response(open(path, 'rb').read(), mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    return Response(b'', status=404)

@app.route('/import_icon')
def import_icon():
    path = os.path.join(BASE, 'import_icon.png')
    if os.path.exists(path):
        resp = Response(open(path, 'rb').read(), mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    return Response(b'', status=404)

@app.route('/jszip.min.js')
def serve_jszip():
    path = os.path.join(BASE, 'jszip.min.js')
    if os.path.exists(path):
        resp = Response(open(path, 'rb').read(), mimetype='application/javascript')
        resp.headers['Cache-Control'] = 'public, max-age=86400'
        return resp
    return Response(b'', status=404)

@app.route('/favicon.png')
@app.route('/favicon.ico')
def favicon():
    path = os.path.join(BASE, 'favicon.png')
    if os.path.exists(path):
        resp = Response(open(path, 'rb').read(), mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    return Response(b'', status=404)

@app.route('/health')
def health():
    return jsonify({'ok': True, 'backend': True, 'frontend': os.path.exists(HTML), 'port': 15789})

@app.route('/info')
def info():
    dev = ''
    if USE_GPU:
        try: dev = cv2.ocl.Device.getDefault().name()
        except: dev = 'OpenCL'
    return jsonify({'gpu': USE_GPU, 'workers': WORKERS,
                    'cv': cv2.__version__, 'device': dev,
                    'corrections': 0, 'corrections_drone': 0})


# ================================================================
# Flask routes — 图像处理
# ================================================================

@app.route('/process', methods=['POST'])
def process():
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f: return jsonify({'ok': False, 'error': 'No image'})
        pad = float(request.form.get('pad', 0.005))
        pad_v = float(request.form.get('pad_v', pad))
        forced = json.loads(request.form['corners']) if request.form.get('corners') else None
        img_name = request.form.get('name', None)
        target_w = request.form.get('target_w', None)
        target_h = request.form.get('target_h', None)
        if target_w is not None: target_w = int(target_w)
        if target_h is not None: target_h = int(target_h)
        jpeg, corners, err = process_image(f.read(), pad, forced, img_name, pad_v, target_w, target_h)
        if err: return jsonify({'ok': False, 'error': err})
        return jsonify({'ok': True, 'jpeg_b64': base64.b64encode(jpeg).decode(),
                        'corners': corners, 'ms': round((time.time() - t0) * 1000)})
    except Exception as e:
        return jsonify({'ok': False, 'error': traceback.format_exc()})

@app.route('/batch', methods=['POST'])
def batch():
    t0 = time.time()
    fs = request.files.getlist('images')
    pad = float(request.form.get('pad', 0.005))
    pad_v = float(request.form.get('pad_v', pad))
    forced = json.loads(request.form['corners']) if request.form.get('corners') else None
    names_raw = request.form.get('names', None)
    names = json.loads(names_raw) if names_raw else [None] * len(fs)
    if len(names) != len(fs): names = [None] * len(fs)
    target_w = request.form.get('target_w', None)
    target_h = request.form.get('target_h', None)
    if target_w is not None: target_w = int(target_w)
    if target_h is not None: target_h = int(target_h)

    def proc(f, name):
        return process_image(f.read(), pad, forced, name, pad_v, target_w, target_h)

    futures = {pool.submit(proc, f, n): i for i, (f, n) in enumerate(zip(fs, names))}
    results = [None] * len(fs)
    for fut, idx in futures.items():
        jpeg, corners, err = fut.result()
        results[idx] = {'ok': False, 'error': err} if err else \
                       {'ok': True, 'jpeg_b64': base64.b64encode(jpeg).decode(), 'corners': corners}
    return jsonify({'results': results, 'ms': round((time.time() - t0) * 1000)})

@app.route('/process_drone', methods=['POST'])
def process_drone():
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f: return jsonify({'ok': False, 'error': 'No image'})
        pad = float(request.form.get('pad', 0.014))
        pad_v = float(request.form.get('pad_v', pad))
        force_w = request.form.get('target_w', None)
        force_h = request.form.get('target_h', None)
        if force_w is not None: force_w = int(force_w)
        if force_h is not None: force_h = int(force_h)
        result, meta_or_err = process_drone_image(f.read(), pad, pad_v, force_w, force_h)
        if isinstance(meta_or_err, str): return jsonify({'ok': False, 'error': meta_or_err})
        panels = []
        for jpeg_bytes, label, corners, roi_bounds in result:
            panels.append({'label': label, 'jpeg_b64': base64.b64encode(jpeg_bytes).decode(),
                           'corners': corners, 'roi': roi_bounds})
        return jsonify({'ok': True, 'panels': panels, 'meta': meta_or_err,
                        'ms': round((time.time() - t0) * 1000)})
    except Exception as e:
        return jsonify({'ok': False, 'error': traceback.format_exc()})

@app.route('/process_drone_single', methods=['POST'])
def process_drone_single_route():
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f: return jsonify({'ok': False, 'error': 'No image'})
        corners = json.loads(request.form['corners'])
        tw = int(request.form.get('target_w', 0))
        th = int(request.form.get('target_h', 0))
        result, err = process_drone_single(f.read(), corners, tw, th)
        if err: return jsonify({'ok': False, 'error': err})
        return jsonify({'ok': True, 'jpeg_b64': base64.b64encode(result['jpeg']).decode(),
                        'corners': result['corners'], 'ms': round((time.time() - t0) * 1000)})
    except Exception as e:
        return jsonify({'ok': False, 'error': traceback.format_exc()})


if __name__ == '__main__':
    gpu_str = ('GPU: ' + (cv2.ocl.Device.getDefault().name() if USE_GPU else '')) if USE_GPU else 'CPU mode'
    port = int(os.environ.get('EL_CROP_PORT', 15789))
    print('=' * 56)
    print('  华矩EL 裁剪工具  离线版')
    print('  ' + gpu_str)
    print('  线程数: %d   OpenCV: %s' % (WORKERS, cv2.__version__))
    print('  检测方法: Hough直线(主) + 亮度阈值(备)')
    print('  本机地址: http://127.0.0.1:%d' % port)
    print('  关闭此窗口退出')
    print('=' * 56)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
