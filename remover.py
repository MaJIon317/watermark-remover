import cv2
import numpy as np
import imghdr
import cairosvg

# -------------------------
# Utilities
# -------------------------
def _to_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

# -------------------------
# Inpainting
# -------------------------
def remove_watermark(image, x, y, w, h, method='telea', radius_ratio=0.3):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    inpaint_flags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    inpaint_radius = max(int(max(w,h) * radius_ratio), 3)
    restored = cv2.inpaint(image, mask, inpaint_radius, inpaint_flags)
    alpha = 0.95
    image[y:y+h, x:x+w] = cv2.addWeighted(restored[y:y+h, x:x+w], alpha, image[y:y+h, x:x+w], 1-alpha, 0)
    return image

# -------------------------
# Template matching
# -------------------------
def find_watermark(image, template, scales=(0.8,0.9,1.0,1.1,1.2), threshold=0.6, rotations=(0,90,180,270)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if template.shape[2] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    tmpl_gray_orig = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    for angle in rotations:
        tmpl_rot = _rotate_image(tmpl_gray_orig, angle)
        for scale in scales:
            nw, nh = max(3,int(tmpl_rot.shape[1]*scale)), max(3,int(tmpl_rot.shape[0]*scale))
            if nw > img_gray.shape[1] or nh > img_gray.shape[0]:
                continue
            tmpl_s = cv2.resize(tmpl_rot, (nw, nh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(img_gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                x, y = max_loc
                return (x, y, nw, nh, angle)
    return None

# -------------------------
# Overlay new watermark
# -------------------------
def overlay_new(image, x, y, wm):
    if wm.shape[2] == 3:
        alpha = np.ones((wm.shape[0], wm.shape[1], 1), dtype=np.uint8) * 255
        wm = np.concatenate([wm, alpha], axis=2)

    h, w = wm.shape[:2]
    y1 = min(image.shape[0], y + h)
    x1 = min(image.shape[1], x + w)
    wm = wm[:y1 - y, :x1 - x]

    overlay = wm[:, :, :3]
    alpha = wm[:, :, 3] / 255.0

    region = image[y:y1, x:x1].astype(np.float32)
    wm_float = overlay.astype(np.float32)
    for c in range(3):
        region[:, :, c] = region[:, :, c] * (1 - alpha) + wm_float[:, :, c] * alpha
    image[y:y1, x:x1] = region.astype(np.uint8)
    return image

def overlay_new_svg(image, x, y, w, h, wm_list, angle=0, scale_up=1.5):
    target_ratio = w / h
    chosen = wm_list[0]
    chosen_is_svg = chosen.strip().startswith(b"<svg") or b"<svg" in chosen

    min_diff = 999999
    for wm_bytes in wm_list:
        is_svg = wm_bytes.strip().startswith(b"<svg") or b"<svg" in wm_bytes
        if is_svg:
            try:
                s = wm_bytes.decode("utf-8")
                import re
                ww = re.search(r'width="(\d+)', s)
                hh = re.search(r'height="(\d+)', s)
                if ww and hh:
                    wr, hr = int(ww.group(1)), int(hh.group(1))
                    diff = abs((wr / hr) - target_ratio)
                    if diff < min_diff:
                        min_diff = diff
                        chosen = wm_bytes
                        chosen_is_svg = True
            except:
                pass
        else:
            img = cv2.imdecode(np.frombuffer(wm_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h0, w0 = img.shape[:2]
                diff = abs((w0 / h0) - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    chosen = wm_bytes
                    chosen_is_svg = False

    # --- Декодируем ---
    if chosen_is_svg:
        png_bytes = cairosvg.svg2png(bytestring=chosen)
        wm = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        wm = cv2.imdecode(np.frombuffer(chosen, np.uint8), cv2.IMREAD_UNCHANGED)

    if wm is None:
        raise ValueError("Failed to decode watermark")

    # --- Поворот ---
    wm = _rotate_image(wm, angle)

    # --- Масштабирование после поворота ---
    h_wm, w_wm = wm.shape[:2]
    scale_x = (w * scale_up) / w_wm
    scale_y = (h * scale_up) / h_wm
    wm = cv2.resize(wm, (int(w_wm*scale_x), int(h_wm*scale_y)), interpolation=cv2.INTER_AREA)

    # --- Центрирование ---
    center_x = x + w // 2
    center_y = y + h // 2
    new_h, new_w = wm.shape[:2]
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)

    return overlay_new(image, new_x, new_y, wm)

# -------------------------
# Main process
# -------------------------
def process(image_bytes, templates, svg_watermarks):
    fmt = imghdr.what(None, h=image_bytes) or "jpg"
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode input image")

    match = None
    for t_bytes in templates:
        tmpl = cv2.imdecode(np.frombuffer(t_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if tmpl is None:
            continue
        bbox = find_watermark(image, tmpl)
        if bbox:
            match = bbox
            break

    if not match:
        ok, out = cv2.imencode(f".{fmt}", image)
        return out.tobytes(), fmt, *image.shape[1::-1]

    x, y, w, h, angle = match
    image_cleaned = remove_watermark(image.copy(), x, y, w, h)
    final = overlay_new_svg(image_cleaned, x, y, w, h, svg_watermarks, angle=angle)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98] if fmt.lower() in ["jpg","jpeg"] else []
    ok, out = cv2.imencode(f".{fmt}", final, encode_param)
    return out.tobytes(), fmt, final.shape[1], final.shape[0]
