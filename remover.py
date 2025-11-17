import cv2
import numpy as np
import imghdr

def remove_watermark(image, x, y, w, h, method='telea', radius_ratio=0.3):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    inpaint_radius = max(int(max(w, h) * radius_ratio), 10)
    restored = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return restored

def find_watermark(image, template, threshold=0.6):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_img, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        return None
    x, y = max_loc
    h, w = template.shape[:2]
    return x, y, w, h

def overlay_new(image, x, y, w, h, new_wm):
    if new_wm.shape[2] == 4:
        alpha = new_wm[:, :, 3] / 255.0
        for c in range(3):
            new_wm[:, :, c] = (new_wm[:, :, c] * alpha).astype(np.uint8)
        new_wm = new_wm[:, :, :3]

    new_wm = cv2.resize(new_wm, (w, h), interpolation=cv2.INTER_AREA)
    alpha = 0.9
    for c in range(3):
        image[y:y+h, x:x+w, c] = (
            alpha * new_wm[:, :, c] + (1 - alpha) * image[y:y+h, x:x+w, c]
        ).astype(np.uint8)
    return image

def process(image_bytes, templates, new_watermark):
    fmt = imghdr.what(None, h=image_bytes)
    if fmt is None:
        fmt = "jpg"

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode input image")

    new_wm = cv2.imdecode(np.frombuffer(new_watermark, np.uint8), cv2.IMREAD_UNCHANGED)
    if new_wm is None:
        raise ValueError("Failed to decode new watermark")

    match = None
    for t_bytes in templates:
        tmpl = cv2.imdecode(np.frombuffer(t_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if tmpl is None:
            continue
        m = find_watermark(image, tmpl)
        if m:
            match = m
            break

    if match:
        x, y, w, h = match
        image = remove_watermark(image, x, y, w, h)
        image = overlay_new(image, x, y, w, h, new_wm)
    else:
        # центрируем новый watermark
        h_i, w_i = image.shape[:2]
        h_w, w_w = new_wm.shape[:2]
        new_w = w_i // 4
        new_h = int(h_w * (new_w / w_w))
        nx = (w_i - new_w) // 2
        ny = (h_i - new_h) // 2
        resized = cv2.resize(new_wm, (new_w, new_h))
        image = overlay_new(image, nx, ny, new_w, new_h, resized)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98] if fmt.lower() in ["jpg", "jpeg"] else []
    ok, out = cv2.imencode(f".{fmt}", image, encode_param)
    if not ok:
        raise ValueError("Failed to encode output image")

    height, width = image.shape[:2]
    return out.tobytes(), fmt, width, height
