from flask import Flask, request, send_file, jsonify
from remover import find_watermark, process  # теперь поддерживает SVG
from io import BytesIO
import requests
import os
import hashlib
import cv2
import numpy as np

app = Flask(__name__)

def download(url):
    """Скачиваем файл БЕЗ кэширования"""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

@app.post("/remove")
def remove():
    # IMAGE — обязательный
    img_url = request.form.get("image")
    if not img_url:
        return {"error": "image URL is required"}, 400
    image_bytes = download(img_url)
    if not image_bytes:
        return {"error": "cannot download image"}, 400

    # WOTEMARK — обязательный, может быть несколько через запятую
    wm_urls = request.form.get("wotemark")
    if not wm_urls:
        return {"error": "wotemark URL(s) required"}, 400

    templates = []
    for url in wm_urls.split(","):
        url = url.strip()
        if url:
            data = download(url)
            if not data:
                return {"error": f"cannot download wotemark: {url}"}, 400
            templates.append(data)

    # TO-WOTEMARK — обязательный, может быть несколько SVG через запятую
    new_wm_urls = request.form.get("to-wotemark")
    if not new_wm_urls:
        return {"error": "to-wotemark URL(s) required"}, 400

    svg_watermarks = []
    for url in new_wm_urls.split(","):
        url = url.strip()
        if url:
            data = download(url)
            if not data:
                return {"error": f"cannot download to-wotemark: {url}"}, 400
            svg_watermarks.append(data)

    # PROCESS — теперь принимает список SVG
    output_bytes, fmt, width, height = process(
        image_bytes=image_bytes,
        templates=templates,
        svg_watermarks=svg_watermarks
    )

    return send_file(
        BytesIO(output_bytes),
        mimetype=f"image/{fmt.lower()}",
        as_attachment=True,
        download_name=f"output.{fmt}"
    )

@app.post("/size")
def watermark_size():
    # Получаем URL исходного изображения
    img_url = request.form.get("image")
    if not img_url:
        return {"error": "image URL is required"}, 400

    image_bytes = download(img_url)
    if not image_bytes:
        return {"error": "cannot download image"}, 400

    # Получаем URL шаблона водяного знака
    tmpl_urls = request.form.get("template")
    if not tmpl_urls:
        return {"error": "template URL(s) required"}, 400

    templates = []
    for url in tmpl_urls.split(","):
        url = url.strip()
        if url:
            data = download(url)
            if not data:
                return {"error": f"cannot download template: {url}"}, 400
            templates.append(data)

    # Декодируем изображение
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "failed to decode image"}, 400

    results = []
    for t_bytes in templates:
        tmpl = cv2.imdecode(np.frombuffer(t_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if tmpl is None:
            continue

        bbox = find_watermark(image, tmpl)
        if bbox:
            x, y, w, h, angle = bbox
            results.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "angle": angle
            })

    if not results:
        return {"error": "watermark not found"}, 404

    return jsonify(results)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8091)
