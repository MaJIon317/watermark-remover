from flask import Flask, request, send_file, jsonify
from remover import process
from io import BytesIO
import requests
import os
import hashlib

app = Flask(__name__)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def url_to_cache_path(url):
    """Преобразует URL в путь кэш-файла по SHA256"""
    h = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.bin")

def download_or_cache(url):
    """Скачиваем файл с кэшированием на диск"""
    cache_path = url_to_cache_path(url)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.content
        with open(cache_path, "wb") as f:
            f.write(data)
        return data
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

@app.post("/remove")
def remove():
    # IMAGE — обязательный
    img_url = request.form.get("image")
    if not img_url:
        return {"error": "image URL is required"}, 400
    image_bytes = download_or_cache(img_url)
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
            data = download_or_cache(url)
            if not data:
                return {"error": f"cannot download wotemark: {url}"}, 400
            templates.append(data)

    # TO-WOTEMARK — обязательный
    new_wm_url = request.form.get("to-wotemark")
    if not new_wm_url:
        return {"error": "to-wotemark URL required"}, 400

    new_wm_bytes = download_or_cache(new_wm_url)
    if not new_wm_bytes:
        return {"error": "cannot download to-wotemark"}, 400

    # PROCESS
    output_bytes, fmt, width, height = process(
        image_bytes=image_bytes,
        templates=templates,
        new_watermark=new_wm_bytes
    )

    return send_file(
        BytesIO(output_bytes),
        mimetype=f"image/{fmt.lower()}",
        as_attachment=True,
        download_name=f"output.{fmt}"
    )

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8091)
