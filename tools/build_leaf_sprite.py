"""Build the embedded leaf-image data for docs/deep_learning.html.

The interactive Week 3 practical trains a real CNN in the browser, so it needs
real tomato leaf photos. Rather than fetching hundreds of files from GitHub at
page load, we downsample them to 64x64 (the same size the notebook uses) and
pack them into a single JPEG sprite sheet that gets base64-embedded into the
HTML. The page slices the sheet back apart on a canvas.

Run once:

    python tools/build_leaf_sprite.py

Writes docs/leaf_data.js, which both docs/deep_learning.html and
docs/deep_learning_answers.html pull in with a plain <script src>. It is kept
as a separate file rather than inlined because it is ~870 KB and two pages
need it; it is a local relative file, so the practical still works offline.

Source: https://github.com/gabrieldgf4/PlantVillage-Dataset (same dataset the
notebook clones).
"""

import base64
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO = "gabrieldgf4/PlantVillage-Dataset"
REF = "master"
TILE = 64  # matches tf.image.resize(img, [64, 64]) in the notebook

HEALTHY_DIR = "Tomato___healthy"
INFECTED_DIRS = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
]

N_HEALTHY = 250
N_PER_INFECTED = 50  # 9 dirs -> 450 infected
JPEG_QUALITY = 82

OUT_JS = Path(__file__).resolve().parent.parent / "docs" / "leaf_data.js"


def list_dir(directory: str) -> list[str]:
    """Return download URLs for the .JPG files in one dataset directory."""
    url = (
        f"https://api.github.com/repos/{REPO}/contents/"
        f"{urllib.parse.quote(directory)}?ref={REF}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "eduxr-sprite-builder"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        entries = json.load(resp)
    urls = [
        e["download_url"]
        for e in entries
        if e["type"] == "file" and e["name"].upper().endswith(".JPG")
    ]
    # Sort so the selection is reproducible if this script is ever re-run.
    return sorted(urls)


def pick_evenly(items: list[str], n: int) -> list[str]:
    """Take n items spread evenly across the list, deterministically."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _mirror(url: str) -> str:
    """jsDelivr mirror of a raw.githubusercontent URL, used as a fallback."""
    prefix = f"https://raw.githubusercontent.com/{REPO}/{REF}/"
    if url.startswith(prefix):
        return f"https://cdn.jsdelivr.net/gh/{REPO}@{REF}/" + url[len(prefix):]
    return url


def fetch_tile(url: str) -> Image.Image:
    """Download one image, with backoff -- raw.githubusercontent rate-limits."""
    last = None
    for attempt in range(6):
        target = url if attempt % 2 == 0 else _mirror(url)
        try:
            req = urllib.request.Request(
                target, headers={"User-Agent": "eduxr-sprite-builder"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return img.resize((TILE, TILE), Image.BILINEAR)
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"failed to fetch {url}") from last


def main() -> None:
    wanted: list[tuple[str, str, int]] = []  # (url, class name, label)

    print(f"listing {HEALTHY_DIR} ...", flush=True)
    for url in pick_evenly(list_dir(HEALTHY_DIR), N_HEALTHY):
        wanted.append((url, "healthy", 0))

    for directory in INFECTED_DIRS:
        print(f"listing {directory} ...", flush=True)
        name = directory.replace("Tomato___", "").replace("_", " ")
        for url in pick_evenly(list_dir(directory), N_PER_INFECTED):
            wanted.append((url, name, 1))

    print(f"downloading {len(wanted)} images ...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        tiles = list(pool.map(fetch_tile, [w[0] for w in wanted]))

    # Square-ish grid. Tiles are 64px so every tile edge lands on a JPEG MCU
    # boundary (16px with 4:2:0 chroma subsampling) -- no bleed between tiles.
    n = len(tiles)
    cols = int(n**0.5 + 0.999)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE, rows * TILE), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * TILE, (i // cols) * TILE))

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"sheet {cols}x{rows} tiles, {sheet.size[0]}x{sheet.size[1]}px, "
          f"{len(buf.getvalue())/1024:.0f} KB jpeg, {len(b64)/1024:.0f} KB base64")
    if len(b64) > 900 * 1024:
        print("WARNING: base64 payload is larger than the ~800 KB budget.", file=sys.stderr)

    classes = sorted({c for _, c, _ in wanted})
    OUT_JS.write_text(
        "/* Generated by tools/build_leaf_sprite.py -- do not edit by hand.\n"
        f"   {n} tomato leaf images from PlantVillage, {TILE}x{TILE}, packed\n"
        f"   into a {cols}x{rows} sprite sheet. Label 0 = healthy, 1 = infected. */\n"
        f"const LEAF_COLS = {cols};\n"
        f"const LEAF_TILE = {TILE};\n"
        f"const LEAF_N = {n};\n"
        f"const LEAF_CLASS_NAMES = {json.dumps(classes)};\n"
        f"const LEAF_LABELS = {json.dumps([w[2] for w in wanted])};\n"
        f"const LEAF_CLASS = {json.dumps([classes.index(w[1]) for w in wanted])};\n"
        f'const LEAF_SPRITE = "data:image/jpeg;base64,{b64}";\n',
        encoding="utf-8",
    )
    print(f"wrote {OUT_JS} ({OUT_JS.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
