# Fine ASCII drifters

Original: **Jellyfish in a aquarium** (August 2011), by **Amada44**.

- Source: https://commons.wikimedia.org/wiki/File:Jellyfish_in_a_aquarium.webm
- Author: https://commons.wikimedia.org/wiki/User:Amada44
- Original file: https://upload.wikimedia.org/wikipedia/commons/5/5d/Jellyfish_in_a_aquarium.webm
- License: **Creative Commons Attribution-ShareAlike 3.0 Unported**, https://creativecommons.org/licenses/by-sa/3.0/

The adapted MP4/WebM videos and PNG posters in this directory are distributed under **CC BY-SA 3.0**. Attribution and license links are also available on the website's About page. No endorsement by the original author is implied.

Modifications: excerpted seconds 9–17; converted to greyscale; clipped input levels 24–164 to full 0–255 contrast; isolated the bright jellyfish using a tracked foreground matte; removed tank reflections and background particles; composed six scaled, reflected, independently timed subjects; sampled fine 4 × 6 pixel cells; rendered binary black/white 7-pixel monospace ASCII glyphs. The empty field is pure white, matching the page, with no visible scene background. The repeated subjects come from the same real footage, with distinct phases and gentle drift paths.

Each subject uses a cosine-eased forward/reverse cycle. All first and last master frames match, producing a seamless 20-second loop. Audio is removed. Desktop and mobile have separate compositions, so the subjects are kept small without cropping or enlargement.

| Asset | Format | Dimensions | Frame rate | Bytes |
| --- | --- | --- | --- | --- |
| drifters-ascii-desktop.mp4 | H.264, preferred | 1920 × 1080 | 15 fps | 2,344,299 |
| drifters-ascii-desktop.webm | VP9 fallback | 1920 × 1080 | 15 fps | 3,233,204 |
| drifters-ascii-mobile.mp4 | H.264, preferred | 780 × 1680 | 15 fps | 1,558,802 |
| drifters-ascii-mobile.webm | VP9 fallback | 780 × 1680 | 15 fps | 2,206,423 |
| drifters-ascii-desktop-poster.png | Still fallback | 1920 × 1080 | — | 17,595 |
| drifters-ascii-mobile-poster.png | Still fallback | 780 × 1680 | — | 12,775 |

The browser downloads one suitable video; reduced-motion/data-saver users receive only a still. The original source is 15,176,025 bytes and is not shipped. `drifters-ascii.json` records its checksum and rendering parameters.

Rebuild with Python (Pillow and NumPy), FFmpeg, and a monospace font:

```sh
python3 tools/render_ascii_background.py --source /path/to/Jellyfish_in_a_aquarium.webm
```

Omit `--source` to download the original into temporary storage. Systems without Menlo should pass `--font /path/to/monospace.ttf`. Processing and loop sample sheets are written to ignored `previews/`; original footage and intermediate frames are not part of the website.
