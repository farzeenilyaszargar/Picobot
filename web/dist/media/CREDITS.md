# ASCII jellyfish background

Original footage: **NOAA Ocean Exploration, Gulf of Mexico 2018**.

- Source page: https://oceanexplorer.noaa.gov/multimedia/video-shorts-ex1803-jelly/
- Source download: https://oceanexplorer.noaa.gov/wp-content/uploads/2021/02/ex1803-jelly.mp4
- NOAA reuse guidance: https://oceanexplorer.noaa.gov/faqs/
- Subject: Poralia sp. jellyfish, filmed at a depth of 1,054 meters.

NOAA Ocean Exploration states that its material is public domain unless otherwise noted, and requests the credit listed with the video. This clip's source credits NOAA Ocean Exploration, Gulf of Mexico 2018, with no third-party copyright notice. No NOAA endorsement is implied.

The background is an adaptation: caption-free seconds 7–15, greyscale conversion, clipped contrast from input levels 12–80 to 0–255, 160 × 60 pixel sampling, then black-and-white monospace ASCII glyphs. A cosine-eased forward/reverse sequence forms a 16-second loop with matching first and last master frames. Audio is removed.

Files:

- `jellyfish-ascii.webm`: VP9, 1280 × 720, 12 fps, about 559 KiB.
- `jellyfish-ascii.mp4`: H.264 fallback, same size/frame rate, about 895 KiB, fast-start metadata.
- `jellyfish-ascii-poster.png`: 6.2 KiB static fallback.
- `jellyfish-ascii.json`: source checksum, processing parameters, and output sizes.

Rebuild with Python (Pillow and NumPy) and FFmpeg:

```sh
python3 tools/render_ascii_background.py --source /path/to/ex1803-jelly.mp4
```

Omit `--source` to download the credited original into a temporary directory. On systems without Menlo, pass `--font /path/to/a/monospace-font.ttf`. Intermediate processing and loop contact sheets are written to `previews/`, outside the served website. The original source and decoded frames are not shipped.
