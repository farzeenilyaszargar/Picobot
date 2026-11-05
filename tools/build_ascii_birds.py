#!/usr/bin/env python3
"""Build a tiny eight-frame SVG sprite of original, hand-shaped ASCII gulls.

Pillow draws an intermediate silhouette; the shipped asset is native SVG text.
No source photo, video, remote fonts, or animation runtime is required.
"""
from pathlib import Path
import math
from html import escape
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
W, H, COLS, ROWS = 192, 120, 48, 24
RAMP = '.:;+xX#@'
frames = []
for frame in range(8):
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)
    flap = math.cos(frame * math.pi / 4)
    # A small gull viewed from below, banking forward: a tapered body and
    # two swept wings whose long tips describe a smooth up/down wingbeat.
    tip = 55 - 36 * flap
    draw.polygon([(91,57),(76,46),(49,tip+1),(14,tip-6),
                  (30,tip+7),(51,tip+17),(77,66),(93,65)], fill=255)
    draw.polygon([(98,56),(112,46),(137,tip+7),(174,tip+2),
                  (157,tip+15),(139,tip+22),(115,67),(98,65)], fill=255)
    draw.ellipse((88,46,102,79), fill=255)
    draw.ellipse((92,38,104,51), fill=255)
    draw.polygon([(102,42),(113,46),(102,48)],fill=255)
    draw.polygon([(91,74),(86,91),(96,84),(102,89),(99,74)],fill=255)
    sample = mask.resize((COLS,ROWS),Image.Resampling.BOX)
    lines = []
    for y in range(ROWS):
        row = ''
        for x in range(COLS):
            v = sample.getpixel((x,y))
            row += RAMP[(x*7+y*3+frame)%len(RAMP)] if v > 100 else ' '
        if row.strip():
            positions = ' '.join(str(x*4) for x,char in enumerate(row) if char != ' ')
            letters = ''.join(char for char in row if char != ' ')
            lines.append(f'<text x="{positions}" y="{y*5+4}">{escape(letters)}</text>')
    frames.append(f'<g transform="translate({frame*W} 0)">' + ''.join(lines) + '</g>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W*8}" height="{H}" viewBox="0 0 {W*8} {H}">
<g fill="#fff6de" font-family="monospace" font-size="5.5" letter-spacing=".69" xml:space="preserve">{''.join(frames)}</g></svg>'''
path=ROOT/'web/dist/assets/ascii-gulls.svg'
path.write_text(svg)
print(path, path.stat().st_size, 'bytes; 8 frames')
