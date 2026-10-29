#!/usr/bin/env python3
"""Render six isolated, finely detailed ASCII drifters from credited aquarium footage.

Requires Pillow, NumPy and FFmpeg. Source/intermediates are never served.
The adapted media is CC BY-SA 3.0; see web/dist/media/CREDITS.md.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SOURCE = 'https://upload.wikimedia.org/wikipedia/commons/5/5d/Jellyfish_in_a_aquarium.webm'
SOURCE_PAGE = 'https://commons.wikimedia.org/wiki/File:Jellyfish_in_a_aquarium.webm'
ROOT = Path(__file__).resolve().parents[1]
FPS, DURATION, CELL_W, CELL_H = 15, 20, 4, 6
RAMP = ' .,:;i1tfLCG08@'
# x, y, maximum extent in pixels, phase, reflection. Avoid the central chat area.
LAYOUTS = {
    'desktop': ((1920, 1080), [
        (.12, .29, 230, .00, False), (.14, .73, 180, .38, True),
        (.87, .22, 205, .17, True), (.90, .64, 170, .63, False),
        (.34, .86, 130, .78, False), (.72, .87, 155, .48, True),
    ]),
    'mobile': ((780, 1680), [
        (.17, .18, 170, .00, False), (.82, .24, 145, .22, True),
        (.12, .74, 140, .38, True), (.85, .77, 165, .63, False),
        (.32, .89, 115, .78, False), (.70, .11, 105, .48, True),
    ]),
}


def run(*args):
    subprocess.run(args, check=True)


def isolate(gray):
    """Track the largest connected jelly, discarding tank reflections and specks."""
    small = np.asarray(gray.resize((320, 180), Image.Resampling.BOX))
    bright = small > 47
    seen = np.zeros_like(bright)
    best = []
    for y, x in zip(*np.nonzero(bright)):
        if seen[y, x]:
            continue
        queue = deque([(y, x)])
        seen[y, x] = True
        component = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                if 0 <= ny < 180 and 0 <= nx < 320 and bright[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(component) > len(best):
            best = component
    if len(best) < 100:
        raise RuntimeError('No stable subject found in source frame')
    mask = np.zeros_like(small)
    ys, xs = zip(*best)
    mask[ys, xs] = 255
    # Expand the subject matte just enough to retain the delicate outer tissue.
    matte = Image.fromarray(mask).filter(ImageFilter.MaxFilter(9)).resize(gray.size, Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.float32)
    # Greyscale -> full-range contrast -> foreground matte. All surrounding water is zero.
    contrast = np.clip((values - 24) * (255 / 140), 0, 255)
    contrast *= np.asarray(matte, dtype=np.float32) / 255
    contrast[contrast < 18] = 0
    bounds = matte.getbbox()
    return Image.fromarray(contrast.astype(np.uint8)).crop(bounds)


def glyph_atlas(font_path):
    font = ImageFont.truetype(str(font_path), 7)
    atlas = []
    for char in RAMP:
        glyph = Image.new('L', (CELL_W, CELL_H), 255)
        ImageDraw.Draw(glyph).text((CELL_W/2, CELL_H/2), char, font=font, fill=0, anchor='mm')
        atlas.append(np.where(np.asarray(glyph) < 190, 0, 255).astype(np.uint8))
    # Ensure density increases even at this very fine character size.
    return np.stack(sorted(atlas, key=lambda glyph: np.count_nonzero(glyph == 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'web/dist/media')
    parser.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/Menlo.ttc'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preview = ROOT / 'previews'
    preview.mkdir(exist_ok=True)
    atlas = glyph_atlas(args.font)
    report = {'source_page': SOURCE_PAGE, 'source_download': SOURCE,
              'credit': 'Amada44 / Wikimedia Commons', 'license': 'CC BY-SA 3.0',
              'license_url': 'https://creativecommons.org/licenses/by-sa/3.0/',
              'segment_seconds': [9, 17], 'duration_seconds': DURATION, 'fps': FPS,
              'ascii_cell': [CELL_W, CELL_H], 'font_size': 7,
              'blackpoint': 24, 'whitepoint': 164,
              'matte': 'largest connected bright subject; expanded 9 cells at 320x180',
              'subjects': 6, 'background': 'isolated subjects on clean white; no tank or particles',
              'loop': 'independently phase-offset, cosine-eased forward/reverse; identical end frames',
              'layouts': {}}
    with tempfile.TemporaryDirectory(prefix='picoboi-drifters-') as temp:
        work = Path(temp)
        source = args.source or work / 'original.webm'
        if not args.source:
            urllib.request.urlretrieve(SOURCE, source)
        report['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        report['source_bytes'] = source.stat().st_size
        decoded = work / 'decoded'
        decoded.mkdir()
        run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', '9', '-t', '8', '-i', str(source),
            '-vf', f'fps={FPS},scale=1280:720', '-an', str(decoded / '%04d.png'))
        originals = [Image.open(path).convert('L') for path in sorted(decoded.glob('*.png'))]
        subjects = [isolate(frame) for frame in originals]
        max_extent = max(max(frame.size) for frame in subjects)
        total = FPS * DURATION
        for variant, (size, placements) in LAYOUTS.items():
            width, height = size
            cols, rows = width // CELL_W, height // CELL_H
            # Preserve the same scale across frames; contraction remains real motion.
            tiles = []
            for _, _, extent, _, flip in placements:
                samples = []
                for subject in subjects:
                    scale = extent / max_extent
                    tile = subject.resize((max(1, round(subject.width*scale)), max(1, round(subject.height*scale))), Image.Resampling.LANCZOS)
                    if flip:
                        tile = tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    samples.append(tile)
                tiles.append(samples)
            rendered = work / variant
            rendered.mkdir()
            first = last = None
            all_boxes = []
            for index in range(total):
                time = index / (total - 1)
                canvas = Image.new('L', size, 0)
                boxes = []
                for n, (x, y, extent, phase, flip) in enumerate(placements):
                    cycle = (time + phase) % 1
                    source_index = round((1 - math.cos(2*math.pi*cycle))/2 * (len(subjects)-1))
                    tile = tiles[n][source_index]
                    # Small drifting paths, returning exactly to their start.
                    dx = math.sin(2*math.pi*(time+phase)) * width*.007
                    dy = math.cos(2*math.pi*(time+phase)) * height*.012
                    px, py = round(x*width + dx - tile.width/2), round(y*height + dy - tile.height/2)
                    canvas.paste(tile, (px, py))
                    boxes.append([px, py, tile.width, tile.height])
                # Fine pixelation is a real sampling step before glyph mapping.
                pixelated = canvas.resize((cols, rows), Image.Resampling.BOX)
                indices = np.minimum((np.asarray(pixelated, dtype=np.float32)/255*(len(atlas)-1)).astype(int),len(atlas)-1)
                frame = atlas[indices].transpose(0,2,1,3).reshape(height,width)
                Image.fromarray(frame).save(rendered / f'{index:04d}.png')
                first = frame if first is None else first
                last = frame
                all_boxes.append(boxes)
                if index == 0:
                    Image.fromarray(frame).save(args.output / f'drifters-ascii-{variant}-poster.png', optimize=True)
                    if variant == 'desktop':
                        sheet = Image.new('L', (1280,720),255)
                        for n, image in enumerate([originals[0], canvas, pixelated.resize(size,Image.Resampling.NEAREST), Image.fromarray(frame)]):
                            sheet.paste(image.resize((640,360)),((n%2)*640,(n//2)*360))
                        sheet.save(preview / 'drifters-processing.png')
            common = ['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(FPS),'-i',str(rendered/'%04d.png'),'-an','-map_metadata','-1']
            run(*common,'-c:v','libvpx-vp9','-crf','45','-b:v','0','-row-mt','1','-pix_fmt','yuv420p',str(args.output/f'drifters-ascii-{variant}.webm'))
            run(*common,'-c:v','libx264','-preset','slow','-crf','32','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/f'drifters-ascii-{variant}.mp4'))
            report['layouts'][variant] = {'size':size,'ascii_grid':[cols,rows],
                'subject_max_extents':[p[2] for p in placements],
                'first_frame_boxes':all_boxes[0], 'master_first_last_identical':bool(np.array_equal(first,last)),
                'first_frame_ink_coverage':float(np.count_nonzero(first==0)/first.size)}
            samples = Image.new('L',(1280,720),255)
            for n, index in enumerate([0,75,150,225]):
                samples.paste(Image.open(rendered/f'{index:04d}.png').resize((640,360)),((n%2)*640,(n//2)*360))
            samples.save(preview/f'drifters-{variant}-samples.png')
        report['outputs'] = {p.name:p.stat().st_size for p in args.output.glob('drifters-ascii-*') if p.suffix in {'.webm','.mp4','.png'}}
        (args.output/'drifters-ascii.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
