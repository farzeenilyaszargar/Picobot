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
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

SOURCE = 'https://upload.wikimedia.org/wikipedia/commons/5/5d/Jellyfish_in_a_aquarium.webm'
SOURCE_PAGE = 'https://commons.wikimedia.org/wiki/File:Jellyfish_in_a_aquarium.webm'
ROOT = Path(__file__).resolve().parents[1]
FPS, DURATION, CELL_W, CELL_H, FONT_SIZE = 24, 24, 6, 9, 10
RAMP = ' .,:;i1tfLCG08@'
# Maximum subject extent, animation phase, reflection.
LAYOUTS = {
    'desktop': ((1920, 1080), [
        (230, 0/6, False), (180, 1/6, True),
        (205, 2/6, True), (170, 3/6, False),
        (130, 4/6, False), (155, 5/6, True),
    ]),
    'mobile': ((780, 1692), [
        (170, 0/6, False), (145, 1/6, True),
        (140, 2/6, True), (165, 3/6, False),
        (115, 4/6, False), (105, 5/6, True),
    ]),
}


def run(*args):
    subprocess.run(args, check=True)


def motion_position(time, index, phase, variant):
    """Wide, continuous figure-eight currents; alternate travel directions."""
    direction = 1 if index % 2 == 0 else -1
    angle = 2 * math.pi * (time + phase)
    radius = .34 if variant == 'mobile' else .39
    x = .5 + radius * math.sin(angle) + .025 * math.sin(3*angle + index*.6)
    y = .5 + .34 * math.sin(direction*2*angle + index*.75) + .025 * math.cos(3*angle)
    return x, y


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
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    atlas = []
    for char in RAMP:
        glyph = Image.new('L', (CELL_W, CELL_H), 255)
        ImageDraw.Draw(glyph).text((CELL_W/2, CELL_H/2), char, font=font, fill=0, anchor='mm')
        atlas.append(np.where(np.asarray(glyph) < 190, 0, 255).astype(np.uint8))
    # Ensure density increases at this medium character size.
    return np.stack(sorted(atlas, key=lambda glyph: np.count_nonzero(glyph == 0)))


def ascii_tile(gray, atlas):
    # Pixelate each subject before moving it. This keeps the characters intact
    # as they travel instead of re-sampling/shimmering against a fixed page grid.
    cols = math.ceil(gray.width / CELL_W)
    rows = math.ceil(gray.height / CELL_H)
    sampled = gray.resize((cols, rows), Image.Resampling.BOX)
    indices = np.minimum((np.asarray(sampled, dtype=np.float32)/255*(len(atlas)-1)).astype(int),len(atlas)-1)
    pixels = atlas[indices].transpose(0,2,1,3).reshape(rows*CELL_H,cols*CELL_W)
    return Image.fromarray(pixels)


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
              'ascii_cell': [CELL_W, CELL_H], 'font_size': FONT_SIZE,
              'blackpoint': 24, 'whitepoint': 164,
              'matte': 'largest connected bright subject; expanded 9 cells at 320x180',
              'subjects': 6, 'background': 'isolated subjects on clean white; no tank or particles',
              'loop': 'wide figure-eight currents, alternating directions; phase-offset body cycles; identical end frames',
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
            for extent, _, flip in placements:
                samples = []
                for subject in subjects:
                    scale = extent / max_extent
                    tile = subject.resize((max(1, round(subject.width*scale)), max(1, round(subject.height*scale))), Image.Resampling.LANCZOS)
                    if flip:
                        tile = tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    samples.append(ascii_tile(tile, atlas))
                tiles.append(samples)
            rendered = work / variant
            rendered.mkdir()
            first = last = None
            all_boxes = []
            for index in range(total):
                time = (index / (total - 1)) % 1
                canvas = Image.new('L', size, 255)
                boxes = []
                for n, (extent, phase, flip) in enumerate(placements):
                    cycle = (time + phase) % 1
                    source_index = round((1 - math.cos(2*math.pi*cycle))/2 * (len(subjects)-1))
                    tile = tiles[n][source_index]
                    x, y = motion_position(time, n, phase, variant)
                    px, py = round(x*width - tile.width/2), round(y*height - tile.height/2)
                    assert 0 <= px and 0 <= py and px+tile.width <= width and py+tile.height <= height
                    # Add isolated foregrounds without erasing another subject at crossings.
                    region = canvas.crop((px, py, px+tile.width, py+tile.height))
                    canvas.paste(ImageChops.darker(region, tile), (px, py))
                    boxes.append([px, py, tile.width, tile.height])
                frame = np.asarray(canvas)
                Image.fromarray(frame).save(rendered / f'{index:04d}.png')
                first = frame if first is None else first
                last = frame
                all_boxes.append(boxes)
                if index == 0:
                    Image.fromarray(frame).save(args.output / f'drifters-ascii-{variant}-poster.png', optimize=True)
                    if variant == 'desktop':
                        sheet = Image.new('L', (1280,720),255)
                        contrasted = subjects[0]
                        pixelated = contrasted.resize((math.ceil(contrasted.width/CELL_W),math.ceil(contrasted.height/CELL_H)),Image.Resampling.BOX).resize(contrasted.size,Image.Resampling.NEAREST)
                        for n, image in enumerate([originals[0], contrasted, pixelated, Image.fromarray(frame)]):
                            sheet.paste(image.resize((640,360)),((n%2)*640,(n//2)*360))
                        sheet.save(preview / 'drifters-processing.png')
            common = ['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(FPS),'-i',str(rendered/'%04d.png'),'-an','-map_metadata','-1']
            run(*common,'-c:v','libvpx-vp9','-crf','45','-b:v','0','-row-mt','1','-pix_fmt','yuv420p',str(args.output/f'drifters-ascii-{variant}.webm'))
            run(*common,'-c:v','libx264','-preset','slow','-crf','32','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/f'drifters-ascii-{variant}.mp4'))
            report['layouts'][variant] = {'size':size,'ascii_grid':[cols,rows],
                'subject_max_extents':[p[0] for p in placements],
                'first_frame_boxes':all_boxes[0], 'master_first_last_identical':bool(np.array_equal(first,last)),
                'first_frame_ink_coverage':float(np.count_nonzero(first==0)/first.size)}
            centers = np.asarray([[[b[0]+b[2]/2, b[1]+b[3]/2] for b in boxes] for boxes in all_boxes])
            report['layouts'][variant]['motion'] = {
                'x_span_pixels': np.ptp(centers[:,:,0],axis=0).round(1).tolist(),
                'y_span_pixels': np.ptp(centers[:,:,1],axis=0).round(1).tolist(),
                'path_length_pixels': np.linalg.norm(np.diff(centers,axis=0),axis=2).sum(axis=0).round(1).tolist(),
                'keyframe_boxes': [all_boxes[index] for index in [0,total//4,total//2,3*total//4]],
            }
            samples = Image.new('L',(1280,720),255)
            for n, index in enumerate([0,total//4,total//2,3*total//4]):
                samples.paste(Image.open(rendered/f'{index:04d}.png').resize((640,360)),((n%2)*640,(n//2)*360))
            samples.save(preview/f'drifters-{variant}-samples.png')
        report['outputs'] = {p.name:p.stat().st_size for p in args.output.glob('drifters-ascii-*') if p.suffix in {'.webm','.mp4','.png'}}
        (args.output/'drifters-ascii.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
