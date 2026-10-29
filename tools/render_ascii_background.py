#!/usr/bin/env python3
"""Build PicoBoi's ASCII jellyfish loop from the credited NOAA source.

Requires Python with Pillow/numpy, plus ffmpeg/ffprobe on PATH.
The original download and intermediate frames stay outside the website.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOURCE = 'https://oceanexplorer.noaa.gov/wp-content/uploads/2021/02/ex1803-jelly.mp4'
SOURCE_PAGE = 'https://oceanexplorer.noaa.gov/multimedia/video-shorts-ex1803-jelly/'
ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, COLS, ROWS, FPS = 1280, 720, 160, 60, 12
RAMP = ' .,:;i1tfLCG08@'


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Existing original NOAA MP4; otherwise downloaded to a temporary directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'web/dist/media')
    parser.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/Menlo.ttc'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='picoboi-ascii-') as temp:
        work = Path(temp)
        source = args.source or work / 'original.mp4'
        if not args.source:
            urllib.request.urlretrieve(SOURCE, source)
        # Pick an eight-second, caption-free drift, sampled at twelve frames/sec.
        decoded = work / 'decoded'
        decoded.mkdir()
        run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', '7', '-t', '8', '-i', str(source),
            '-vf', f'fps={FPS},scale=960:540', '-an', str(decoded / '%04d.png'))
        font = ImageFont.truetype(str(args.font), 12)
        cell_w, cell_h = WIDTH // COLS, HEIGHT // ROWS
        atlas = []
        for char in RAMP:
            glyph = Image.new('L', (cell_w, cell_h), 255)
            ImageDraw.Draw(glyph).text((cell_w / 2, cell_h / 2), char, font=font, fill=0, anchor='mm')
            # Full black/white glyphs: no blurred grey pixels in the master.
            atlas.append(np.where(np.asarray(glyph) < 190, 0, 255).astype(np.uint8))
        atlas = np.stack(atlas)
        frames = []
        for filename in sorted(decoded.glob('*.png')):
            gray = Image.open(filename).convert('L')
            contrasted = np.clip((np.asarray(gray, dtype=np.float32) - 12) * (255 / 68), 0, 255).astype(np.uint8)
            pixels = Image.fromarray(contrasted).resize((COLS, ROWS), Image.Resampling.BOX)
            indices = np.minimum((np.asarray(pixels, dtype=np.float32) / 255 * (len(RAMP) - 1)).astype(int), len(RAMP) - 1)
            frame = atlas[indices].transpose(0, 2, 1, 3).reshape(HEIGHT, WIDTH)
            frames.append(frame)
        # A cosine-timed forward/reverse loop has zero velocity at both turns.
        # Last and first frames match exactly, eliminating a visible seam.
        total = 16 * FPS
        rendered = work / 'ascii'
        rendered.mkdir()
        for index in range(total):
            phase = (1 - math.cos(2 * math.pi * index / (total - 1))) / 2
            source_index = round(phase * (len(frames) - 1))
            Image.fromarray(frames[source_index]).save(rendered / f'{index:04d}.png')
        poster_index = len(frames) // 2
        Image.fromarray(frames[poster_index]).save(args.output / 'jellyfish-ascii-poster.png', optimize=True)
        # Native decoders do the work at runtime; no per-frame canvas processing.
        common = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(FPS),
                  '-i', str(rendered / '%04d.png'), '-an', '-map_metadata', '-1']
        run(*common, '-c:v', 'libvpx-vp9', '-crf', '40', '-b:v', '0', '-row-mt', '1',
            '-pix_fmt', 'yuv420p', str(args.output / 'jellyfish-ascii.webm'))
        run(*common, '-c:v', 'libx264', '-preset', 'slow', '-crf', '29', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(args.output / 'jellyfish-ascii.mp4'))
        # Visual QA evidence: greyscale, contrast, pixelation, ASCII, and loop samples.
        original = Image.open(sorted(decoded.glob('*.png'))[poster_index]).convert('L')
        contrasted = Image.fromarray(np.clip((np.asarray(original, dtype=np.float32) - 12) * (255 / 68), 0, 255).astype(np.uint8))
        pixelated = contrasted.resize((COLS, ROWS), Image.Resampling.BOX).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)
        stages = [original, contrasted, pixelated, Image.fromarray(frames[poster_index])]
        sheet = Image.new('L', (1280, 720), 255)
        for index, stage in enumerate(stages):
            sheet.paste(stage.resize((640, 360)), ((index % 2) * 640, (index // 2) * 360))
        preview = ROOT / 'previews'
        preview.mkdir(exist_ok=True)
        sheet.save(preview / 'jellyfish-processing.png')
        loop = Image.new('L', (1280, 720), 255)
        for n, index in enumerate([0, 48, 96, 144]):
            loop.paste(Image.open(rendered / f'{index:04d}.png').resize((640, 360)), ((n % 2) * 640, (n // 2) * 360))
        loop.save(preview / 'jellyfish-loop-frames.png')
        report = {
            'source_page': SOURCE_PAGE, 'source_download': SOURCE,
            'credit': 'NOAA Ocean Exploration, Gulf of Mexico 2018',
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_bytes': source.stat().st_size,
            'segment_seconds': [7, 15], 'duration_seconds': 16, 'fps': FPS,
            'size': [WIDTH, HEIGHT], 'ascii_grid': [COLS, ROWS],
            'blackpoint': 12, 'whitepoint': 80, 'glyph_ramp': RAMP,
            'loop': 'cosine-eased forward/reverse; identical first and last master frames',
            'master_first_last_identical': (rendered / '0000.png').read_bytes() == (rendered / f'{total - 1:04d}.png').read_bytes(),
            'outputs': {p.name: p.stat().st_size for p in args.output.glob('jellyfish-ascii*') if p.suffix in {'.mp4', '.webm', '.png'}},
        }
        (args.output / 'jellyfish-ascii.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
