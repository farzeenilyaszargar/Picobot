"""Check that every subject travels broadly, smoothly, and stays inside the frame."""
import importlib.util
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ascii_renderer', root / 'tools/render_ascii_background.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
for name, ((width, height), subjects) in renderer.LAYOUTS.items():
    assert width % renderer.CELL_W == 0 and height % renderer.CELL_H == 0
    for index, (extent, phase, flip) in enumerate(subjects):
        normalized = np.array([renderer.motion_position(t,index,phase,name) for t in np.linspace(0,1,renderer.FPS*renderer.DURATION)])
        points = normalized * [width,height]
        assert np.allclose(points[0],points[-1]), 'Loop has a discontinuity'
        assert np.ptp(normalized[:,0]) > .6, 'Subject remains in one horizontal area'
        assert np.ptp(normalized[:,1]) > .55, 'Subject remains in one vertical area'
        assert (points >= extent/2).all(), 'Subject clips at the top/left edge'
        assert (points <= np.array([width,height])-extent/2).all(), 'Subject clips at the bottom/right edge'
        steps = np.linalg.norm(np.diff(points,axis=0),axis=1)
        assert steps.max() < max(width,height)*.012, 'Motion jumps between frames'
        assert steps.sum() > 2*max(width,height), 'Movement is a local wobble'
    for time in np.linspace(0,1,240):
        xs = [renderer.motion_position(time,i,p[1],name)[0] for i,p in enumerate(subjects)]
        assert min(xs)<.3 and max(xs)>.7, 'The group bunches onto one side of the screen'
    print(f'{name}: broad smooth routes, no clipping, and subjects stay distributed across the screen')
